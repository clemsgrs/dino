import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.distributed = torch.cuda.device_count() > 1

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        m = 1
        if self.distributed:
            dist.all_reduce(batch_center)
            m = dist.get_world_size()
        batch_center = batch_center / (len(teacher_output) * m)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class RobustTripletLoss(nn.Module):
    """
    Triplet loss computed per device.

    - Anchor / positive: same label, different sample
    - Anchor / negative: different label

    mining:
        "batch_hard": hardest positive & hardest negative per anchor
        "batch_all" : all valid (anchor, positive, negative) triplets in batch
    Returns:
        (loss, num_valid_anchors)
    """
    def __init__(
        self,
        margin: float = 0.2,
        mining: str = "batch_hard",
    ):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def _pairwise_distances(self, embeddings: Tensor) -> Tensor:
        """
        Pairwise Euclidean distances.
        embeddings: [B, D]
        returns: [B, B]
        """
        sq_norm = (embeddings ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dist = sq_norm + sq_norm.t() - 2.0 * embeddings @ embeddings.t()
        dist.clamp_min_(1e-16)
        return torch.sqrt(dist)

    def _triplet_loss_batch_hard(
        self,
        *,
        dist_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        embeddings_dtype: torch.dtype,
    ) -> tuple[Tensor, int]:
        """
        Batch-hard triplet loss:
        - Hardest positive: max distance among positives for each anchor
        - Hardest negative: min distance among negatives for each anchor
        """
        device = dist_matrix.device

        # Hardest positive: max distance among positives
        # Mask non-positives with -1.0 (since distances >= 0)
        dist_pos = dist_matrix.clone()
        dist_pos[~positive_mask] = -1.0
        hardest_positive_dist, _ = dist_pos.max(dim=1)

        # Hardest negative: min distance among negatives
        # Mask non-negatives with infinity
        dist_neg = dist_matrix.clone()
        dist_neg[~negative_mask] = float('inf')
        hardest_negative_dist, _ = dist_neg.min(dim=1)

        # Valid anchors must have at least one positive and one negative
        valid_anchors_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        num_valid_anchors = valid_anchors_mask.sum().item()

        if num_valid_anchors == 0:
            zero = torch.zeros((), device=device, dtype=embeddings_dtype)
            return {"loss": zero, "num_valid_anchors": 0}

        # Compute loss per anchor
        triplet_loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        # Average over valid anchors
        loss = triplet_loss[valid_anchors_mask].mean()

        return {"loss": loss, "num_valid_anchors": num_valid_anchors}

    def _triplet_loss_batch_all(
        self,
        *,
        dist_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        embeddings_dtype: torch.dtype,
    ) -> tuple[Tensor, int]:
        """
        Batch-all triplet loss:
        Uses all (anchor, positive, negative) triplets in the batch.

        For each anchor i:
            j: positives (same organ)
            k: negatives (different organ)
        We compute relu(d(i,j) - d(i,k) + margin) for all valid (j, k), then average
        over valid anchors.
        """
        device = dist_matrix.device
        B = dist_matrix.shape[0]

        # [B, B, 1] and [B, 1, B]
        dist_ap = dist_matrix.unsqueeze(2)  # i,j,1
        dist_an = dist_matrix.unsqueeze(1)  # i,1,k

        # Masks: [B, B, 1] for positives, [B, 1, B] for negatives
        pos_mask_3d = positive_mask.unsqueeze(2)  # i,j,1
        neg_mask_3d = negative_mask.unsqueeze(1)  # i,1,k

        # Valid triplets mask: i,j,k
        triplet_mask = pos_mask_3d & neg_mask_3d  # [B, B, B]

        # Raw triplet losses: [B, B, B]
        triplet_losses = dist_ap - dist_an + self.margin
        triplet_losses = torch.relu(triplet_losses)

        # Zero out invalid triplets
        triplet_losses = triplet_losses * triplet_mask

        # Count non-zero triplets per anchor
        per_anchor_triplet_count = triplet_mask.view(B, -1).sum(dim=1)  # [B]
        valid_anchor_mask = per_anchor_triplet_count > 0

        num_valid_anchors = int(valid_anchor_mask.sum().item())
        if num_valid_anchors == 0:
            zero = torch.zeros((), device=device, dtype=embeddings_dtype)
            return {"loss": zero, "num_valid_anchors": 0}

        # Sum over j,k then average per valid anchor
        sum_per_anchor = triplet_losses.view(B, -1).sum(dim=1)  # [B]
        avg_per_anchor = torch.zeros_like(sum_per_anchor)
        avg_per_anchor[valid_anchor_mask] = (
            sum_per_anchor[valid_anchor_mask] / per_anchor_triplet_count[valid_anchor_mask]
        )

        loss = avg_per_anchor[valid_anchor_mask].mean()
        return {"loss": loss, "num_valid_anchors": num_valid_anchors}

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, int]:

        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            zero = torch.zeros((), device=device, dtype=embeddings.dtype)
            return {"loss": zero, "num_valid_anchors": 0}

        labels = labels.view(-1).to(device=device)
        assert labels.shape[0] == B, "organ_labels must have shape [B]"

        dist_matrix = self._pairwise_distances(embeddings)  # [B, B]

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        positive_mask = labels_eq & ~eye
        negative_mask = ~labels_eq

        if self.mining == "batch_all":
            return self._organ_triplet_loss_batch_all(
                dist_matrix=dist_matrix,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
                embeddings_dtype=embeddings.dtype,
            )
        else:
            # default: batch_hard
            return self._organ_triplet_loss_batch_hard(
                dist_matrix=dist_matrix,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
                embeddings_dtype=embeddings.dtype,
            )


class OrganTripletLoss(nn.Module):
    """
    Organ-level triplet loss computed per device.

    - Anchor / positive: same organ label, different sample
    - Anchor / negative: different organ label

    mining:
        "batch_hard": hardest positive & hardest negative per anchor
        "batch_all" : all valid (anchor, positive, negative) triplets in batch
    Returns:
        (loss, num_valid_anchors)
    """
    def __init__(
        self,
        margin: float = 0.2,
        mining: str = "batch_hard",
    ):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def _pairwise_distances(self, embeddings: Tensor) -> Tensor:
        """
        Pairwise Euclidean distances.
        embeddings: [B, D]
        returns: [B, B]
        """
        sq_norm = (embeddings ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dist = sq_norm + sq_norm.t() - 2.0 * embeddings @ embeddings.t()
        dist.clamp_min_(1e-16)
        return torch.sqrt(dist)

    def _organ_triplet_loss_batch_hard(
        self,
        *,
        dist_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        embeddings_dtype: torch.dtype,
    ) -> tuple[Tensor, int]:
        """
        Batch-hard triplet loss:
        - Hardest positive: max distance among positives for each anchor
        - Hardest negative: min distance among negatives for each anchor
        """
        device = dist_matrix.device

        # Hardest positive: max distance among positives
        # Mask non-positives with -1.0 (since distances >= 0)
        dist_pos = dist_matrix.clone()
        dist_pos[~positive_mask] = -1.0
        hardest_positive_dist, _ = dist_pos.max(dim=1)

        # Hardest negative: min distance among negatives
        # Mask non-negatives with infinity
        dist_neg = dist_matrix.clone()
        dist_neg[~negative_mask] = float('inf')
        hardest_negative_dist, _ = dist_neg.min(dim=1)

        # Valid anchors must have at least one positive and one negative
        valid_anchors_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        num_valid_anchors = valid_anchors_mask.sum().item()

        if num_valid_anchors == 0:
            zero = torch.zeros((), device=device, dtype=embeddings_dtype)
            return {"loss": zero, "num_valid_anchors": 0}

        # Compute loss per anchor
        triplet_loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        # Average over valid anchors
        loss = triplet_loss[valid_anchors_mask].mean()

        return {"loss": loss, "num_valid_anchors": num_valid_anchors}

    def _organ_triplet_loss_batch_all(
        self,
        *,
        dist_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        embeddings_dtype: torch.dtype,
    ) -> tuple[Tensor, int]:
        """
        Batch-all triplet loss:
        Uses all (anchor, positive, negative) triplets in the batch.

        For each anchor i:
            j: positives (same organ)
            k: negatives (different organ)
        We compute relu(d(i,j) - d(i,k) + margin) for all valid (j, k), then average
        over valid anchors.
        """
        device = dist_matrix.device
        B = dist_matrix.shape[0]

        # [B, B, 1] and [B, 1, B]
        dist_ap = dist_matrix.unsqueeze(2)  # i,j,1
        dist_an = dist_matrix.unsqueeze(1)  # i,1,k

        # Masks: [B, B, 1] for positives, [B, 1, B] for negatives
        pos_mask_3d = positive_mask.unsqueeze(2)  # i,j,1
        neg_mask_3d = negative_mask.unsqueeze(1)  # i,1,k

        # Valid triplets mask: i,j,k
        triplet_mask = pos_mask_3d & neg_mask_3d  # [B, B, B]

        # Raw triplet losses: [B, B, B]
        triplet_losses = dist_ap - dist_an + self.margin
        triplet_losses = torch.relu(triplet_losses)

        # Zero out invalid triplets
        triplet_losses = triplet_losses * triplet_mask

        # Count non-zero triplets per anchor
        per_anchor_triplet_count = triplet_mask.view(B, -1).sum(dim=1)  # [B]
        valid_anchor_mask = per_anchor_triplet_count > 0

        num_valid_anchors = int(valid_anchor_mask.sum().item())
        if num_valid_anchors == 0:
            zero = torch.zeros((), device=device, dtype=embeddings_dtype)
            return {"loss": zero, "num_valid_anchors": 0}

        # Sum over j,k then average per valid anchor
        sum_per_anchor = triplet_losses.view(B, -1).sum(dim=1)  # [B]
        avg_per_anchor = torch.zeros_like(sum_per_anchor)
        avg_per_anchor[valid_anchor_mask] = (
            sum_per_anchor[valid_anchor_mask] / per_anchor_triplet_count[valid_anchor_mask]
        )

        loss = avg_per_anchor[valid_anchor_mask].mean()
        return {"loss": loss, "num_valid_anchors": num_valid_anchors}

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, int]:

        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            zero = torch.zeros((), device=device, dtype=embeddings.dtype)
            return zero, 0

        labels = labels.view(-1).to(device=device)
        assert labels.shape[0] == B, "organ_labels must have shape [B]"

        dist_matrix = self._pairwise_distances(embeddings)  # [B, B]

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        positive_mask = labels_eq & ~eye
        negative_mask = ~labels_eq

        if self.mining == "batch_all":
            return self._organ_triplet_loss_batch_all(
                dist_matrix=dist_matrix,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
                embeddings_dtype=embeddings.dtype,
            )
        else:
            # default: batch_hard
            return self._organ_triplet_loss_batch_hard(
                dist_matrix=dist_matrix,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
                embeddings_dtype=embeddings.dtype,
            )


class OrganTripletMarginLoss(nn.TripletMarginLoss):
    """
    Wrapper around nn.TripletMarginLoss that performs extra mining step.

    Modes:
    - "batch_hard": Mines hardest positive and negative for each anchor.
      Uses nn.TripletMarginLoss for the final computation.
    - "batch_all": Computes loss for all valid triplets.
      Does NOT use nn.TripletMarginLoss forward pass directly due to memory constraints
      of materializing all triplets, but implements the equivalent logic with broadcasting.
    """
    def __init__(
        self,
        margin: float = 0.2,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: str = 'mean',
        mining: str = "batch_hard"
    ):
        super().__init__(margin=margin, p=p, eps=eps, swap=swap, reduction=reduction)
        self.mining = mining

    def forward(self, embeddings: Tensor, labels: Tensor) -> tuple[Tensor, int]:
        """
        Args:
            embeddings: [B, D]
            labels: [B]
        Returns:
            (loss, num_valid_anchors)
        """
        device = embeddings.device
        B = embeddings.shape[0]

        # 1. Compute pairwise distances for mining
        # torch.cdist computes p-norm distance (default p=2 is Euclidean)
        dist_matrix = torch.cdist(embeddings, embeddings, p=self.p)

        # 2. Create masks
        labels = labels.view(-1).to(device)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)

        positive_mask = labels_eq & ~eye
        negative_mask = ~labels_eq

        if self.mining == "batch_all":
            return self._batch_all_loss(dist_matrix, positive_mask, negative_mask)

        # --- Batch Hard Logic ---

        # 3. Identify valid anchors (must have at least 1 pos and 1 neg)
        valid_anchors_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        valid_indices = valid_anchors_mask.nonzero(as_tuple=False).view(-1)
        num_valid_anchors = valid_indices.numel()

        if num_valid_anchors == 0:
            # Return zero loss with gradient requirement if needed
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        # 4. Mine Hardest Triplets
        # For positives: we want max dist. Set invalid to -1 (since dist >= 0)
        dist_pos = dist_matrix.clone()
        dist_pos[~positive_mask] = -1.0
        hardest_pos_idx = dist_pos.argmax(dim=1) # [B]

        # For negatives: we want min dist. Set invalid to inf
        dist_neg = dist_matrix.clone()
        dist_neg[~negative_mask] = float('inf')
        hardest_neg_idx = dist_neg.argmin(dim=1) # [B]

        # 5. Gather embeddings for valid anchors only
        # We select the anchor, its hardest positive, and its hardest negative
        anchor_embeddings = embeddings[valid_indices]
        positive_embeddings = embeddings[hardest_pos_idx[valid_indices]]
        negative_embeddings = embeddings[hardest_neg_idx[valid_indices]]

        # 6. Compute standard TripletMarginLoss
        loss = super().forward(anchor_embeddings, positive_embeddings, negative_embeddings)

        return {"loss": loss, "num_valid_anchors": num_valid_anchors}

    def _batch_all_loss(self, dist_matrix, positive_mask, negative_mask):
        """
        Computes batch-all triplet loss using broadcasting to avoid
        materializing B^3 triplets.
        """
        device = dist_matrix.device
        B = dist_matrix.shape[0]

        # [B, B, 1] and [B, 1, B]
        dist_ap = dist_matrix.unsqueeze(2)  # i,j,1
        dist_an = dist_matrix.unsqueeze(1)  # i,1,k

        # Masks: [B, B, 1] for positives, [B, 1, B] for negatives
        pos_mask_3d = positive_mask.unsqueeze(2)  # i,j,1
        neg_mask_3d = negative_mask.unsqueeze(1)  # i,1,k

        # Valid triplets mask: i,j,k
        triplet_mask = pos_mask_3d & neg_mask_3d  # [B, B, B]

        # Raw triplet losses: [B, B, B]
        # loss = relu(d(a,p) - d(a,n) + margin)
        triplet_losses = dist_ap - dist_an + self.margin
        triplet_losses = torch.relu(triplet_losses)

        # Zero out invalid triplets
        triplet_losses = triplet_losses * triplet_mask

        # Count non-zero triplets per anchor
        per_anchor_triplet_count = triplet_mask.view(B, -1).sum(dim=1)  # [B]
        valid_anchor_mask = per_anchor_triplet_count > 0
        num_valid_anchors = int(valid_anchor_mask.sum().item())

        if num_valid_anchors == 0:
             return torch.tensor(0.0, device=device, requires_grad=True), 0

        # Sum over j,k then average per valid anchor
        sum_per_anchor = triplet_losses.view(B, -1).sum(dim=1)
        avg_per_anchor = torch.zeros_like(sum_per_anchor)
        avg_per_anchor[valid_anchor_mask] = (
            sum_per_anchor[valid_anchor_mask] / per_anchor_triplet_count[valid_anchor_mask]
        )

        # Final reduction (assuming mean)
        if self.reduction == 'mean':
            loss = avg_per_anchor[valid_anchor_mask].mean()
        elif self.reduction == 'sum':
            loss = avg_per_anchor[valid_anchor_mask].sum()
        else:
            # For 'none', we'd ideally return the per-anchor loss, but the signature
            # expects a scalar usually. We'll return mean here to be safe or
            # the user should handle the shape.
            loss = avg_per_anchor

        return {"loss": loss, "num_valid_anchors": num_valid_anchors}
