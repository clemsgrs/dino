import sys
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Dict, Tuple

from dino.distributed import is_main_process, is_enabled_and_multiple_gpus, get_global_size


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from a single model.

    Args:
        model: Model to extract features from (backbone only, no head)
        loader: DataLoader returning (index, image, label)
        device: Device to run inference on

    Returns:
        features: Tensor of shape (N, embed_dim)
        labels: Tensor of shape (N,)
    """
    model.eval()
    distributed = is_enabled_and_multiple_gpus()
    features = None
    labels = []

    with tqdm.tqdm(
        loader,
        desc="Feature extraction",
        unit=" batch",
        ncols=80,
        leave=False,
        file=sys.stdout,
        disable=not is_main_process(),
    ) as t:
        for batch in t:
            index, img, label = batch
            img = img.to(device, non_blocking=True)
            index = index.to(device, non_blocking=True)
            labels.extend(label.clone().tolist())

            feats = model(img).clone()

            # Initialize storage on main process
            if is_main_process() and features is None:
                features = torch.zeros(len(loader.dataset), feats.shape[-1], device=device)

            if distributed:
                ngpu = get_global_size()
                # Gather indices
                y_all = torch.empty(ngpu, index.size(0), dtype=index.dtype, device=device)
                y_l = list(y_all.unbind(0))
                dist.all_gather(y_l, index)
                index_all = torch.cat(y_l)

                # Gather features
                feats_all = torch.empty(
                    ngpu, feats.size(0), feats.size(1),
                    dtype=feats.dtype, device=device
                )
                output_l = list(feats_all.unbind(0))
                dist.all_gather(output_l, feats)

                if is_main_process():
                    features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                if features is None:
                    features = torch.zeros(len(loader.dataset), feats.shape[-1], device=device)
                features[index.cpu().tolist(), :] = feats

    labels = torch.tensor(labels).long()
    return features, labels


@torch.no_grad()
def extract_multiple_features(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract features from both student and teacher models.

    Args:
        student: Student model backbone
        teacher: Teacher model backbone
        loader: DataLoader returning (index, image, label)
        device: Device to run inference on

    Returns:
        features: Dict with 'student' and 'teacher' tensors of shape (N, embed_dim)
        labels: Tensor of shape (N,)
    """
    student.eval()
    teacher.eval()
    distributed = is_enabled_and_multiple_gpus()

    student_features = None
    teacher_features = None
    labels = []

    with tqdm.tqdm(
        loader,
        desc="Feature extraction",
        unit=" batch",
        ncols=80,
        leave=False,
        file=sys.stdout,
        disable=not is_main_process(),
    ) as t:
        for batch in t:
            index, img, label = batch
            img = img.to(device, non_blocking=True)
            index = index.to(device, non_blocking=True)
            labels.extend(label.clone().tolist())

            student_feats = student(img).clone()
            teacher_feats = teacher(img).clone()

            # Initialize storage on main process
            if is_main_process() and student_features is None:
                student_features = torch.zeros(
                    len(loader.dataset), student_feats.shape[-1], device=device
                )
                teacher_features = torch.zeros(
                    len(loader.dataset), teacher_feats.shape[-1], device=device
                )

            if distributed:
                ngpu = get_global_size()

                # Gather indices
                y_all = torch.empty(ngpu, index.size(0), dtype=index.dtype, device=device)
                y_l = list(y_all.unbind(0))
                dist.all_gather(y_l, index)
                index_all = torch.cat(y_l)

                # Gather student features
                student_feats_all = torch.empty(
                    ngpu, student_feats.size(0), student_feats.size(1),
                    dtype=student_feats.dtype, device=device
                )
                student_output_l = list(student_feats_all.unbind(0))
                dist.all_gather(student_output_l, student_feats)

                # Gather teacher features
                teacher_feats_all = torch.empty(
                    ngpu, teacher_feats.size(0), teacher_feats.size(1),
                    dtype=teacher_feats.dtype, device=device
                )
                teacher_output_l = list(teacher_feats_all.unbind(0))
                dist.all_gather(teacher_output_l, teacher_feats)

                if is_main_process():
                    student_features.index_copy_(0, index_all, torch.cat(student_output_l))
                    teacher_features.index_copy_(0, index_all, torch.cat(teacher_output_l))
            else:
                if student_features is None:
                    student_features = torch.zeros(
                        len(loader.dataset), student_feats.shape[-1], device=device
                    )
                    teacher_features = torch.zeros(
                        len(loader.dataset), teacher_feats.shape[-1], device=device
                    )
                idx_list = index.cpu().tolist()
                student_features[idx_list, :] = student_feats
                teacher_features[idx_list, :] = teacher_feats

    labels = torch.tensor(labels).long()
    features = {"student": student_features, "teacher": teacher_features}
    return features, labels
