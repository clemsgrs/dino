from typing import List, Tuple, Union

import torch
import torch.nn as nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class MultiCropWrapperWithFeatures(nn.Module):
    """MultiCropWrapper that also returns intermediate features (CLS tokens).

    Used for domain adversarial training where we need access to the backbone
    features before the projection head.
    """

    def __init__(self, backbone, head):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(
        self, x, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional feature extraction.

        Args:
            x: List of image tensors at different resolutions.
            return_features: If True, also return CLS token features before head.

        Returns:
            If return_features=False: head output [total_crops, out_dim]
            If return_features=True: (head_output, features) where features is
                [total_crops, embed_dim]
        """
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        output = torch.empty(0).to(x[0].device)
        features = torch.empty(0).to(x[0].device) if return_features else None

        for end_idx in idx_crops:
            _feat = self.backbone(torch.cat(x[start_idx:end_idx]))
            if isinstance(_feat, tuple):
                _feat = _feat[0]
            if return_features:
                features = torch.cat((features, _feat))
            _out = self.head(_feat)
            output = torch.cat((output, _out))
            start_idx = end_idx

        if return_features:
            return output, features
        return output
