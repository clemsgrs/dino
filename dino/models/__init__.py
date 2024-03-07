import torch
import torch.nn as nn

from typing import Optional
from pathlib import Path

from dino.utils import update_state_dict
from dino.models.vision_transformer import vit_small


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


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        embed_dim: int = 384,
        mask_attn_patch: bool = False,
        img_size_pretrained: Optional[int] = None,
        verbose: bool = True,
    ):
        super(PatchEmbedder, self).__init__()
        checkpoint_key = "teacher"

        self.vit_patch = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            mask_attn=mask_attn_patch,
            img_size_pretrained=img_size_pretrained,
        )

        if Path(pretrain_vit_patch).is_file():
            if verbose:
                print("Loading pretrained weights for patch-level Transformer")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_patch}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained patch-level Transformer")
        for param in self.vit_patch.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

    def forward(self, x):
        # x = [B, 3, img_size, img_size]
        feature = self.vit_patch(x).detach().cpu()  # [B, 384]
        return feature
