import torch

from PIL import Image
from pathlib import Path
from typing import Callable


def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features_dir: str,
        transform: Callable,
    ):
        self.features_list = [f for f in Path(features_dir).glob("*.pt")]
        self.transform = transform

    def __getitem__(self, idx: int):
        f = torch.load(self.features_list[idx])
        f = self.transform(f)
        label = torch.zeros(1, 1)
        return f, label

    def __len__(self):
        return len(self.features_list)
