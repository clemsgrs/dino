import torch
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision import datasets
from typing import Callable, Optional, Any
from torchvision.datasets.folder import default_loader



def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class ImagePretrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        label_name: Optional[str] = None,
    ):
        self.df = tiles_df
        self.transform = transform
        self.loader = loader
        self.label_name = label_name

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        path = row.tile_path
        tile = self.loader(path)
        if self.transform is not None:
            tile = self.transform(tile)
        label = -1
        if self.label_name is not None:
            label = row[self.label_name]
        return tile, label

    def __len__(self):
        return len(self.df)


class HierarchicalPretrainingDataset(torch.utils.data.Dataset):
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
