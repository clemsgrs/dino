import torch
import pandas as pd

from PIL import Image
from pathlib import Path
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


class CenterAwareImageFolder(torch.utils.data.Dataset):
    """ImageFolder variant that extracts center/domain labels from directory structure.

    Expected structure: data_dir/center_A/img1.png, data_dir/center_B/img2.png, etc.

    Args:
        root: Root directory containing center subdirectories.
        transform: Transform to apply to images.
        loader: Image loader function.

    Attributes:
        centers: Sorted list of center names.
        center_to_idx: Mapping from center name to index.
        num_centers: Number of unique centers.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        self.root = Path(root)
        self.transform = transform
        self.loader = loader

        # Build center mapping from subdirectories
        self.centers = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if len(self.centers) == 0:
            raise ValueError(f"No subdirectories found in {root}")

        self.center_to_idx = {c: i for i, c in enumerate(self.centers)}
        self.num_centers = len(self.centers)

        # Build sample list: (path, center_idx)
        self.samples = []
        for center in self.centers:
            center_dir = self.root / center
            for img_path in center_dir.iterdir():
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(img_path), self.center_to_idx[center]))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")

    def __getitem__(self, idx: int):
        path, center_idx = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, center_idx

    def __len__(self):
        return len(self.samples)
