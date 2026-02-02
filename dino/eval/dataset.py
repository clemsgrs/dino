import torch
import pandas as pd

from PIL import Image
from typing import Callable, Optional
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.folder import default_loader


class EvalDataset(torch.utils.data.Dataset):
    """Dataset for evaluation that returns (index, image, label).

    The index is returned to allow proper feature storage during distributed extraction.
    Labels are automatically encoded to integers if they are strings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        image_col: str = "image_path",
        label_col: str = "label",
        label_encoder: Optional[LabelEncoder] = None,
    ):
        """
        Args:
            df: DataFrame with image paths and labels
            transform: Torchvision transforms to apply
            image_col: Column name for image paths
            label_col: Column name for labels
            label_encoder: Optional pre-fit LabelEncoder for consistent label mapping
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.loader = default_loader

        # Handle label encoding
        labels = self.df[label_col].values
        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(labels)
        elif not pd.api.types.is_integer_dtype(self.df[label_col]):
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = None
            self.labels = labels.astype(int)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row[self.image_col]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return idx, image, label

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))
