import pandas as pd
from pathlib import Path
from torchvision import datasets
from typing import Callable, Optional


class ImageFolderWithFilenameDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform,
        )

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[idx]
        fname = Path(path).stem
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, fname


class ImageFolderWithMetadata(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        label: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform,
        )
        self.df = df
        self.label = label
        # map label unique values to integers
        unique_labels = self.df[self.label].unique().tolist()
        self.num_classes = len(unique_labels)
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        # add case_id column to dataframe if not present
        if "case_id" not in self.df.columns:
            self.df["case_id"] = self.df.wsi_path.apply(lambda x: Path(x).stem)

        self.case_id_to_label = dict(zip(self.df["case_id"], self.df[self.label]))

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[idx]
        fname = Path(path).stem
        case_id = "_".join(fname.split("_")[:-2])
        label = self.case_id_to_label[case_id]
        label_class = self.label2idx[label]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label_class