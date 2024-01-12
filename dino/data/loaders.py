from typing import Callable, Optional

from .datasets import (
    ImageNet,
    ImageNet22k,
    PathologyDataset,
    PathologyFoundationDataset,
    KNNDataset,
)


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split", "subset")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "Pathology":
        class_ = PathologyDataset
        if "subset" in kwargs:
            subset = kwargs["subset"]
            kwargs["subset"] = PathologyDataset.Subset(subset)
    elif name == "PathologyFoundation":
        class_ = PathologyFoundationDataset
        if "subset" in kwargs:
            subset = kwargs["subset"]
            kwargs["subset"] = PathologyFoundationDataset.Subset(subset)
    elif name == "KNN":
        class_ = KNNDataset
        if "split" in kwargs:
            kwargs["split"] = KNNDataset.Split[kwargs["split"]]
        if "subset" in kwargs:
            subset = kwargs["subset"]
            kwargs["subset"] = KNNDataset.Subset(subset)
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    verbose: bool = True,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    if verbose:
        print(f'Using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    if verbose:
        print(f"# of dataset samples: {len(dataset):,d}\n")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset
