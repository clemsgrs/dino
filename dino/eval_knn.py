import os
import tqdm
import hydra
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pathlib import Path
from typing import Optional
from sklearn import metrics
from omegaconf import DictConfig
from torchvision import transforms

import dino.models.vision_transformer as vits
from dino.data import ImagePretrainingDataset
from dino.log import initialize_wandb
from dino.distributed import is_main_process


class ReturnIndexDataset(ImagePretrainingDataset):
    def __getitem__(self, idx):
        img, label = super(ReturnIndexDataset, self).__getitem__(idx)
        return idx, img, label


def prepare_data(
    query_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size_per_gpu,
    distributed,
    num_workers,
    label_name: Optional[str] = None,
):
    # ============ preparing data ... ============
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    query_dataset = ReturnIndexDataset(
        query_df, transform=transform, label_name=label_name
    )
    test_dataset = ReturnIndexDataset(
        test_df, transform=transform, label_name=label_name
    )
    if distributed:
        sampler = torch.utils.data.DistributedSampler(query_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(query_dataset)
    query_data_loader = torch.utils.data.DataLoader(
        query_dataset,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return query_data_loader, test_data_loader


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if Path(pretrained_weights).is_file():
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
    else:
        print("WARNING: provided weights don't exist, random weights will be used.")


def multi_scale(samples, model):
    v = None
    for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(
                samples, scale_factor=s, mode="bilinear", align_corners=False
            )
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


def extract_feature_pipeline(
    query_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_dir: str,
    arch: str,
    input_size: int,
    patch_size: int,
    pretrained_weights: str,
    checkpoint_key: str,
    batch_size_per_gpu: int,
    distributed: bool,
    save_features: bool = False,
    use_cuda: bool = True,
    num_workers: int = 10,
    label_name: Optional[str] = None,
):
    # ============ preparing data ... ============
    query_data_loader, test_data_loader = prepare_data(
        query_df,
        test_df,
        batch_size_per_gpu,
        distributed,
        num_workers,
        label_name,
    )
    print(
        f"Data loaded with {len(query_data_loader.dataset)} query and {len(test_data_loader.dataset)} eval imgs."
    )

    # ============ building network ... ============
    model = vits.__dict__[arch](img_size=input_size, patch_size=patch_size, num_classes=0)
    print(f"Model {arch} {patch_size}x{patch_size} built.")
    model.cuda()
    print("Loading pretrained weights...")
    load_pretrained_weights(model, pretrained_weights, checkpoint_key)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for query set...")
    query_features, query_labels = extract_features(
        model, query_data_loader, distributed, use_cuda
    )
    print("Extracting features for test set...")
    test_features, test_labels = extract_features(
        model, test_data_loader, distributed, use_cuda
    )

    if is_main_process():
        query_features = nn.functional.normalize(query_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # save features and labels
    if save_features and is_main_process():
        torch.save(query_features.cpu(), Path(features_dir, "query_feat.pt"))
        torch.save(test_features.cpu(), Path(features_dir, "test_feat.pt"))
        torch.save(query_labels.cpu(), Path(features_dir, "query_labels.pt"))
        torch.save(test_labels.cpu(), Path(features_dir, "test_labels.pt"))
    return query_features, test_features, query_labels, test_labels


@torch.no_grad()
def extract_features(model, loader, distributed, use_cuda=True, multiscale=False):
    features = None
    labels = []

    with tqdm.tqdm(
        loader,
        desc=("Feature extraction"),
        unit=" slide",
        ncols=80,
        unit_scale=loader.batch_size,
        leave=True,
    ) as t:
        for i, batch in enumerate(t):
            index, img, label = batch
            img = img.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            labels.extend(label.clone().tolist())
            if multiscale:
                feats = multi_scale(img, model)
            else:
                feats = model(img).clone()

            # init storage feature matrix
            if is_main_process() and features is None:
                features = torch.zeros(len(loader.dataset), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                t.display(
                    f"Storing features into tensor of shape {features.shape}", pos=1
                )
                print()

            if distributed:
                ngpu = dist.get_world_size()
                y_all = torch.empty(
                    ngpu, index.size(0), dtype=index.dtype, device=index.device
                )
                y_l = list(y_all.unbind(0))
                y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
                y_all_reduce.wait()
                index_all = torch.cat(y_l)

                # share features between processes
                feats_all = torch.empty(
                    ngpu,
                    feats.size(0),
                    feats.size(1),
                    dtype=feats.dtype,
                    device=feats.device,
                )
                output_l = list(feats_all.unbind(0))
                output_all_reduce = torch.distributed.all_gather(
                    output_l, feats, async_op=True
                )
                output_all_reduce.wait()

                # update storage feature matrix
                if is_main_process():
                    if use_cuda:
                        features.index_copy_(0, index_all, torch.cat(output_l))
                    else:
                        features.index_copy_(
                            0, index_all.cpu(), torch.cat(output_l).cpu()
                        )
            else:
                features[list(index), :] = feats

    labels = torch.tensor(labels).long()

    return features, labels


@torch.no_grad()
def knn_classifier(
    query_features, query_labels, test_features, test_labels, k, T, num_classes
):
    acc, total = 0.0, 0
    test_probs = np.empty((0, num_classes))
    query_features = query_features.t()
    num_test_images, num_chunks = test_labels.shape[0], min(test_labels.shape[0], 100)
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(query_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        # the use of min ensures we don't compute features more than once if num_test_images is not divisible by num_chunks
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, query_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = query_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        p = (probs / probs.sum(dim=-1).unsqueeze(-1)).cpu().detach().numpy()
        test_probs = np.append(test_probs, p, axis=0)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        acc = acc + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

    acc = acc * 100.0 / total
    if num_classes == 2:
        auc = metrics.roc_auc_score(test_labels.cpu(), test_probs[:, 1])
    else:
        auc = metrics.roc_auc_score(test_labels.cpu(), test_probs, multi_class="ovr")

    return acc, auc


def load_features_and_labels_from_disk(
    df, features_dir, label_name: str = "label", header: str = "query"
):
    all_feature_paths = [fp for fp in features_dir.glob("*.pt")]
    df["stem"] = df.filename.apply(lambda x: Path(x).stem)
    feature_paths = [fp for fp in all_feature_paths if fp.stem in df.stem.values]

    features, labels = [], []
    with tqdm.tqdm(
        feature_paths,
        desc=f"Loading {header} features from disk",
        unit=" img",
        leave=True,
    ) as t:
        for fp in t:
            f = torch.load(fp)
            features.append(f)
            label = df[df.stem == fp.stem][label_name].values[0]
            labels.append(label)

    features = torch.stack(features)
    features = nn.functional.normalize(features, dim=1, p=2)
    labels = torch.tensor(labels).long()

    return features, labels


@hydra.main(
    version_base="1.2.0",
    config_path="config",
    config_name="knn",
)
def main(cfg: DictConfig):
    run_distributed = torch.cuda.device_count() > 1
    if run_distributed:
        torch.distributed.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        if gpu_id == 0:
            print("Distributed session successfully initialized")
    else:
        gpu_id = -1

    if is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("processed", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    cudnn.benchmark = True

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    if is_main_process():
        if output_dir.exists():
            print(f"{output_dir} already exists!")
        output_dir.mkdir(parents=True, exist_ok=True)

    query_df = pd.read_csv(cfg.data.query_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    if cfg.data.features_dir is not None:
        query_features, query_labels = load_features_and_labels_from_disk(
            query_df, Path(cfg.data.features_dir), header="query"
        )
        test_features, test_labels = load_features_and_labels_from_disk(
            test_df, Path(cfg.data.features_dir), header="test"
        )
    else:
        # need to extract features !
        features_dir = Path(output_dir, "features")
        if is_main_process():
            features_dir.mkdir(exist_ok=True, parents=True)
        if is_main_process():
            features_dir.mkdir(exist_ok=True, parents=True)
        (
            query_features,
            test_features,
            query_labels,
            test_labels,
        ) = extract_feature_pipeline(
            query_df,
            test_df,
            features_dir,
            cfg.model.arch,
            cfg.model.input_size,
            cfg.model.patch_size,
            cfg.model.pretrained_weights,
            cfg.model.checkpoint_key,
            cfg.batch_size_per_gpu,
            run_distributed,
            save_features=cfg.save_features,
            use_cuda=cfg.speed.use_cuda,
            num_workers=cfg.speed.num_workers,
            label_name=cfg.label_name,
        )

    if is_main_process():
        assert len(torch.unique(query_labels)) == len(
            torch.unique(test_labels)
        ), "query & test dataset have different number of classes!"
        num_classes = len(torch.unique(query_labels))
        if cfg.speed.use_cuda:
            query_features, query_labels = query_features.cuda(), query_labels.cuda()
            test_features, test_labels = test_features.cuda(), test_labels.cuda()

        print("Features are ready!\nStarting kNN classification.")
        for k in cfg.nb_knn:
            acc, auc = knn_classifier(
                query_features,
                query_labels,
                test_features,
                test_labels,
                k,
                cfg.temperature,
                num_classes,
            )
            print(f"{k}-NN classifier result:")
            print(f"- auc: {auc}")
            print(f"- accuracy: {acc:.2f}%")

    if run_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
