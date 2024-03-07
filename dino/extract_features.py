import os
import tqdm
import wandb
import torch
import argparse
import datetime
import pandas as pd
import multiprocessing as mp

from pathlib import Path

from dino.models import PatchEmbedder
from dino.log import initialize_wandb
from dino.distributed import is_main_process
from dino.data import ImageFolderWithNameDataset, make_classification_eval_transform
from dino.utils.config import get_cfg_from_args, write_config


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINO training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--level", type=str, default="patch")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def main(args):
    cfg = get_cfg_from_args(args)

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

    if run_distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{gpu_id}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    cfg.output_dir = str(output_dir)

    features_dir = Path(output_dir, "features")
    if is_main_process():
        if output_dir.exists():
            print(f"{output_dir} already exists! deleting it...")
        output_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(exist_ok=True)

    if is_main_process():
        write_config(cfg, cfg.output_dir)

    model = PatchEmbedder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        pretrain_vit_patch=cfg.pretrain_vit_patch,
        verbose=(gpu_id in [-1, 0]),
    )

    transform = make_classification_eval_transform()
    dataset = ImageFolderWithNameDataset(cfg.data_dir, transform)

    if run_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    num_workers = min(mp.cpu_count(), cfg.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    if gpu_id == -1:
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device, non_blocking=True)

    if is_main_process():
        print()

    filenames, feature_paths = [], []

    with tqdm.tqdm(
        loader,
        desc="Feature Extraction",
        unit=" img",
        ncols=80,
        unit_scale=cfg.batch_size,
        position=0,
        leave=True,
        disable=not (gpu_id in [-1, 0]),
    ) as t1:
        with torch.no_grad():
            for i, batch in enumerate(t1):
                imgs, fnames = batch
                imgs = imgs.to(device, non_blocking=True)
                features = model(imgs)
                for k, f in enumerate(features):
                    fname = fnames[k]
                    feature_path = Path(features_dir, f"{fname}.pt")
                    torch.save(f, feature_path)
                    filenames.append(fname)
                    feature_paths.append(feature_path)
                if cfg.wandb.enable and not run_distributed:
                    wandb.log({"processed": i + imgs.shape[0]})

    features_df = pd.DataFrame.from_dict(
        {
            "filename": filenames,
            "feature_path": feature_paths,
        }
    )

    if run_distributed:
        features_csv_path = Path(output_dir, f"features_{gpu_id}.csv")
    else:
        features_csv_path = Path(output_dir, "features.csv")
    features_df.to_csv(features_csv_path, index=False)

    if run_distributed:
        torch.distributed.barrier()
        if is_main_process():
            dfs = []
            for gpu_id in range(torch.cuda.device_count()):
                fp = Path(output_dir, f"features_{gpu_id}.csv")
                df = pd.read_csv(fp)
                dfs.append(df)
                os.remove(fp)
            features_df = pd.concat(dfs, ignore_index=True)
            features_df = features_df.drop_duplicates()
            features_df.to_csv(Path(output_dir, "features.csv"), index=False)

    if cfg.wandb.enable and is_main_process() and run_distributed:
        wandb.log({"processed": len(features_df)})


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
