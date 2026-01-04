import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
import wandb

import dino.models.vision_transformer as vits
from dino.components import DINOLoss
from dino.data import HierarchicalPretrainingDataset, RegionDataAugmentationDINO
from dino.distributed import (
    get_global_rank,
    get_global_size,
    is_enabled_and_multiple_gpus,
    is_main_process,
)
from dino.log import update_log_dict
from dino.models import MultiCropWrapper
from dino.utils import (
    compute_time,
    cosine_scheduler,
    fix_random_seeds,
    get_params_groups,
    has_batchnorms,
    resume_from_checkpoint,
    setup,
    start_from_checkpoint,
    train_one_epoch,
)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("dino", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command using \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):

    cfg = setup(args, level="patch")
    output_dir = Path(cfg.output_dir)

    fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    snapshot_dir = Path(output_dir, "snapshots")
    if not cfg.resume and is_main_process():
        if output_dir.exists():
            print(f"WARNING: {output_dir} already exists! Deleting its content...")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
        snapshot_dir.mkdir(exist_ok=True, parents=True)

    # preparing data
    if is_main_process():
        print("Loading data...")

    transform = RegionDataAugmentationDINO(
        cfg.aug.global_crops_scale,
        cfg.aug.local_crops_number,
        cfg.aug.local_crops_scale,
        cfg.model.input_size,
        cfg.model.patch_size,
    )

    # using custom dataset for our [256 x 384] tensors ("local" features)
    dataset = HierarchicalPretrainingDataset(cfg.data_dir, transform)
    if cfg.training.pct:
        print(f"Pre-training on {cfg.training.pct*100}% of the data")
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)

    if is_enabled_and_multiple_gpus():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.training.batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    if is_main_process():
        print(f"Pretraining data loaded ({len(dataset)} regions)")

    # building student and teacher networks
    if is_main_process():
        print("Building student and teacher networks...")
    student = vits.__dict__[cfg.model.arch](
        img_size=cfg.model.input_size,
        patch_size=cfg.model.patch_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    teacher = vits.__dict__[cfg.model.arch](
        img_size=cfg.model.input_size, patch_size=cfg.model.patch_size
    )
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
            norm_last_layer=cfg.model.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
        ),
    )

    # move networks to gpu
    gpu_id = get_global_rank()
    device = f"cuda:{gpu_id}"
    student, teacher = student.to(device), teacher.to(device)

    # synchronize batch norms (if any)
    if has_batchnorms(student) and is_enabled_and_multiple_gpus():
        # we need DDP wrapper to have synchro batch norms working...
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[gpu_id], output_device=gpu_id
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    if is_enabled_and_multiple_gpus():
        student = nn.parallel.DistributedDataParallel(
            student,
            device_ids=[gpu_id],
            output_device=gpu_id,
        )

    # optionally start student from existing checkpoint
    if cfg.start_from_checkpoint:
        ckpt_path = Path(cfg.start_from_checkpoint)
        start_from_checkpoint(
            ckpt_path,
            student,
        )

    # teacher and student start with the same weights
    student_sd = student.state_dict()
    nn.modules.utils.consume_prefix_in_state_dict_if_present(student_sd, "module.")
    teacher_without_ddp.load_state_dict(student_sd)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # total number of crops = 2 global crops + local_crops_number
    crops_number = cfg.aug.local_crops_number + 2
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        crops_number,
        cfg.model.warmup_teacher_temp,
        cfg.model.teacher_temp,
        cfg.model.warmup_teacher_temp_epochs,
        cfg.training.nepochs,
    )
    dino_loss = dino_loss.to(device)

    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if cfg.speed.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    assert (
        cfg.training.nepochs >= cfg.training.warmup_epochs
    ), f"nepochs ({cfg.training.nepochs}) must be greater than or equal to warmup_epochs ({cfg.training.warmup_epochs})"
    base_lr = (
        cfg.optim.lr * (cfg.training.batch_size_per_gpu * get_global_size()) / 256.0
    )
    lr_schedule = cosine_scheduler(
        base_lr,
        cfg.optim.lr_scheduler.min_lr,
        cfg.training.nepochs,
        len(data_loader),
        warmup_epochs=cfg.training.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        cfg.optim.lr_scheduler.weight_decay,
        cfg.optim.lr_scheduler.weight_decay_end,
        cfg.training.nepochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        cfg.model.momentum_teacher,
        1,
        cfg.training.nepochs,
        len(data_loader),
    )
    if is_main_process():
        print("Models built, kicking off training")

    epochs_run = 0

    # leverage torch native fault tolerance
    snapshot_path = Path(snapshot_dir, "latest.pt")
    if is_enabled_and_multiple_gpus() and snapshot_path.exists():
        if is_main_process():
            print("Loading snapshot")
        snapshot = torch.load(snapshot_path, map_location=device)
        epochs_run = snapshot["epoch"]
        student.load_state_dict(snapshot["student"])
        teacher.load_state_dict(snapshot["teacher"])
        optimizer.load_state_dict(snapshot["optimizer"])
        dino_loss.load_state_dict(snapshot["dino_loss"])
        if fp16_scaler is not None:
            fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
        if is_main_process():
            print(f"Resuming training from snapshot at Epoch {epochs_run}")

    elif cfg.resume:
        ckpt_path = Path(cfg.resume_from_checkpoint)
        epochs_run = resume_from_checkpoint(
            ckpt_path,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        if is_main_process():
            print(f"Resuming training from checkpoint at Epoch {epochs_run}")

    start_time = time.time()

    with tqdm.tqdm(
        range(epochs_run, cfg.training.nepochs),
        desc=("Hierarchical DINO Pretraining"),
        unit=" epoch",
        ncols=100,
        leave=True,
        initial=epochs_run,
        total=cfg.training.nepochs,
        file=sys.stdout,
        position=0,
        disable=not is_main_process(),
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            if cfg.wandb.enable and is_main_process():
                log_dict = {"epoch": epoch}

            if is_enabled_and_multiple_gpus():
                data_loader.sampler.set_epoch(epoch)

            # training one epoch of DINO
            train_stats = train_one_epoch(
                student,
                teacher,
                teacher_without_ddp,
                dino_loss,
                data_loader,
                optimizer,
                lr_schedule,
                wd_schedule,
                momentum_schedule,
                epoch,
                cfg.training.nepochs,
                fp16_scaler,
                cfg.training.clip_grad,
                cfg.training.freeze_last_layer,
                gpu_id,
            )

            if cfg.wandb.enable and is_main_process():
                update_log_dict("train", train_stats, log_dict, step="epoch")

            # save snapshot and log to wandb
            if is_main_process():
                snapshot = {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                }
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()

                save_path = Path(snapshot_dir, f"epoch_{epoch:03}.pt")
                if (
                    cfg.logging.save_snapshot_every
                    and epoch % cfg.logging.save_snapshot_every == 0
                ):
                    torch.save(snapshot, save_path)
                torch.save(snapshot, Path(snapshot_dir, "latest.pt"))

                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if is_main_process():
                with open(Path(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            if is_main_process():
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1}/{cfg.training.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Pretraining time {}".format(total_time_str))

    if is_enabled_and_multiple_gpus():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
