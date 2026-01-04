import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
import wandb
from torchvision import datasets

import dino.models.vision_transformer as vits
from dino.components import DINOLoss, EarlyStoppingDINO, OrganTripletLoss, OrganTripletMarginLoss
from dino.data import PatchDataAugmentationDINO, ImageFolderWithMetadata
from dino.distributed import (
    get_global_rank,
    get_global_size,
    is_enabled_and_multiple_gpus,
    is_main_process,
)
from dino.eval import prepare_data
from dino.log import update_log_dict
from dino.models import MultiCropWrapper, DomainAdversary
from dino.utils import (
    compute_time,
    cosine_scheduler,
    fix_random_seeds,
    get_params_groups,
    has_batchnorms,
    resume_from_checkpoint,
    setup,
    train_one_epoch,
    tune_one_epoch,
)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("dino", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
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
    features_dir = Path(output_dir, "features")
    if not cfg.resume and is_main_process():
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        if cfg.early_stopping.tune_every and cfg.early_stopping.knn.save_features:
            features_dir.mkdir(exist_ok=True, parents=True)

    # preparing data
    if is_main_process():
        print("Loading data...")

    # ============ preparing tuning data ============
    if is_main_process() and cfg.early_stopping.tune_every:
        # only do it from master rank as tuning is not being run distributed for now
        query_df = pd.read_csv(cfg.early_stopping.downstream.query_csv)
        test_df = pd.read_csv(cfg.early_stopping.downstream.test_csv)

        num_workers = min(mp.cpu_count(), cfg.early_stopping.downstream.num_workers)
        if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
            num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

        downstream_query_loader, downstream_test_loader = prepare_data(
            query_df,
            test_df,
            cfg.early_stopping.downstream.batch_size_per_gpu,
            False,
            num_workers,
            cfg.early_stopping.downstream.label_name,
        )
        print(
            f"Tuning data loaded with {len(downstream_query_loader.dataset)} query patches and {len(downstream_test_loader.dataset)} test patches."
        )

    transform = PatchDataAugmentationDINO(
        cfg.aug.global_crop_size,
        cfg.aug.local_crop_size,
        cfg.aug.global_crops_scale,
        cfg.aug.local_crops_scale,
        cfg.aug.local_crops_number,
        solarization=cfg.aug.solarization,
    )

    # ============ preparing training data ============
    dataset_loading_start_time = time.time()
    if cfg.loss.domain_adv.enabled:
        df = pd.read_csv(cfg.loss.domain_adv.csv)
        dataset = ImageFolderWithMetadata(cfg.data_dir, df=df, label=cfg.loss.domain_adv.label_name, transform=transform)
    else:
        dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    dataset_loading_end_time = time.time() - dataset_loading_start_time
    total_time_str = str(datetime.timedelta(seconds=int(dataset_loading_end_time)))
    if is_main_process():
        print(f"Pretraining data loaded in {total_time_str} ({len(dataset)} patches)")

    if cfg.training.pct:
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)
        if is_main_process():
            print(
                f"Pretraining on {cfg.training.pct*100}% of the data: {len(dataset):,d} samples\n"
            )

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

    # building student and teacher networks
    if is_main_process():
        print("Building student and teacher networks...")
    student = vits.__dict__[cfg.model.arch](
        img_size=cfg.model.input_size,
        patch_size=cfg.model.patch_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    teacher = vits.__dict__[cfg.model.arch](img_size=cfg.model.input_size, patch_size=cfg.model.patch_size)
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
        return_backbone=cfg.loss.domain_adv.enabled,
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
            student, device_ids=[gpu_id], output_device=gpu_id
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

    triplet_loss = None
    if cfg.loss.organ_triplet.enabled:
        if cfg.loss.organ_triplet.torch_implementation:
            triplet_loss = OrganTripletMarginLoss(
                margin=cfg.loss.organ_triplet.margin,
                mining=cfg.loss.organ_triplet.mining,
            )
        else:
            triplet_loss = OrganTripletLoss(
                margin=cfg.loss.organ_triplet.margin,
                mining=cfg.loss.organ_triplet.mining,
            )
        triplet_loss = triplet_loss.to(device)
    
    adv_head, adv_loss = None, None
    if cfg.loss.domain_adv.enabled:
        num_domains = dataset.num_classes
        adv_head = DomainAdversary(
            in_dim=embed_dim,
            num_domains=num_domains,
            dropout=0.1,
        ).to(device)
        adv_loss = nn.CrossEntropyLoss()
        adv_loss = adv_loss.to(device)

    params_groups = get_params_groups(student)

    if adv_head is not None:
        params_groups.append({
            "params": adv_head.parameters(),
            "weight_decay": 0.0,   # recommended for small MLP heads
        })

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
        cfg.model.momentum_teacher, 1, cfg.training.nepochs, len(data_loader)
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
        if triplet_loss is not None:
            triplet_loss.load_state_dict(snapshot["triplet_loss"])
        if adv_head is not None:
            adv_head.load_state_dict(snapshot["adv_head"])
        if adv_loss is not None:
            adv_loss.load_state_dict(snapshot["adv_loss"])
        if fp16_scaler is not None:
            fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
        if is_main_process():
            print(f"Resuming training from snapshot at epoch {epochs_run}")

    elif cfg.resume:
        ckpt_path = Path(cfg.resume_from_checkpoint)
        epochs_run = resume_from_checkpoint(
            ckpt_path,
            verbose=(gpu_id in [-1, 0]),
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
            triplet_loss=triplet_loss,
        )
        if is_main_process():
            print(f"Resuming training from checkpoint at epoch {epochs_run}")

    early_stopping = EarlyStoppingDINO(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=snapshot_dir,
        save_every=cfg.early_stopping.save_every,
        verbose=True,
    )

    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(epochs_run, cfg.training.nepochs),
        desc=("DINO Pretraining"),
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
                triplet_loss=triplet_loss,
                triplet_loss_delay_epoch=cfg.loss.organ_triplet.delay_epoch,
                triplet_loss_weight=cfg.loss.organ_triplet.weight,
                dino_loss_weight=cfg.loss.dino.weight,
                adv_head=adv_head,
                adv_loss=adv_loss,
                adv_loss_delay_epoch=cfg.loss.domain_adv.delay_epoch,
                adv_loss_weight=cfg.loss.domain_adv.weight,
            )

            if cfg.wandb.enable and is_main_process():
                update_log_dict("train", train_stats, log_dict, step="epoch")

            if is_main_process():
                snapshot = {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                }
                if triplet_loss is not None and epoch >= cfg.loss.organ_triplet.delay_epoch:
                    snapshot["triplet_loss"] = triplet_loss.state_dict()
                if adv_head is not None:
                    snapshot["adv_head"] = adv_head.state_dict()
                if adv_loss is not None and epoch >= cfg.loss.domain_adv.delay_epoch:
                    snapshot["adv_loss"] = adv_loss.state_dict()
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()

            # only run tuning on rank 0, otherwise one has to take care of gathering knn metrics from multiple gpus
            tune_results = None
            if (
                cfg.early_stopping.enable
                and cfg.early_stopping.tune_every
                and epoch % cfg.early_stopping.tune_every == 0
                and is_main_process()
            ):
                tune_results = tune_one_epoch(
                    epoch,
                    student,
                    teacher_without_ddp,
                    downstream_query_loader,
                    downstream_test_loader,
                    features_dir,
                    cfg.model.arch,
                    cfg.model.patch_size,
                    cfg.model.drop_path_rate,
                    cfg.early_stopping.knn.k,
                    cfg.early_stopping.knn.temperature,
                    False,
                    cfg.early_stopping.knn.save_features,
                    cfg.early_stopping.knn.use_cuda,
                )

                if cfg.wandb.enable and is_main_process():
                    update_log_dict("tune", tune_results, log_dict, step="epoch")

            if is_main_process():
                early_stopping(epoch, tune_results, snapshot)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            if stop:
                tqdm.tqdm.write(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

            # log to wandb
            if is_main_process() and cfg.wandb.enable:
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

            # ensure other gpus wait until gpu_0 is finished with tuning before starting next training iteration
            if is_enabled_and_multiple_gpus():
                torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_main_process():
        print("Pretraining time {}".format(total_time_str))

    if is_enabled_and_multiple_gpus():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = get_args_parser(add_help=True).parse_args()
    main(args)
