import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import time
import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
import wandb
from torchvision import datasets

import dino.models.vision_transformer as vits
from dino.components import DINOLoss, EarlyStoppingDINO
from dino.eval import Tuner
from dino.data import PatchDataAugmentationDINO
from dino.distributed import (
    get_global_rank,
    get_global_size,
    is_enabled_and_multiple_gpus,
    is_main_process,
)
from dino.log import update_log_dict
from dino.models import MultiCropWrapper, MultiCropWrapperWithFeatures
from dino.models.domain_classifier import DomainClassifier
from dino.components.gradient_reversal import GradientReversalLayer
from dino.components.adversarial_loss import DomainAdversarialLoss
from dino.data.dataset import CenterAwareImageFolder
from dino.utils import (
    compute_time,
    cosine_scheduler,
    fix_random_seeds,
    get_params_groups,
    has_batchnorms,
    resume_from_checkpoint,
    setup,
    train_one_epoch,
)

logger = logging.getLogger("dino")


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

    cfg = setup(args)
    output_dir = Path(cfg.output_dir)

    fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    snapshot_dir = Path(output_dir, "snapshots")
    if not cfg.resume and is_main_process():
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(exist_ok=True, parents=True)

    # preparing data
    if is_main_process():
        logger.info("Loading data...")

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
    num_centers = None
    if cfg.adversarial.enable:
        dataset = CenterAwareImageFolder(cfg.data_dir, transform=transform)
        num_centers = dataset.num_centers
    else:
        dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    dataset_loading_end_time = time.time() - dataset_loading_start_time
    total_time_str = str(datetime.timedelta(seconds=int(dataset_loading_end_time)))
    if is_main_process():
        logger.info(f"Pretraining data loaded in {total_time_str} ({len(dataset)} patches)")
        if cfg.adversarial.enable:
            logger.info(f"Domain adversarial training enabled with {num_centers} centers: {dataset.centers}")

    if cfg.training.pct:
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)
        if is_main_process():
            logger.info(
                f"Pretraining on {cfg.training.pct*100}% of the data: {len(dataset):,d} samples"
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
        logger.info("Building student and teacher networks...")
    student = vits.__dict__[cfg.model.arch](
        img_size=cfg.model.input_size,
        patch_size=cfg.model.patch_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    teacher = vits.__dict__[cfg.model.arch](img_size=cfg.model.input_size, patch_size=cfg.model.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    # use MultiCropWrapperWithFeatures for adversarial training to extract CLS tokens
    WrapperClass = MultiCropWrapperWithFeatures if cfg.adversarial.enable else MultiCropWrapper
    student = WrapperClass(
        student,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
            norm_last_layer=cfg.model.norm_last_layer,
        ),
    )
    teacher = WrapperClass(
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
    total_iterations = cfg.training.nepochs * len(data_loader)
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        crops_number,
        cfg.model.warmup_teacher_temp,
        cfg.model.teacher_temp,
        cfg.model.warmup_teacher_temp_pct,
        total_iterations,
    )
    dino_loss = dino_loss.to(device)

    # domain adversarial training components
    domain_classifier = None
    domain_loss_fn = None
    grl = None
    if cfg.adversarial.enable:
        grl = GradientReversalLayer()

        domain_classifier = DomainClassifier(
            input_dim=embed_dim,
            num_domains=num_centers,
            hidden_dims=list(cfg.adversarial.classifier.hidden_dims),
            dropout=cfg.adversarial.classifier.dropout,
        ).to(device)

        if is_enabled_and_multiple_gpus():
            domain_classifier = nn.parallel.DistributedDataParallel(
                domain_classifier, device_ids=[gpu_id], output_device=gpu_id
            )

        domain_loss_fn = DomainAdversarialLoss(
            max_lambda=cfg.adversarial.max_lambda,
            gamma=cfg.adversarial.gamma,
            total_iterations=total_iterations,
            warmup_pct=cfg.adversarial.warmup_pct,
        )

        if is_main_process():
            logger.info(f"Domain classifier initialized: {embed_dim} -> {num_centers} centers")

    params_groups = get_params_groups(student)
    # add domain classifier parameters to optimizer if enabled
    if domain_classifier is not None:
        dc_params = domain_classifier.module.parameters() if is_enabled_and_multiple_gpus() else domain_classifier.parameters()
        params_groups.append({"params": list(dc_params), "lr": cfg.adversarial.optim.lr, "weight_decay": cfg.adversarial.optim.weight_decay})
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if cfg.speed.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    warmup_epochs = int(cfg.training.warmup_pct * cfg.training.nepochs)
    assert (
        cfg.training.nepochs >= warmup_epochs
    ), f"nepochs ({cfg.training.nepochs}) must be greater than or equal to warmup_epochs ({warmup_epochs})"
    base_lr = (
        cfg.optim.lr * (cfg.training.batch_size_per_gpu * get_global_size()) / 256.0
    )
    lr_schedule = cosine_scheduler(
        base_lr,
        cfg.optim.lr_scheduler.min_lr,
        cfg.training.nepochs,
        len(data_loader),
        warmup_epochs=warmup_epochs,
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
        logger.info("Models built, kicking off training")

    epochs_run = 0

    # leverage torch native fault tolerance
    snapshot_path = Path(snapshot_dir, "latest.pt")
    if is_enabled_and_multiple_gpus() and snapshot_path.exists():
        if is_main_process():
            logger.info("Loading snapshot")
        snapshot = torch.load(snapshot_path, map_location=device)
        epochs_run = snapshot["epoch"]
        student.load_state_dict(snapshot["student"])
        teacher.load_state_dict(snapshot["teacher"])
        optimizer.load_state_dict(snapshot["optimizer"])
        dino_loss.load_state_dict(snapshot["dino_loss"])
        if fp16_scaler is not None:
            fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
        if is_main_process():
            logger.info(f"Resuming training from snapshot at epoch {epochs_run}")

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
        )
        if is_main_process():
            logger.info(f"Resuming training from checkpoint at epoch {epochs_run}")

    early_stopping = EarlyStoppingDINO(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=snapshot_dir,
        save_every=cfg.early_stopping.save_every,
        verbose=True,
    )

    # Initialize unified tuner for downstream/robustness benchmarking
    tuner = None
    if cfg.tuning.enable:
        tuning_dir = output_dir / "tuning"
        tuner = Tuner(cfg.tuning, device, output_dir=tuning_dir)
        if is_main_process():
            logger.info("Unified tuner initialized")

    # Baseline evaluation with random weights (before any training)
    if tuner is not None and epochs_run == 0:
        baseline_results = tuner.tune(student, teacher, epoch=-1)
        if is_main_process():
            if cfg.wandb.enable:
                baseline_log = {"epoch": 0}
                update_log_dict("tune", tuner.get_log_metrics(baseline_results), baseline_log, step="epoch")
                wandb.log(baseline_log, step=0)
            else:
                logger.info("Baseline tuning results (random weights):")
                for metric_name, value in tuner.get_log_metrics(baseline_results).items():
                    logger.info(f"  {metric_name}: {value}")
        if is_enabled_and_multiple_gpus():
            torch.distributed.barrier()

    freeze_last_layer_iter = int(cfg.training.freeze_last_layer_pct * total_iterations)
    save_every_iter = int(cfg.training.save_every_pct * total_iterations)
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
                log_dict = {"epoch": epoch + 1}

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
                freeze_last_layer_iter,
                save_every_iter,
                snapshot_dir,
                gpu_id,
                domain_classifier=domain_classifier,
                domain_loss_fn=domain_loss_fn,
                grl=grl,
                dc_clip_grad=cfg.adversarial.optim.clip_grad if cfg.adversarial.enable else 0,
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
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()
                if domain_classifier is not None:
                    snapshot["domain_classifier"] = domain_classifier.state_dict()

            # Run tuning plugins (downstream + robustness) if enabled
            tune_results = None
            primary_results = None
            if tuner is not None:
                tune_results = tuner.tune(student, teacher, epoch)
                if is_main_process():
                    if cfg.wandb.enable:
                        update_log_dict("tune", tuner.get_log_metrics(tune_results), log_dict, step="epoch")
                    else:
                        logger.info(f"Tuning results at epoch {epoch + 1}:")
                        for metric_name, value in tuner.get_log_metrics(tune_results).items():
                            logger.info(f"  {metric_name}: {value}")
                    # Get primary benchmark results for early stopping
                    primary_results = tuner.get_primary_results(tune_results)

            if is_main_process():
                early_stopping(epoch, primary_results, snapshot)

            # log to wandb
            if is_main_process() and cfg.wandb.enable:
                wandb.log(log_dict, step=epoch + 1)

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
        logger.info(f"Pretraining time {total_time_str}")

    if is_enabled_and_multiple_gpus():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = get_args_parser(add_help=True).parse_args()
    main(args)
