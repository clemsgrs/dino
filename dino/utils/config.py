import logging
import os
import torch
import datetime

from pathlib import Path
from omegaconf import OmegaConf

import dino.distributed as distributed
from dino.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from dino.configs import default_patch_config, default_region_config

logger = logging.getLogger("dino")


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_file(config_file, level: str):
    if level == "patch":
        default_config = default_patch_config
    elif level == "region":
        default_config = default_region_config
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def get_cfg_from_args(args, level: str):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        args.opts += [f"output_dir={args.output_dir}"]
    if level == "patch":
        default_config = default_patch_config
    elif level == "region":
        default_config = default_region_config
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    return cfg


def setup(args, level: str):
    """
    Basic configuration setup without any distributed or GPU-specific initialization.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    distributed.enable(overwrite=True)
    cfg = get_cfg_from_args(args, level)

    if cfg.resume:
        run_id = cfg.resume_dirname
    elif not args.skip_datetime:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    else:
        run_id = ""

    if distributed.is_main_process() and cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    if distributed.is_enabled():
        obj = [run_id]
        torch.distributed.broadcast_object_list(obj, 0, device=torch.device(f"cuda:{distributed.get_local_rank()}"))
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=cfg.resume or args.skip_datetime, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    if distributed.is_main_process():
        logger.info("git:\n  {}\n".format(get_sha()))
        cfg_path = write_config(cfg, cfg.output_dir)
        if cfg.wandb.enable:
            wandb_run.save(cfg_path)
    return cfg


def setup_distributed():
    """
    Distributed/GPU setup. This function handles:
      - Enabling distributed mode.
      - Distributed logging, seeding adjustments based on rank
    """
    distributed.enable(overwrite=True)

    torch.distributed.barrier()

    # update random seed using rank
    rank = distributed.get_global_rank()
    fix_random_seeds(rank)