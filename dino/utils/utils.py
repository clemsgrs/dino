import logging
import os
import tqdm
import wandb
import torch
import random
import subprocess
import numpy as np
import torch.nn as nn

from typing import Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("dino")


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def compute_gradient_stats(model, clip=None, model_name="model"):
    """Compute gradient statistics and optionally clip gradients.

    Args:
        model: The model to compute gradient stats for.
        clip: If provided, clip gradients to this max norm (per-parameter).
        model_name: Name for logging purposes.

    Returns:
        dict with keys:
            - {model_name}_grad_norm: Total L2 norm across all parameters
            - {model_name}_grad_max: Maximum per-parameter gradient norm
            - {model_name}_clip_fraction: Fraction of parameters that were clipped (0 if clip=None)
    """
    total_norm_sq = 0.0
    max_norm = 0.0
    num_params = 0
    num_clipped = 0

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            num_params += 1

            if clip is not None:
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
                    num_clipped += 1

    total_norm = total_norm_sq ** 0.5
    clip_fraction = num_clipped / num_params if num_params > 0 else 0.0

    return {
        f"{model_name}_grad_norm": total_norm,
        f"{model_name}_grad_max": max_norm,
        f"{model_name}_clip_fraction": clip_fraction,
    }


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k, v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f"{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes"
    return updated_state_dict, msg


def start_from_checkpoint(ckpt_path, model):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return
    logger.info(f"Pretrained weights found at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["teacher"]
    state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)
    logger.info(msg)


def resume_from_checkpoint(ckpt_path, verbose: bool = True, **kwargs):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return 0
    if verbose:
        logger.info(f"Found checkpoint at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                sd = checkpoint[key]
                nn.modules.utils.consume_prefix_in_state_dict_if_present(sd, "module.")
                msg = value.load_state_dict(sd, strict=False)
                if verbose:
                    logger.info(
                        f"=> loaded '{key}' from checkpoint: '{ckpt_path}' with msg {msg}"
                    )
            except TypeError:
                try:
                    sd = checkpoint[key]
                    nn.modules.utils.consume_prefix_in_state_dict_if_present(
                        sd, "module."
                    )
                    msg = value.load_state_dict(sd)
                    if verbose:
                        logger.info(f"=> loaded '{key}' from checkpoint: '{ckpt_path}'")
                except ValueError:
                    if verbose:
                        logger.warning(
                            f"=> failed to load '{key}' from checkpoint: '{ckpt_path}'"
                        )
        elif verbose:
            logger.warning(f"=> key '{key}' not found in checkpoint: '{ckpt_path}'")
    return epoch


def cosine_scheduler(
    base_value,
    final_value,
    nepochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(nepochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == nepochs * niter_per_ep
    return schedule


def load_weights(model, state_dict):
    # remove `module.` prefix induced by DDP
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    # remove `backbone.` prefix induced by multicrop wrapper
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "backbone.")
    # state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        tqdm.tqdm.write(str(msg))
    else:
        tqdm.tqdm.write("All keys matched successfully")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags is None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.wandb.resume_id:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
            id=cfg.wandb.resume_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
        )
    run.define_metric("epoch", summary="max")
    return run