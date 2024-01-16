import sys
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def hydra_argv_remapper(argv_map):
    """
    Call this function before main
    argv_map is a dict that remaps specific args to something else that hydra will gracefully not choke on
        ex: {'--foo':'standard.hydra.override.foo', '--bar':'example.bar'}
    workaround hydra behaviour with command line flags
    kindly given at: https://github.com/facebookresearch/hydra/issues/446#issuecomment-881031746
    """

    argv = sys.argv

    # Remap the args
    for k in argv_map.keys():
        if k in argv:
            i = argv.index(k)
            new_arg = f"{argv_map[k]}={argv[i].split('=')[-1]}"
            argv.append(new_arg)
            del argv[i]

    # Replace sys.argv with our remapped argv
    sys.argv = argv


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
    print(f"Pretrained weights found at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["teacher"]
    state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(msg)


def resume_from_checkpoint(ckpt_path, verbose: bool = True, **kwargs):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return
    if verbose:
        print(f"Found checkpoint at {ckpt_path}")

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
                    print(
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
                        print(f"=> loaded '{key}' from checkpoint: '{ckpt_path}'")
                except ValueError:
                    if verbose:
                        print(
                            f"=> failed to load '{key}' from checkpoint: '{ckpt_path}'"
                        )
        elif verbose:
            print(f"=> key '{key}' not found in checkpoint: '{ckpt_path}'")
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


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_weights(model, state_dict):
    # remove `module.` prefix induced by DDP
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    # remove `backbone.` prefix induced by multicrop wrapper
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "backbone.")
    # state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    return msg
