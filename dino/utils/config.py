import os

from omegaconf import OmegaConf

from dino.config import patch_default_config, region_default_config


def write_config(cfg, output_dir, name="config.yaml"):
    print(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    if args.level == "patch":
        default_cfg = OmegaConf.create(patch_default_config)
    elif args.level == "region":
        default_cfg = OmegaConf.create(region_default_config)
    else:
        raise KeyError(
            f"Level should be in ['patch', 'region'] (provided level: {args.level})"
        )
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg
