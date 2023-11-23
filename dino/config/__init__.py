from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(Path(__file__).parent.resolve() / config_filename)


patch_default_config = load_config("patch")
region_default_config = load_config("region")


def load_and_merge_config(config_name: str, level: str):
    if level == "patch":
        default_config = OmegaConf.create(patch_default_config)
    elif level == "region":
        default_config = OmegaConf.create(region_default_config)
    else:
        raise KeyError(f"Level should be in ['patch', 'region'] (provided level: {level})")
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)