# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.
[[`arXiv`](https://arxiv.org/abs/2104.14294)]

<div align="left">
  <img width="70%" alt="DINO illustration" src=".github/dino.gif">
</div>

## Installation

This codebase has been developed with :
- python 3.9
- pytorch 1.12.0
- CUDA 11.3
- torchvision 0.13.0

Make sure to install the requirements: `pip3 install -r requirements.txt`

:warning: To execute the commands provided in the next sections for training and evaluation, the `dino` package should be included in the Python module search path :

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/dino"
```

## Data preparation

The dataset you intend to pretrain on should be structured as follow:

```bash
ROOT_DIR/
  └──patch_256_pretraining/
        └──imgs/
            ├── patch_1.jpg
            ├── patch_2.jpg
            └── ...
  └──region_4096_pretraining/
      ├── slide_1_region_1.pt
      ├── slide_1_region_2.pt
      └── ...
```

Where:
- `patch_256_pretraining/imgs/`: directory of patches (e.g. in `.jpg` format) extracted using [HS2P](https://github.com/clemsgrs/hs2p), used to pretrain the first Transformer block (ViT_patch).
- `region_4096_pretraining/`: directory of pre-extracted region-level features for each region, generated using `python3 dino/extract_features.py`. Each `*.pt` file is a `[npatch × 384]`-sized Tensor, which contains the sequence of pre-extracted ViT_patch features for each `[patch_size × patch_size]` patch in a given region. This folder is used to pretain the intermediate Transformer block (ViT_region).


## Training

In the following python commands, make sure to replace `{gpu}` with the number of gpus available for pretraining.

### Vanilla ViT DINO pretraining :sauropod:

Distributed pretraining of a vanilla ViT-S/16 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/patch.py
```


### Hierarchical pretraining :t-rex:

Distributed pretraining of a ViT-S/4096_256 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/region.py
```