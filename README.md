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

### Vanilla pretraining

The dataset you intend to pretrain on should be structured as follow:

```bash
patch_pretraining/
   └──imgs/
       ├── patch_1.jpg
       ├── patch_2.jpg
       └── ...
```

Where `patch_pretraining/imgs/` is the directory of patches (e.g. in `.jpg` format) extracted using [HS2P](https://github.com/clemsgrs/hs2p), used to pretrain the first Transformer block (ViT_patch).

### Hierarchical pretraining

In case you want to run hierarchical pretraining, you need to structure your data as follow:

```bash
region_pretraining/
   ├── slide_1_region_1.pt
   ├── slide_1_region_2.pt
   └── ...
```

Where `region_pretraining/` is the directory of pre-extracted region-level features for each region, generated using `python3 dino/extract_features.py`. Each `*.pt` file is a `[npatch × 384]`-sized Tensor, which contains the sequence of pre-extracted ViT_patch features for each `[patch_size × patch_size]` patch in a given region. This folder is used to pretain the intermediate Transformer block (ViT_region).


## Training

In the following python commands, make sure to replace `{gpu}` with the number of gpus available for pretraining.

### Vanilla ViT DINO pretraining :sauropod:

Update the config file `dino/config/patch.yaml` to match your local setup.<br>
Then kick off distributed pretraining of a vanilla ViT-S/16 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/patch.py
```

Alternatively, you can check `notebooks/vanilla_dino.ipynb`.

### Hierarchical pretraining :t-rex:

Update the config file `dino/config/region.yaml` to match your local setup.<br>
Then kick off distributed pretraining of a ViT-S/4096_256 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/region.py
```

Alternatively, you can check `notebooks/hierarchical_dino.ipynb`.