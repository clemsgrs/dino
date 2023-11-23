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