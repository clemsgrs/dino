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

You need to wrap up your data in a tarball file:

1. ensure images are all in one directory
2. create a single large tarball file that contains all images

    ```shell
    tar -cvf pretrain_dataset.tar /path/to/image/folder
    ```

3. infer the auxiliary files `pretrain_entries.npy` and `file_indices.npy`

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/pretrain_dataset.tar \
      --output_root /path/to/output/folder \
      --prefix pretrain
    ```

    The `entries.npy` file will record:
    - a dummy class index (we set it to 0 for all images since we’re not using classes)
    - a unique filename index for each image
    - the start and end offsets of each image within the tarball file

    The `file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename.

4. dump `pretrain_dataset.tar`, `entries.npy` and `file_indices.npy` in a common folder (e.g. `/root/data`)
5. udpate `train.dataset_path` in `dino/config/patch.yaml` (replace `/root/data` with the folder you chose at step 4)

## Training

In the following python commands, make sure to replace `{gpu}` with the number of gpus available for pretraining.

### Vanilla ViT DINO pretraining :sauropod:

Update the config file `dino/config/patch.yaml` to match your local setup.<br>
Then kick off distributed pretraining of a vanilla ViT-S/16 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/patch.py --config-file dino/config/patch.yaml
```

### Hierarchical pretraining :t-rex:

Update the config file `dino/config/region.yaml` to match your local setup.<br>
Then kick off distributed pretraining of a ViT-S/4096_256 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/region.py --config-file dino/config/region.yaml
```