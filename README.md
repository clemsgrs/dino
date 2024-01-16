# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO.<br>
For details, see **Emerging Properties in Self-Supervised Vision Transformers**.
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

1. Ensure images are all in one directory
2. Create a single large tarball file that contains all images and name it `pretrain_dataset.tar` :

    ```shell
    tar -chf pretrain_dataset.tar /path/to/image/folder
    ```

3. Infer the auxiliary files `pretrain_entries.npy` and `pretrain_file_indices.npy` :

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/pretrain_dataset.tar \
      --output_root /path/to/output/folder \
      --name pretrain
    ```

    The `pretrain_entries.npy` file will record:
    - a dummy class index (we set it to 0 for all images since we’re not using classes)
    - a unique filename index for each image
    - the start and end offsets of each image within the tarball file

    The `pretrain_file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename.

4. Dump `pretrain_dataset.tar`, `pretrain_entries.npy` and `pretrain_file_indices.npy` in a common folder (e.g. `/root/data`)

5. Udpate `train.dataset_path` in `dino/config/patch.yaml` :

    ```yaml
    train:
      dataset_path: Pathology:root=/root/data
    ```

    (replace `/root/data` with the folder you chose at step 4)

## Training

In the following python commands, make sure to replace `{gpu}` with the number of gpus available for pretraining.

### Vanilla ViT DINO pretraining :sauropod:

Update the config `dino/config/patch.yaml` if you want to change some parameters.<br>
Then kick off distributed pretraining of a vanilla ViT-S/16 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/patch.py --config-file dino/config/patch.yaml
```

### Hierarchical pretraining :t-rex:

Update the config `dino/config/region.yaml` if you want to change some parameters.<br>
Then kick off distributed pretraining of a ViT-S/4096_256 :

```bash
python3 -m torch.distributed.run --nproc_per_node={gpu} dino/region.py --config-file dino/config/region.yaml
```
