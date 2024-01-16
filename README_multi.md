# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO.<br>
For details, see **Emerging Properties in Self-Supervised Vision Transformers**.
[[`arXiv`](https://arxiv.org/abs/2104.14294)]

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

1. ensure pretraining images are all in one directory
2. create a single large tarball file that contains all pretraining images

    ```shell
    tar -chf pretrain_dataset.tar /path/to/image/folder
    ```

3. infer the **fold-specific** auxiliary file `pretrain_entries_${fold}.npy` and `pretrain_file_indices.npy`:

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/pretrain_dataset.tar \
      --output_root /path/to/output/folder \
      --prefix pretrain \
      --restrict /path/to/filenames_${fold}.txt \
      --suffix ${fold}
    ```

    The `pretrain_entries_${fold}.npy` file will record:
    - a dummy class index (we set it to 0 for all images since we’re not using classes)
    - a unique filename index for each image
    - the start and end offsets of each image within the tarball file

    Using the `--restrict` flag in the previous command ensure we only save this information for the images whose filename is in `/path/to/filenames_${fold}.txt`.

    The `pretrain_file_indices.npy` file consists in a dictionnary mapping filename index to corresponding filename (for all images).

4. dump `pretrain_dataset.tar`, `pretrain_entries_${fold}.npy` and `pretrain_file_indices.npy` in a common folder (e.g. `/root/data`)
5. udpate `train.dataset_path` in `dino/config/patch.yaml` (replace `/root/data` with the folder you chose at step 4)

6. (optional) ensure downstream tuning images are all in one directory
7. (optional) create a single large tarball file that contains all downstream tuning images
8. (optional) infer the **fold-specific** auxiliary file `query_entries_${fold}.npy` and `query_file_indices.npy`:

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/downstream_dataset.tar \
      --output_root /path/to/output/folder \
      --csv /path/to/output/query_${fold}.csv \
      --prefix query \
      --restrict /path/to/output/query_${fold}.txt \
      --suffix ${fold}
    ```

9. (optional) infer the **fold-specific** auxiliary file `test_entries_${fold}.npy` and `test_file_indices.npy`:

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/downstream_dataset.tar \
      --output_root /path/to/output/folder \
      --csv /path/to/output/test_${fold}.csv \
      --prefix test \
      --restrict /path/to/output/test_${fold}.txt \
      --suffix ${fold}
    ```

10. (optional) doublecheck that the various `.npy` files contain the expected information

    ```shell
    python3 scripts/test_entries.py \
      --image_root /path/to/original/image/folder \
      --tarball_path /path/to/pretrain_dataset.tar \
      --entries_path /path/to/output/folder/pretrain_entries_${fold}.npy \
      --file_indices_path /path/to/output/folder/pretrain_file_indices.npy \
      --sample_size 1000
    ```


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