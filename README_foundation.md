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

## Pretraining data preparation

You need to have one `.tar` file for each cohort you intend to train on.<br>
Let's assume each is named `{cohort_name}.tar`.

Then, for each cohort:

1. Infer the auxiliary files `{cohort_name}_entries.npy` and `{cohort_name}_file_indices.npy` :

    ```shell
    python3 scripts/infer_entries.py \
        --tarball_path /path/to/{cohort_name}.tar \
        --output_root /path/to/output/folder \
        --restrict /path/to/filenames.txt \
        --name {cohort_name}
    ```

    Using the `--restrict` flag is optional.<br>
    It allows you to restrict the `npy` files to the images whose filename is in the provided `/path/to/filenames.txt`.

2. Dump `{cohort_name}.tar`, `{cohort_name}_entries.npy` and `{cohort_name}_file_indices.npy` in a common folder (e.g. `/root/data`)

Once you have completed the previous steps for each cohort :

3. Concatenate cohort entries in a single `pretrain_entries.npy` file :

    ```shell
    python3 scripts/concat_entries.py \
    --root /path/to/common/folder \
    --output_root /path/to/output/folder
    ```

4. Udpate `train.dataset_path` in `dino/config/patch.yaml` :

    ```yaml
    train:
      dataset_path: PathologyFoundation:root=/root/data
    ```

    (replace `/root/data` with the folder you chose at step 2)


## (optional) Downstream data preparation

*This section describes the steps to follow in case you want to run tuning on a downstream task dataset with patch-level labels.*

1. Create a `.csv` file containing downstream patches' filenames and labels:

    ```
    filename,label
    downstream_patch_1.jpg,3
    downstream_patch_2.jpg,1
    ...
    ```

2. Create a single tarball file that contains all downstream tuning patches and name it `downstream_dataset.tar`
3. Infer the auxiliary files `query_entries.npy` and `query_file_indices.npy` :

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/downstream_dataset.tar \
      --output_root /path/to/output/folder \
      --csv /path/to/csv/file.csv \
      --restrict /path/to/output/query.txt \
      --prefix query
    ```

    `/path/to/output/query.txt` should contain the list of filnames for the patches in the query subset of the downstream dataset.

4. Infer the auxiliary file `test_entries.npy` and `test_file_indices.npy`:

    ```shell
    python3 scripts/infer_entries.py \
      --tarball_path /path/to/downstream_dataset.tar \
      --output_root /path/to/output/folder \
      --csv /path/to/csv/file.csv \
      --restrict /path/to/output/test.txt \
      --prefix test
    ```

    `/path/to/output/test.txt` should contain the list of filnames for the patches in the test subset of the downstream dataset.

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
