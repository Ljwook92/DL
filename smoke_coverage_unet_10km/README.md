# Smoke Coverage U-Net 10 km Pilot

This bundle contains the learning-ready tensor dataset for the wildfire smoke
coverage pilot.

## Files

- `data/temporal_unet_patch_dataset.npz`
  - `x`: `(257, 80, 128, 128)` float32 input tensor
  - `y`: `(257, 1, 128, 128)` float32 next-day HMS smoke label mask
  - `valid`: `(257, 1, 128, 128)` float32 valid-cell mask
- `data/temporal_unet_patch_metadata.csv`
  - one row per fire-day patch
  - train periods: `202406` to `202410`
  - test periods: `202506` to `202510`
- `data/temporal_unet_channels.json`
  - ordered input channel names and imputation medians
- `scripts/train_temporal_unet_smoke_coverage.py`
  - compact PyTorch U-Net training/evaluation script

## Setup On HPC

Clone and pull the Git LFS data:

```bash
git clone https://github.com/Ljwook92/DL.git
cd DL
git lfs pull
```

Create an environment. Install the CUDA-compatible PyTorch build recommended by
the HPC cluster, then install the remaining packages:

```bash
pip install -r smoke_coverage_unet_10km/requirements.txt
```

## Train

```bash
python smoke_coverage_unet_10km/scripts/train_temporal_unet_smoke_coverage.py \
  --dataset-dir smoke_coverage_unet_10km/data \
  --output-dir smoke_coverage_unet_10km/hpc_runs/unet_cuda_tversky_fp70 \
  --train-periods 202406 202407 202408 202409 202410 \
  --test-periods 202506 202507 202508 202509 202510 \
  --epochs 80 \
  --batch-size 8 \
  --base-channels 32 \
  --pos-weight-cap 2 \
  --loss bce_tversky \
  --bce-weight 0.3 \
  --dice-weight 0.7 \
  --tversky-alpha 0.7 \
  --tversky-beta 0.3 \
  --device cuda \
  --example-threshold 0.5
```

The main coverage metric is fire-day patch IoU:

- `temporal_unet_test_patch_iou_by_sample.csv`
- `temporal_unet_smoke_coverage_metrics.json`

## Current Local Baseline

The local MPS pilot used 12 epochs, batch size 2, and base channels 16. The
first HPC run used 80 epochs, batch size 8, and base channels 32 with BCE loss.
It ran successfully but still did not beat the tabular HGB baseline by IoU. The
recommended next run is `bce_tversky` with `tversky-alpha=0.7` and
`tversky-beta=0.3`, which penalizes false-positive plume area more strongly.
