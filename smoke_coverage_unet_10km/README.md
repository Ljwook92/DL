# Smoke Coverage U-Net 10 km Pilot

This bundle contains the learning-ready tensor dataset for the wildfire smoke
coverage pilot.

## Files

- `data/temporal_unet_patch_dataset.npz`
  - `x`: `(257, 80, 128, 128)` float32 input tensor
  - `y`: `(257, 1, 128, 128)` float32 next-day HMS smoke label mask
  - `valid`: `(257, 1, 128, 128)` float32 valid-cell mask
- `data/temporal_unet_patch_dataset_hgb_prior.npz`
  - `x`: `(257, 81, 128, 128)` float32 input tensor
  - the additional channel is `hgb_prior_prob_smoke`
  - use this for the hybrid HGB-prior residual U-Net
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

Recommended HPC run: use the HGB probability field as the prior and train the
higher-capacity `resattn_unetpp` CNN as a residual spatial-refinement model.

```bash
python smoke_coverage_unet_10km/scripts/train_temporal_unet_smoke_coverage.py \
  --dataset-npz smoke_coverage_unet_10km/data/temporal_unet_patch_dataset_hgb_prior.npz \
  --metadata-csv smoke_coverage_unet_10km/data/temporal_unet_patch_metadata_hgb_prior.csv \
  --channels-json smoke_coverage_unet_10km/data/temporal_unet_channels_hgb_prior.json \
  --output-dir smoke_coverage_unet_10km/hpc_runs/hgb_prior_resattn_unetpp_bc48 \
  --train-periods 202406 202407 202408 202409 202410 \
  --test-periods 202506 202507 202508 202509 202510 \
  --epochs 60 \
  --batch-size 4 \
  --model resattn_unetpp \
  --base-channels 48 \
  --dropout 0.10 \
  --learning-rate 0.00005 \
  --pos-weight-cap 5 \
  --loss bce \
  --residual-prior-channel hgb_prior_prob_smoke \
  --zero-init-output \
  --device cuda \
  --example-threshold 0.5
```

The older pure U-Net tensor can still be used by passing `--dataset-dir
smoke_coverage_unet_10km/data`, but the hybrid prior run is the preferred HPC
experiment.

The main coverage metric is fire-day patch IoU:

- `temporal_unet_test_patch_iou_by_sample.csv`
- `temporal_unet_smoke_coverage_metrics.json`

## Current Local Baseline

The local MPS pilot used 12 epochs, batch size 2, and base channels 16. The
first HPC pure U-Net runs did not beat the tabular HGB baseline by IoU. The
recommended next run is a higher-capacity hybrid residual model: HGB supplies
the cell-level probability prior and `resattn_unetpp` learns a spatial
correction to that prior.
