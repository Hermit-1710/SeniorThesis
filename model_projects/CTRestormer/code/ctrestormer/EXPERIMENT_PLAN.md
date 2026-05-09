# CTRestormer Experiment Plan

## Goal

Build a stronger transformer-based CT denoising model than the current local CTformer, while keeping RED-CNN and CTNAFNet as baselines.

## Baselines

1. RED-CNN
2. CTformer
3. CTNAFNet
4. CTRestormer

## Data split

- training patients: all except `L506`
- test patient: `L506`
- current dataset: `3mm B30`

## Training stages

### Stage 1

Smoke test:

- `100` to `300` iterations
- verify loss decreases
- verify GPU memory is safe

### Stage 2

Pilot training:

- `30` to `50` epochs
- compare metrics with CTformer and RED-CNN

### Stage 3

Main training:

- `150` to `200` epochs
- export test metrics and result figures

## Metrics

- PSNR
- SSIM
- RMSE
- visual sharpness around edges
- over-smoothing risk

## Ablations worth doing

1. `MSE` vs `HybridLoss`
2. `64x64` vs larger patch if memory allows
3. with and without 1mm data
4. transformer-only vs transformer + refinement depth changes

## 1mm B30 advice

Recommended strategy:

- do not mix 1mm and 3mm slices blindly at the start
- first establish a clean 3mm-only baseline
- then try one of these:
  - joint training with a `slice_thickness` flag as metadata
  - fine-tune on 3mm after mixed pretraining
  - separate model for 1mm if visual texture is clearly different

Why:

- 1mm slices have different noise texture and anatomical sharpness
- naive mixing can hurt 3mm evaluation
- but mixed pretraining may improve robustness if handled carefully
