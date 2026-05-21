# Thesis Experiment Expansion Plan

This file lists the most useful additional experiments for the ST4001 thesis. The priority is to add evidence that the models remove noise rather than anatomical structure.

## Priority 1: Residual Noise Map Analysis

For each test slice and model, compute:

- Reference low-dose residual: `input - target`
- Model removed component: `input - prediction`
- Remaining prediction error: `prediction - target`

Recommended figures:

1. One visual panel for a representative slice:
   - low-dose input
   - full-dose target
   - RED-CNN output
   - CTformer output
   - CTRestormer output
   - `input - target`
   - `input - RED-CNN`
   - `input - CTformer`
   - `input - CTRestormer`
2. Histogram panel of residual values for the same slice.
3. Average radial power spectrum of residual maps over all L506 test slices.

Recommended quantitative metrics:

- Residual mean and standard deviation.
- Correlation between `input - prediction` and `input - target`.
- Correlation between `abs(gradient(target))` and `abs(gradient(input - prediction))` to estimate edge leakage.
- ROI residual statistics for air, soft tissue, and bone regions.

Why it helps the thesis:

- It answers whether each model removes similar noise.
- It reveals whether the best PSNR model may be removing anatomical edges.
- It creates a stronger discussion than PSNR/SSIM alone.

## Priority 2: CTRestormer Checkpoint Study

Current checkpoints with known results:

| Checkpoint | PSNR | SSIM | RMSE |
| --- | ---: | ---: | ---: |
| 11500 | 32.1294 | 0.8997 | 10.0689 |
| 16500 | 32.3584 | 0.9049 | 9.8145 |
| 18000 | 32.3534 | 0.9038 | 9.8134 |
| 21500 | 32.6738 | 0.9096 | 9.4842 |

Next action:

- Test `22000` if time permits.
- Plot PSNR/SSIM/RMSE versus checkpoint.
- Keep `21500` as current best unless `22000` improves clearly.

Why it helps:

- Shows that fine-tuning mattered.
- Supports the claim that CTRestormer needed more than the early checkpoint.

## Priority 3: Noise Texture / Frequency-Domain Analysis

Compute 2D FFT power spectra for:

- `input - target`
- `input - RED-CNN`
- `input - CTformer`
- `input - CTRestormer`

Then radially average into 1D curves.

Why it helps:

- CT denoising should preserve realistic noise texture.
- It can show whether models remove high-frequency noise only or also suppress lower-frequency anatomical texture.

## Priority 4: ROI-Level Analysis

Use simple threshold masks from the full-dose target:

- Air/background
- Soft tissue
- Bone/high-intensity region

Compute PSNR/RMSE or residual standard deviation per ROI.

Why it helps:

- CT denoising behavior may differ between soft tissue and bone.
- Soft tissue analysis is important because low-contrast details are clinically sensitive.

## Priority 5: Noise2Noise-Style Experiment

Direct Noise2Noise requires two independent noisy observations of the same anatomy. The current prepared AAPM data likely does not provide this directly, so do not claim true clinical Noise2Noise unless such data is available.

Feasible version:

1. Start from full-dose target `y`.
2. Create two synthetic noisy versions:
   - `x1 = y + noise1`
   - `x2 = y + noise2`
3. Train a model from `x1` to `x2`.
4. Evaluate output against `y`.

How to write it honestly:

- Call it `pseudo Noise2Noise`.
- State that synthetic noise may not match true CT acquisition noise.
- Use it as an auxiliary experiment, not the main result.

## Other Good Extensions

- Loss ablation: MSE vs Charbonnier vs Hybrid.
- Tiling ablation: compare different tile overlaps during CTRestormer inference.
- Data expansion: 1mm+3mm pretraining followed by 3mm fine-tuning.
- Efficiency comparison: parameters, inference time, GPU memory.
- Failure-case analysis: show slices where CTRestormer is worse than CTformer or RED-CNN.
