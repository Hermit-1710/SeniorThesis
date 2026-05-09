# CTRestormer Experiment Log

## 2026-05-09

- Current evaluated checkpoint: `ctrestormer_11500iter.ckpt`
- Metrics at 11500 iter:
  - Original low-dose: PSNR `29.2489`, SSIM `0.8759`, RMSE `14.2416`
  - CTRestormer: PSNR `32.1294`, SSIM `0.8997`, RMSE `10.0689`
- Baseline targets:
  - CTformer: PSNR `32.3852`, SSIM `0.9026`, RMSE `9.8366`
  - RED-CNN: PSNR `32.6656`, SSIM `0.9067`, RMSE `9.4867`
- Loss status at 11500 iter:
  - Last 500 mean: `0.0073220`
  - Last 200 mean: `0.0070472`
  - Last 100 mean: `0.0067108`
  - Previous 100 mean: `0.0073836`
- Interpretation:
  - CTRestormer is denoising effectively but is still below CTformer and RED-CNN.
  - Loss has entered a slow plateau region, but the latest short window is still improving.

## Next Run

Resume from `11500 iter` with lower learning rate:

- `lr=1e-4`
- `batch_size=2`
- `patch_n=4`
- `patch_size=64`
- `grad_accum_steps=2`
- `use_amp=true`
- `augment_mode=advanced`
- Save every `500` iterations

Recommended next evaluation checkpoints:

- `16500 iter`
- `21500 iter`
- Stop earlier if validation metrics degrade or loss becomes unstable.

Decision rule:

- If PSNR passes `32.3852`, CTRestormer has beaten the reproduced CTformer baseline.
- If PSNR approaches or passes `32.6656`, CTRestormer is competitive with RED-CNN.
- If PSNR remains below `32.2` after 21500 iter and loss is flat, switch to a new strategy instead of simply extending training.
