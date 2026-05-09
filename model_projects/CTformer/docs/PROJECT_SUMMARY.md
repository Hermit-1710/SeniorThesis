# CTformer Project Summary

Primary code:

- `code\CTformer.py`
- `code\T2T_transformer_block.py`
- `code\token_performer.py`
- `code\token_transformer.py`
- `code\main.py`
- `code\solver.py`

Main results:

- `runs\train_100ep_bs4`
- `runs\bench_10iter`
- `runs\bench_bs4_20iter`
- `runs\bench_bs8_20iter`
- `runs\bench_bs4_100iter`

Known local baseline:

- Checkpoint: `T2T_vit_54718iter.ckpt`
- PSNR: `32.3852`
- SSIM: `0.9026`
- RMSE: `9.8366`

Supplemental long-training check:

- Purpose: verify whether the reproduced CTformer baseline was under-trained.
- Resume checkpoint: `T2T_vit_54718iter.ckpt`
- Final checkpoint: `T2T_vit_100000iter.ckpt`
- AMP was disabled for this continuation because CTformer + MSE loss produced `nan` values with AMP.
- Loss at `100000 iter`:
  - Last 1000 mean: `0.0018202`
  - Last 500 mean: `0.0018350`
  - Last 100 mean: `0.0017671`
- Test result at `100000 iter`:
  - PSNR: `32.6688`
  - SSIM: `0.9067`
  - RMSE: `9.5197`

Interpretation:

- Longer CTformer training substantially improves the reproduced CTformer baseline.
- CTformer `100000 iter` slightly exceeds the previous RED-CNN PSNR/SSIM, but its RMSE remains slightly worse than RED-CNN.
- CTRestormer `21500 iter` remains the best local result: `32.6738 / 0.9096 / 9.4842`.

Included extras:

- `model_pretrained`: original pretrained CTformer weights
- `test_assets`: quick local sample slices for checking output behavior

Recommended use:

- Use this folder when you want the reproduced CTformer baseline code and its historical training/test outputs grouped together.
