# CTRestormer Project Summary

Primary code:

- `code\ctrestormer\model.py`
- `code\ctrestormer\README.md`
- `code\ctrestormer\EXPERIMENT_PLAN.md`
- `code\ctrestormer\train_ctrestormer_v1.bat`
- `code\main.py`
- `code\solver.py`

Main results:

- `runs\ctrestormer_smoke`
- `runs\ctrestormer_smoke2`
- `runs\ctrestormer_smoke3`
- `runs\ctrestormer_v1`

Current evaluation snapshot:

- Latest tested checkpoint in this round: `ctrestormer_11500iter.ckpt`
- Test metrics from the latest evaluation:
  - `PSNR 32.1294`
  - `SSIM 0.8997`
  - `RMSE 10.0689`

Included extras:

- `test_assets`: quick local sample slices for checking output behavior

Recommended use:

- Use this folder when you want the new Transformer main model code, experiment plan, pilot training outputs, and current test results grouped together.
