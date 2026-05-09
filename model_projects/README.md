# Model Projects

This folder reorganizes the work into three independent model-focused project folders:

- `CTformer`
- `CTNAFNet`
- `CTRestormer`

Each project folder uses the same layout:

- `code`: source code and runnable entry files used for that model workflow
- `docs`: model-specific notes and experiment summary
- `runs`: training, testing, checkpoints, logs, and exported figures/results
- `test_assets`: small sample test assets copied for quick inspection

Notes:

- The original root project remains unchanged so the currently validated training and testing workflow is preserved.
- The training dataset `npy_img_3mm_B30` is not duplicated here to avoid unnecessary disk usage.
- For formal experiment continuation, keep using the root environment and dataset path, and use these folders as model-specific code/result packages.
