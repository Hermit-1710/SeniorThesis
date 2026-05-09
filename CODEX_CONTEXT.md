# CODEX_CONTEXT

This document is the handoff context for continuing the CT denoising work in a fresh Codex project.

Workspace:

- Main repo: `E:\a_ST\CTformer\CTformer-main`
- Original DICOM training data: `E:\a_ST\AAA-ST训练数据\3mm B30`
- Prepared NPY dataset: `E:\a_ST\CTformer\CTformer-main\npy_img_3mm_B30`
- Python environment: `E:\Anaconda\envs\CTformer`
- GPU target: NVIDIA GeForce RTX 3060 Laptop GPU, 6GB VRAM

## Current Goal

The thesis goal is to implement and evaluate a stronger Transformer-based CT denoising model. CTformer and RED-CNN are baselines. CTNAFNet is a modern CNN restoration baseline. CTRestormer is the new Transformer main model.

The user wants the project files organized by model and wants to continue training/evaluating CTRestormer, while watching whether the loss has genuinely plateaued.

## Important Current State

The project has been reorganized into:

- `E:\a_ST\CTformer\CTformer-main\model_projects\CTformer`
- `E:\a_ST\CTformer\CTformer-main\model_projects\CTNAFNet`
- `E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer`

Each model project contains:

- `code`: model-specific source code and runnable entry files
- `docs`: project summaries and notes
- `runs`: checkpoints, logs, loss curves, and test outputs
- `test_assets`: small sample test slices

The original root project was intentionally kept temporarily during reorganization to avoid breaking the known-working training/test workflow. The user later said it can be deleted after the split is complete. Before deleting, verify the model-specific folders are truly self-contained and that CTRestormer can train/test from its own `code` directory.

## Existing Reproduction And Baselines

### CTformer

Main original files:

- `CTformer.py`
- `T2T_transformer_block.py`
- `token_performer.py`
- `token_transformer.py`
- `main.py`
- `solver.py`

Main run directory:

- `runs\train_100ep_bs4`

Local test result from the reproduced CTformer run:

- PSNR: `32.3852`
- SSIM: `0.9026`
- RMSE: `9.8366`

The CTformer visualization output was previously exported under:

- `E:\a_ST\CTformer\CTformer-main\runs\train_100ep_bs4\fig`

There was a concern that exported test image order looked different from RED-CNN. The loader/test export was adjusted to preserve and annotate original slice IDs, so filenames like `L506_88.png` should be preferred over plain numeric indexes.

### RED-CNN Baseline

RED-CNN lives separately under:

- `E:\a_ST\RED-CNN`

Local RED-CNN test result:

- PSNR: `32.6656`
- SSIM: `0.9067`
- RMSE: `9.4867`

This currently beats the reproduced CTformer result on the same local split.

### CTNAFNet

Folder:

- `E:\a_ST\CTformer\CTformer-main\advanced_ct_denoiser`

Reorganized copy:

- `E:\a_ST\CTformer\CTformer-main\model_projects\CTNAFNet`

Nature:

- NAFNet-style residual U-Net
- CNN/image-restoration baseline, not Transformer
- Multi-scale encoder-decoder, skip connections, residual output
- Uses NAF blocks

Training choices:

- Optimizer: AdamW
- Scheduler: cosine
- Loss: HybridLoss = `0.8 * Charbonnier + 0.2 * SSIM`
- Designed to be stronger than old RED-CNN as a modern CNN baseline

Pilot/checkpoint directory:

- `runs\ctnafnet_v1`

Known checkpoints:

- `ctnafnet_50iter.ckpt`
- `ctnafnet_100iter.ckpt`

## CTRestormer

This is the new Transformer main model.

Original folder:

- `E:\a_ST\CTformer\CTformer-main\ctrestormer`

Reorganized copy:

- `E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer`

Key files:

- `ctrestormer\model.py`
- `ctrestormer\README.md`
- `ctrestormer\EXPERIMENT_PLAN.md`
- `ctrestormer\train_ctrestormer_v1.bat`
- `main.py`
- `solver.py`

Architecture:

- Restormer-style multi-scale image restoration Transformer
- Overlap patch embedding
- Encoder-decoder hierarchy
- Downsample/upsample stages using PixelUnshuffle/PixelShuffle
- Transformer blocks with:
  - 2D LayerNorm
  - MDTA-style attention
  - Gated depthwise feed-forward network
  - Residual connections
- Final residual output: model predicts denoised image with input residual connection

3060 Laptop GPU adapted config:

- `restormer_dim=24`
- `restormer_num_blocks=1,2,2,3`
- `restormer_num_refinement_blocks=1`
- `restormer_heads=1,2,4,4`
- `batch_size=2`
- `patch_n=4`
- `patch_size=64`
- `grad_accum_steps=2`
- `use_amp=true`
- `augment_mode=advanced`
- `loss_name=hybrid`
- `optimizer=adamw`
- `scheduler=cosine`
- `lr=2e-4`
- `weight_decay=1e-4`
- `num_workers=0`

Training run:

- `runs\ctrestormer_v1`

Known checkpoints saved:

- `ctrestormer_50iter.ckpt`
- `ctrestormer_100iter.ckpt`
- `ctrestormer_500iter.ckpt`
- ...
- `ctrestormer_11500iter.ckpt`

The long training process was stopped to run evaluation. Last observed training state before stopping:

- Around `STEP [11840]`
- Epoch around `11/201`
- Overall progress around `5.46%`
- Recent printed losses mostly around `0.004` to `0.012`

Important: the latest complete checkpoint evaluated was `11500 iter`, not `11840`.

### CTRestormer Test Result At 11500 Iter

Test command used the 3060-compatible config and `test_iters=11500`.

Original low-dose input:

- PSNR: `29.2489`
- SSIM: `0.8759`
- RMSE: `14.2416`

CTRestormer prediction:

- PSNR: `32.1294`
- SSIM: `0.8997`
- RMSE: `10.0689`

Current comparison:

- RED-CNN: `PSNR 32.6656`, `SSIM 0.9067`, `RMSE 9.4867`
- CTformer: `PSNR 32.3852`, `SSIM 0.9026`, `RMSE 9.8366`
- CTRestormer 11500 iter: `PSNR 32.1294`, `SSIM 0.8997`, `RMSE 10.0689`

Conclusion so far:

- CTRestormer is denoising effectively but has not yet surpassed CTformer or RED-CNN at only 11500 iterations.
- It was only about 5.4% through the planned training schedule, so it may still improve.
- However, the loss appears to be in a plateau-like region.

### Loss Plateau Check

The latest inspected loss file:

- `runs\ctrestormer_v1\loss_11500_iter.npy`

Quantitative loss summary:

- Number of stored losses: `11400`
- First 500 mean: `0.0311548`
- Last 2000 mean: `0.0073072`
- Last 1000 mean: `0.0073500`
- Last 500 mean: `0.0073220`
- Last 200 mean: `0.0070472`
- Previous 100 mean: `0.0073836`
- Last 100 mean: `0.0067108`
- Last 100 delta vs previous 100: `-0.0006728`
- Last 500 delta vs previous 500: about `-0.000056`
- Last 1000 delta vs previous 1000: about `+0.0000856`

Interpretation:

- The loss is not completely frozen.
- It is in a clear plateau region.
- Continuing training may still help, but pure continuation at the same settings is likely to improve slowly.

Recommended next training strategy:

- Do not only train blindly for many more hours at the same settings.
- Continue from `11500 iter`, but consider a fine-tuning phase:
  - lower learning rate to `5e-5` or `1e-4`
  - keep AMP and gradient accumulation
  - evaluate every fixed interval, such as every `5000` or `10000` iterations
  - optionally export visual results periodically
- If the loss does not improve and test PSNR/SSIM does not improve after a few checkpoints, switch strategy:
  - stronger data augmentation
  - mixed 3mm + 1mm pretraining, then 3mm fine-tune
  - architectural adjustment for local-window or channel attention balance

## Important Code Changes Already Made

### Data Preparation

`prep.py` was adapted to support nested DICOM folder layouts such as:

- `3mm B30\full_3mm\L067\full_3mm\...`
- `3mm B30\quarter_3mm\L067\quarter_3mm\...`

### Loader

`loader.py`:

- returns `slice_id` in test mode
- supports `augment_mode`
- supports advanced augmentation

### Augmentation

`data_aug.py`:

- includes `advanced_data_augmentation`

### Main/Solver

`main.py`:

- fixed boolean parsing
- added model/loss/optimizer/scheduler args
- added CTNAFNet args
- added CTRestormer args
- added AMP and gradient accumulation args
- no hard-coded CUDA device requirement

`solver.py`:

- supports `model_name`:
  - `ctformer`
  - `ctnafnet`
  - `ctrestormer`
- supports `loss_name`:
  - `mse`
  - `charbonnier`
  - `hybrid`
- supports optimizer:
  - `adam`
  - `adamw`
- supports cosine scheduler
- supports AMP and gradient accumulation
- logs training progress to `train.log`
- saves loss files and `loss.png`
- test output supports original slice IDs in filenames
- CTRestormer test required tiled inference to avoid OOM on 512x512 full-image attention

Important OOM fix:

- Direct full-image CTRestormer inference caused CUDA OOM, trying to allocate roughly `256 GiB`.
- A tiled inference helper was added in root `solver.py` for CTRestormer test:
  - split image into overlapping 64x64 patches
  - run model on patch batches
  - aggregate center crops back to 512x512

If continuing from the model-specific `CTRestormer\code` folder, ensure this tiled inference patch exists there too.

## Current Reorganization Work In Progress

The following model-project folders already exist:

- `model_projects\CTformer`
- `model_projects\CTNAFNet`
- `model_projects\CTRestormer`

Initial docs added:

- `model_projects\README.md`
- `model_projects\CTformer\docs\PROJECT_SUMMARY.md`
- `model_projects\CTNAFNet\docs\PROJECT_SUMMARY.md`
- `model_projects\CTRestormer\docs\PROJECT_SUMMARY.md`

To make the folders self-contained, `losses.py` was added to each `code` folder:

- `model_projects\CTformer\code\losses.py`
- `model_projects\CTNAFNet\code\losses.py`
- `model_projects\CTRestormer\code\losses.py`

The model-specific `solver.py` files were partially changed so they do not rely on other model folders:

- CTformer solver should only import/build CTformer.
- CTNAFNet solver should only import/build CTNAFNet.
- CTRestormer solver should only import/build CTRestormer.

Before deleting root files, verify:

- `model_projects\CTRestormer\code\solver.py` contains the tiled inference patch.
- `model_projects\CTRestormer\code\ctrestormer\train_ctrestormer_v1.bat` points to the new code directory and the correct data/run paths.
- A smoke train/test command runs successfully from `model_projects\CTRestormer\code`.

## Recommended CTRestormer Resume Command

Use the new project code directory once self-contained:

```powershell
cd /d E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\code
& 'E:\Anaconda\envs\CTformer\python.exe' .\main.py `
  --mode train `
  --model_name ctrestormer `
  --loss_name hybrid `
  --optimizer adamw `
  --scheduler cosine `
  --saved_path 'E:\a_ST\CTformer\CTformer-main\npy_img_3mm_B30' `
  --save_path 'E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\runs\ctrestormer_v1' `
  --device cuda `
  --batch_size 2 `
  --patch_n 4 `
  --patch_size 64 `
  --num_epochs 201 `
  --print_iters 20 `
  --save_iters 500 `
  --decay_iters 0 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --num_workers 0 `
  --result_fig false `
  --augment_mode advanced `
  --restormer_dim 24 `
  --restormer_num_blocks 1,2,2,3 `
  --restormer_num_refinement_blocks 1 `
  --restormer_heads 1,2,4,4 `
  --grad_accum_steps 2 `
  --use_amp true `
  --resume_iters 11500
```

Reason for `lr=1e-4`:

- The current `2e-4` run has entered a plateau-like region.
- Lower LR fine-tuning may improve stability and test metrics.

## Recommended CTRestormer Test Command

```powershell
cd /d E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\code
& 'E:\Anaconda\envs\CTformer\python.exe' .\main.py `
  --mode test `
  --model_name ctrestormer `
  --loss_name hybrid `
  --optimizer adamw `
  --scheduler cosine `
  --saved_path 'E:\a_ST\CTformer\CTformer-main\npy_img_3mm_B30' `
  --save_path 'E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\runs\ctrestormer_v1' `
  --device cuda `
  --batch_size 2 `
  --patch_n 4 `
  --patch_size 64 `
  --num_workers 0 `
  --result_fig true `
  --test_iters 11500 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --augment_mode advanced `
  --restormer_dim 24 `
  --restormer_num_blocks 1,2,2,3 `
  --restormer_num_refinement_blocks 1 `
  --restormer_heads 1,2,4,4 `
  --grad_accum_steps 2 `
  --use_amp true
```

## Data Expansion Advice

The user considered adding paired `1mm B30` CT slices as extra training data.

Recommendation:

- First preserve a clean 3mm-only baseline.
- Do not immediately mix 1mm and 3mm without tracking domain effects.
- Good plan:
  - pretrain on 3mm + 1mm paired data
  - fine-tune on 3mm only
  - evaluate on the same 3mm test patient/split

Reason:

- 1mm and 3mm CT images have different slice thickness, noise texture, and anatomical continuity.
- Naive mixing may help generalization, but it can also slightly hurt the exact 3mm target benchmark.

## Next Concrete Tasks

1. Finish making `model_projects\CTRestormer` self-contained.
2. Verify CTRestormer can train/test from `model_projects\CTRestormer\code`.
3. After verification, remove or archive redundant root-level project files if the user still wants cleanup.
4. Resume CTRestormer from `11500 iter` using lower LR fine-tuning.
5. Run periodic evaluation and compare against:
   - RED-CNN: `32.6656 / 0.9067 / 9.4867`
   - CTformer: `32.3852 / 0.9026 / 9.8366`
   - CTRestormer current: `32.1294 / 0.8997 / 10.0689`

