# Advanced CT Denoiser

This folder contains a stronger denoising baseline for the same Mayo-style low-dose CT dataset used by the current CTformer and RED-CNN experiments.

## Why this model

The current local results show:

- `REDCNN`: `PSNR 32.6656`, `SSIM 0.9067`, `RMSE 9.4867`
- `CTformer`: `PSNR 32.3852`, `SSIM 0.9026`, `RMSE 9.8366`

On this dataset and training setup, the shallow CTformer implementation is currently weaker than RED-CNN. The main reasons are:

- very shallow transformer trunk (`depth=1`)
- pure MSE training
- patch-wise training with limited context
- weak local inductive bias compared with convolutional restoration networks

This new model uses a NAFNet-style residual U-Net because it is a strong fit for image restoration under limited data and moderate GPU memory.

## Model summary

Model name: `CTNAFNet`

Core design:

- residual learning: output is `input + residual`
- encoder-decoder U-Net structure
- NAF blocks at every scale
- lightweight channel attention
- multi-scale skip connections
- all-PyTorch implementation, no extra packages

Recommended default configuration in this repo:

- base width: `32`
- encoder blocks: `1,1,2`
- middle blocks: `4`
- decoder blocks: `1,1,1`
- patch size: `64`
- batch size: `4`

For a `64x64` training patch, the feature flow is:

1. input: `1 x 64 x 64`
2. stem conv -> `32 x 64 x 64`
3. encoder stage 1 -> `32 x 64 x 64`
4. downsample -> `64 x 32 x 32`
5. encoder stage 2 -> `64 x 32 x 32`
6. downsample -> `128 x 16 x 16`
7. encoder stage 3 -> `128 x 16 x 16`
8. downsample -> `256 x 8 x 8`
9. middle stage -> `256 x 8 x 8`
10. upsample + skip -> `128 x 16 x 16`
11. decoder stage 1 -> `128 x 16 x 16`
12. upsample + skip -> `64 x 32 x 32`
13. decoder stage 2 -> `64 x 32 x 32`
14. upsample + skip -> `32 x 64 x 64`
15. decoder stage 3 -> `32 x 64 x 64`
16. output conv -> `1 x 64 x 64`
17. final prediction -> `input + residual`

## NAF block

Each block contains:

1. `LayerNorm2d`
2. pointwise expansion convolution
3. depthwise `3x3` convolution
4. `SimpleGate`
5. simple channel attention
6. pointwise projection back to the original width
7. residual scaling with learnable `beta`
8. a second feed-forward style branch with another learnable residual scale `gamma`

This gives stronger local modeling than the current CTformer while keeping the network lightweight enough for an RTX 3060 Laptop GPU.

Compared with the current local CTformer implementation, the key advantages are:

- stronger local inductive bias for edges and textures
- multi-scale receptive field without relying on very deep attention
- residual prediction that fits denoising well
- better optimization stability with restoration-oriented blocks
- no dependence on tiling-specific transformer token design

## Training recipe

Recommended training configuration:

- optimizer: `AdamW`
- learning rate: `2e-4`
- weight decay: `1e-4`
- loss: hybrid restoration loss
  - `0.8 * Charbonnier`
  - `0.2 * SSIM loss`
- scheduler: cosine annealing
- patch training: `64x64`
- batch size: `4`
- workers: `0` or `2` on Windows

Detailed training loop:

1. load one full low-dose / full-dose slice pair
2. randomly sample `patch_n=4` paired `64x64` patches
3. apply the same augmentation to input and target patch
4. stack patches into the effective batch
5. predict clean patch with residual U-Net
6. compute hybrid loss
7. backpropagate with `AdamW`
8. step the cosine scheduler every iteration
9. save checkpoint every `500` iterations
10. evaluate later on the same `L506` holdout patient

The hybrid loss is:

`0.8 * Charbonnier + 0.2 * SSIM loss`

Where:

- `Charbonnier` behaves like a smooth `L1` loss and is robust to outliers
- `SSIM loss` adds structural consistency so the network preserves organ boundaries and low-contrast textures better than pure pixel-wise loss

## Why this training setup

- `Charbonnier` is more robust than plain MSE for denoising and preserves edges better.
- `SSIM` adds structural guidance, which matters for CT textures and organ boundaries.
- `AdamW + cosine` is usually easier to optimize than the current `Adam + fixed halving` setup.
- The residual U-Net shape is a better bias than the current shallow transformer for this dataset size.

Recommended practical schedule on your machine:

- warm start benchmark: `100-300` iterations
- short pilot: `30-50` epochs
- main run: `150-200` epochs
- optional longer run if validation/test keeps improving: `250-300` epochs

Because this model is heavier than the current CTformer, it is worth watching:

- GPU memory
- average iteration time
- whether SSIM improves together with PSNR
- whether the test patient starts to show over-smoothing

## Expected improvement

Reasonable expectation on the same split and test patient:

- versus current CTformer:
  - `+0.3 dB` to `+0.8 dB` PSNR
  - `+0.003` to `+0.010` SSIM
- versus current RED-CNN:
  - `0.0 dB` to `+0.4 dB` PSNR if training converges well
  - `0.001` to `+0.006` SSIM

This is an informed estimate, not a guarantee. RED-CNN is already quite strong on this exact setup, so beating it is plausible but not automatic.

The current local baseline numbers on the same holdout patient are:

- `CTformer`: `PSNR 32.3852`, `SSIM 0.9026`, `RMSE 9.8366`
- `REDCNN`: `PSNR 32.6656`, `SSIM 0.9067`, `RMSE 9.4867`

So the realistic target for this new model is:

- first beat CTformer reliably
- then try to match or exceed RED-CNN
- target zone:
  - PSNR around `32.8` to `33.1`
  - SSIM around `0.908` to `0.913`
  - RMSE below `9.4`

## Suggested training command

```powershell
& 'E:\Anaconda\envs\CTformer\python.exe' .\main.py `
  --model_name ctnafnet `
  --saved_path '.\npy_img_3mm_B30' `
  --save_path '.\runs\ctnafnet_v1' `
  --device cuda `
  --batch_size 4 `
  --patch_n 4 `
  --patch_size 64 `
  --num_epochs 201 `
  --print_iters 20 `
  --save_iters 500 `
  --decay_iters 0 `
  --scheduler cosine `
  --optimizer adamw `
  --loss_name hybrid `
  --lr 2e-4 `
  --weight_decay 1e-4 `
  --num_workers 0 `
  --result_fig false
```
