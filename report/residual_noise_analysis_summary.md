# Residual Noise Analysis Summary

This analysis was performed on representative held-out test slice `L506_88`.

## Generated Files

- Residual panel: `report/figures/residual_noise/L506_88_residual_panel.png`
- Residual histogram: `report/figures/residual_noise/L506_88_residual_histogram.png`
- Residual power spectrum: `report/figures/residual_noise/L506_88_residual_power_spectrum.png`
- Metrics table: `report/figures/residual_noise/L506_88_residual_metrics.csv`
- Saved arrays: input, target, model predictions, and removed residual `.npy` files in `report/figures/residual_noise/`

## Metrics

| Model | Residual Corr. to `input-target` | Edge Leakage Corr. | Removed STD | Removed MAE | Error MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CTformer | 0.7259 | 0.2506 | 9.8406 | 4.3490 | 4.1493 |
| CTRestormer | 0.7300 | 0.3699 | 11.3863 | 5.2081 | 4.1636 |
| RED-CNN | 0.7358 | 0.3683 | 11.7800 | 5.3967 | 4.1823 |

## Interpretation

All three models remove residuals that are positively correlated with the reference low-dose residual `input-target`, which suggests that they are removing real low-dose noise patterns rather than arbitrary content.

CTformer is more conservative: it removes a lower-amplitude residual and has the lowest edge leakage correlation. CTRestormer removes a stronger residual than CTformer and has slightly higher correlation with the reference residual, but its edge leakage correlation is also higher and close to RED-CNN. This means CTRestormer is more aggressive: it may remove noise more effectively, but the residual maps should be inspected to ensure that anatomical edges are not being suppressed.

For the thesis, the safest conclusion is:

> CTRestormer achieves the best PSNR/SSIM among current checkpoints and removes residual patterns similar to the low-dose reference residual. However, its higher edge leakage score suggests that the model is more aggressive than CTformer, so its numerical improvement should be interpreted together with visual residual maps and structure-preservation analysis.

## Recommended Next Step

Extend this analysis from one slice to all L506 test slices and report the average and standard deviation of:

- residual correlation to `input-target`
- edge leakage correlation
- removed residual standard deviation
- removed residual MAE
- prediction error MAE
