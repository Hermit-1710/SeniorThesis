# ST4001 Senior Thesis Excellent Draft Plan

> Topic: Transformer-Based Low-Dose CT Image Denoising with CTRestormer and Residual Noise Analysis

## 1. Thesis Positioning

This thesis should be written as an engineering research thesis, not a simple reproduction report. The central story should be:

1. Low-dose CT denoising is clinically important because dose reduction introduces structured noise and streak-like degradation.
2. CNN and Transformer denoisers improve image quality, but a strong thesis must evaluate not only PSNR/SSIM but also what type of signal is removed.
3. This project builds a complete local LDCT denoising pipeline, adapts a Restormer-style Transformer to CT images under limited GPU memory, and introduces residual noise analysis to interpret model behavior.

## 2. Core Contributions to Emphasize

The contribution section should be explicit and should appear in Chapter 1.

1. **End-to-end LDCT denoising pipeline.**  
   The project prepares paired low-dose/normal-dose CT slices, implements patient-level splitting, trains multiple denoising models, evaluates them with PSNR/SSIM/RMSE, and exports visual and numerical results.

2. **CTRestormer model adaptation for LDCT.**  
   The thesis adapts a Restormer-style hierarchical Transformer to CT denoising, including overlap patch embedding, multi-scale encoder-decoder restoration, residual prediction, memory-aware patch training, AMP, gradient accumulation, and tiled inference for full 512 x 512 slices.

3. **Fair comparison with CNN and Transformer baselines.**  
   RED-CNN and CTformer are used as representative CNN and Transformer baselines, allowing the proposed model to be compared across architecture families.

4. **Checkpoint and training behavior analysis.**  
   CTRestormer checkpoints at 11500, 16500, 18000, and 21500 iterations are compared to show how denoising quality evolves during training.

5. **Residual noise map analysis.**  
   The thesis compares `input - target` noise with `input - prediction` residuals. Correlation, removed residual standard deviation, edge leakage, histograms, and frequency spectra are used to study whether a model mainly removes noise or also suppresses anatomical structure.

6. **Practical reproducibility under limited hardware.**  
   The implementation records practical choices needed to run LDCT denoising on a local workstation, especially GPU memory constraints and tiled inference.

## 3. Recommended Chapter Structure

### Front Matter

Required template pages:

1. Cover
2. Checklist
3. Academic integrity commitment
4. Acknowledgements
5. Abstract
6. Table of contents

Recommended abstract structure:

1. Problem: low-dose CT reduces radiation but increases noise.
2. Method: compare RED-CNN, CTformer, and CTRestormer.
3. Experiment: AAPM-Mayo paired LDCT/NDCT slices, patient-level test split.
4. Results: CTRestormer 21500 achieves PSNR 32.6738, SSIM 0.9096, RMSE 9.4842 on L506.
5. Contribution: residual noise analysis shows denoising behavior beyond aggregate image quality metrics.

### Chapter 1. Introduction

Recommended length: 3-4 pages.

Required content:

1. Background of CT imaging, radiation dose, and LDCT noise.
2. Why denoising is needed before clinical interpretation.
3. Limitation of traditional filters and simple smoothing.
4. Deep learning trend from CNNs to Transformers.
5. Research objectives.
6. Explicit contribution list.
7. Thesis organization.

Key writing target:

The reader should understand why this project is not only about obtaining a higher PSNR, but also about evaluating whether the denoised images preserve anatomical structures.

### Chapter 2. Literature Review

Recommended length: 4-5 pages.

Suggested sections:

1. LDCT reconstruction and post-processing denoising.
2. CNN-based denoising, including RED-CNN and U-Net style methods.
3. Transformer-based image restoration and CT denoising.
4. Residual/noise analysis and interpretability for medical image denoising.
5. Summary of gaps addressed by this thesis.

Required improvement:

The current thesis should expand this chapter with at least 10 references and compare methods conceptually rather than listing papers one by one.

### Chapter 3. Methodology

Recommended length: 5-6 pages.

Suggested sections:

1. Problem formulation: paired LDCT-to-NDCT restoration.
2. Dataset and preprocessing:
   - AAPM-Mayo 3mm B30 data.
   - Ten patient IDs.
   - L506 held out for testing.
   - 512 x 512 CT slices.
   - HU normalization and display window.
3. Baseline models:
   - RED-CNN.
   - CTformer.
4. Proposed CTRestormer:
   - Overall architecture.
   - Overlap patch embedding.
   - Hierarchical encoder-decoder.
   - Transformer restoration blocks.
   - Residual prediction.
   - Tiled inference.
5. Loss functions and optimization.
6. Evaluation metrics:
   - PSNR, SSIM, RMSE.
   - Residual correlation.
   - Removed residual standard deviation.
   - Edge leakage.
   - Histogram and power spectrum.

Key writing target:

This chapter should make the implementation reproducible. A reader should know what was trained, what was held out, what metric was computed, and how full-resolution slices were inferred.

### Chapter 4. Experimental Design

Recommended length: 3-4 pages.

Suggested sections:

1. Experimental environment:
   - OS, Python/PyTorch, GPU, CUDA if available.
2. Dataset split table.
3. Training configuration table.
4. Inference configuration.
5. Evaluation protocol.
6. Experimental questions:
   - Does CTRestormer improve LDCT image quality?
   - How does it compare with RED-CNN and CTformer?
   - Does later training improve metrics?
   - What type of residual signal does each model remove?

### Chapter 5. Results

Recommended length: 5-6 pages.

Suggested sections:

1. Main quantitative comparison.
2. Checkpoint comparison for CTRestormer.
3. Qualitative CT image comparison.
4. Training curve.
5. Failure cases and visual limitations if available.

Recommended tables and figures:

1. Table 5.1 Dataset split.
2. Table 5.2 Training configuration.
3. Table 5.3 Quantitative denoising metrics.
4. Figure 5.1 Bar chart of PSNR/SSIM/RMSE.
5. Figure 5.2 Qualitative comparison on L506_88.
6. Figure 5.3 CTRestormer training loss curve.

### Chapter 6. Residual Noise Analysis and Discussion

Recommended length: 4-5 pages.

Suggested sections:

1. Motivation: why PSNR/SSIM are not enough.
2. Residual definitions:
   - True noise approximation: `input - target`.
   - Removed residual: `input - prediction`.
   - Prediction error: `prediction - target`.
3. Visual residual maps.
4. Residual statistics.
5. Frequency-domain comparison.
6. Edge leakage discussion.
7. Interpretation:
   - CTformer is more conservative in residual removal.
   - CTRestormer removes stronger residual texture.
   - RED-CNN and CTRestormer show similar high metric performance, but residual maps help distinguish behavior.
8. Limitations:
   - Current residual visual analysis is representative, not whole-dataset proof unless extended.
   - Residual correlation does not directly prove clinical safety.

Recommended figures:

1. Figure 6.1 Standardized prediction comparison.
2. Figure 6.2 Removed residual maps with shared scale.
3. Figure 6.3 Prediction error maps with shared scale.
4. Figure 6.4 Residual metric bar chart.
5. Figure 6.5 Residual histogram.
6. Figure 6.6 Residual power spectrum.

### Chapter 7. Conclusion and Future Work

Recommended length: 2-3 pages.

Required content:

1. Summary of completed work.
2. Direct response to research objectives.
3. Main findings.
4. Limitations.
5. Future work:
   - Evaluate residual metrics over all L506 slices.
   - Add ROI-based organ/tissue analysis.
   - Add pseudo Noise2Noise or self-supervised extension.
   - Add reader study or clinical task-based evaluation.
   - Improve computational efficiency.

### References

Minimum: 10 references. Recommended: 12-18 references.

Must include:

1. RED-CNN original paper.
2. CTformer paper.
3. Restormer paper.
4. AAPM-Mayo LDCT dataset/challenge source.
5. At least 2 medical image denoising survey or representative LDCT papers.
6. At least 2 Transformer/image restoration papers.
7. At least 1 metric or SSIM paper.

## 4. Figure and Table Standards

### Figure Standards

1. Number figures by chapter: Figure 5.1, Figure 5.2, Figure 6.1.
2. Every figure must be cited in the main text before or near its placement.
3. Captions should explain:
   - Dataset or slice ID.
   - Model checkpoint if relevant.
   - CT display window or residual scale.
   - Main comparison purpose.
4. CT images should use the same window across compared panels.
5. Residual maps should use a diverging colormap and shared color scale.
6. Avoid overly wide single figures. Split large panels into logical subfigures.
7. Use high-resolution PNG files and avoid screenshots with small unreadable text.

### Table Standards

1. Number tables by chapter: Table 5.1, Table 6.1.
2. Use arrows in metric headers:
   - PSNR ↑
   - SSIM ↑
   - RMSE ↓
3. Bold the best value in each metric column when appropriate.
4. Include units where meaningful, for example HU for RMSE if computed in HU-like intensity.
5. Align decimals consistently.
6. Add short notes below tables when needed, especially for test patient and checkpoint definitions.

## 5. Formatting Standards for the Word Draft

1. Use the ZJUI template as the base document.
2. Keep font and heading styles consistent.
3. Avoid mixed caption formats such as `Figure 1`, `Figure 4`, and unnumbered image labels in the same chapter.
4. Use one citation style consistently.
5. Keep paragraphs academic and specific; avoid vague phrases such as "the result is good" without metric or visual evidence.
6. Replace placeholder personal information before final submission:
   - Name
   - Student ID
   - Major
   - Supervisor
   - Date

## 6. Target Page Allocation

To satisfy the 20-page main body requirement with a strong margin:

1. Chapter 1: 3-4 pages
2. Chapter 2: 4-5 pages
3. Chapter 3: 5-6 pages
4. Chapter 4: 3-4 pages
5. Chapter 5: 5-6 pages
6. Chapter 6: 4-5 pages
7. Chapter 7: 2-3 pages

Recommended total main body: 26-33 pages before references.

## 7. Current Highest-Priority Writing Tasks

1. Expand Chapter 2 into a real literature review with 10+ citations.
2. Make Chapter 3 reproducible with dataset, preprocessing, architecture, and inference details.
3. Move contributions into a prominent numbered list in Chapter 1.
4. Standardize all figure and table numbering.
5. Replace the wide residual panel with clearer standardized residual figures.
6. Add a limitation paragraph explaining that current residual maps are representative unless extended to all L506 slices.
7. Add a future-work subsection on whole-test-set residual analysis and Noise2Noise-style training.
