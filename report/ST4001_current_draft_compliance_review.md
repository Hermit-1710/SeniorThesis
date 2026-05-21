# Current Draft Compliance Review

Reviewed document:

`report/Appendix_6_filled_ST4001_senior_thesis_draft_v3_residual.docx`

Review date: 2026-05-20

## 1. Current Draft Snapshot

Automatic DOCX inspection shows:

1. Paragraphs with text: 217
2. Approximate English word-like count: 4316
3. Tables detected: 4
4. Figures detected: 9
5. References currently listed: 7

Current main chapters:

1. Chapter 1. Introduction
2. Chapter 2. Literature Review
3. Chapter 3. Methodology
4. Chapter 4. Experimental Setup
5. Chapter 5. Experimental Results
6. Chapter 6. Discussion and Extended Experiments
7. Chapter 7. Conclusion

## 2. Compliance Against Hard ST4001 Requirements

| Requirement | Current Status | Action |
|---|---:|---|
| Use ZJUI Appendix VI template | Mostly satisfied | Continue using the Word template; do not rewrite from blank. |
| Main body at least 20 pages excluding references | High risk | Current word count is likely too short. Expand literature review, methodology, experiment design, and discussion. |
| Include title, abstract, contents, main thesis, references, appendix | Mostly satisfied | Check final front matter order and template placeholders. |
| Include literature review/project introduction | Partially satisfied | Expand into a real 4-5 page review with stronger synthesis. |
| Include research work, experiments, programming/design/testing/theory | Partially satisfied | Add reproducibility details: dataset split, training config, inference protocol, implementation environment. |
| Include detailed outcome analysis | Partially satisfied | Current metrics and residual analysis are good, but need more structured interpretation and limitations. |
| Include conclusion | Satisfied but short | Expand conclusion to directly answer objectives and contributions. |
| References at least 10 | Not satisfied | Add at least 5 more scholarly references; target 12-18. |
| Chinese references <=3 | Satisfied | Current references are all English. |
| Website references <=3 | Satisfied | Current AAPM website is one website reference. |
| Plagiarism check readiness | Not ready | Rewrite related work in original synthesis style and avoid copied model-paper phrasing. |
| Final personal/template fields completed | Not satisfied | Replace name, student ID, major, supervisor, date, biography placeholders. |

## 3. Thesis Quality Review

### Strengths

1. The thesis already has a clear project topic: low-dose CT denoising.
2. It compares three model families or checkpoints: RED-CNN, CTformer, and CTRestormer.
3. It includes quantitative metrics on a held-out patient-level split.
4. It includes a meaningful new contribution: residual noise map analysis.
5. The results are specific and reproducible enough to support further writing.

### Major Gaps

1. **Contribution section still needs more authority.**  
   The thesis should state the work as a set of completed engineering/research contributions, not only as "we trained a model".

2. **Literature review is too short for an excellent thesis.**  
   It should explain the evolution from classical denoising to CNNs, from CNNs to Transformers, and from metric-only evaluation to residual/structure-preservation analysis.

3. **Methodology lacks enough reproducibility detail.**  
   A strong thesis should document preprocessing, normalization, patch sampling, training settings, tiled inference, checkpoint selection, and metric definitions.

4. **References do not meet the minimum count.**  
   Current references: 7. Required minimum: 10. Recommended target: 12-18.

5. **Figure and table numbering is inconsistent.**  
   Current captions use Figure 1, Figure 2, Table 1, Table 2. For thesis quality, use chapter-based numbering such as Table 5.1 and Figure 6.2.

6. **Residual analysis should be carefully bounded.**  
   Current residual maps are representative for slice L506_88. Unless extended to all L506 slices, the text should not claim whole-dataset proof.

7. **Page count is likely insufficient.**  
   A 4316-word English draft is usually not enough to produce a robust 20-page main thesis body unless formatting is very loose. The target should be closer to 9000-12000 words for a polished 26-33 page main body.

## 4. Required Structural Revision

The draft should be reorganized toward this stronger structure:

1. Chapter 1: Introduction
   - Strengthen motivation.
   - Put a numbered contribution list in the chapter.
   - Make the research questions explicit.

2. Chapter 2: Literature Review
   - Expand to 4-5 pages.
   - Add at least 5 references.
   - Add a final gap-analysis paragraph connecting literature to this project.

3. Chapter 3: Methodology
   - Move dataset/preprocessing details here.
   - Add CTRestormer architecture details.
   - Add tiled inference details.
   - Define residual metrics formally.

4. Chapter 4: Experimental Design
   - Add dataset split table.
   - Add training configuration table.
   - Add evaluation protocol.
   - Separate experimental design from results.

5. Chapter 5: Results
   - Present main metrics.
   - Present checkpoint study.
   - Present qualitative examples.
   - Keep result interpretation close to figures/tables.

6. Chapter 6: Residual Noise Analysis and Discussion
   - Make residual analysis a formal analysis chapter.
   - Use standardized residual figures.
   - Add histogram and spectrum discussion.
   - Add limitations and risk of structure removal.

7. Chapter 7: Conclusion and Future Work
   - Summarize completed contributions.
   - State limitations honestly.
   - Present concrete future experiments.

## 5. Figure and Table Revision Plan

### Existing Figure/Table Issues

1. Captions are not numbered by chapter.
2. One caption-like explanatory paragraph begins with "Table 1 shows..." and was detected as a table caption; this should be normal body text.
3. The residual panel is visually dense and should be split into standardized figures.
4. Some captions need slice ID, checkpoint, and CT window/residual scale.

### Recommended Final Figures

1. Figure 3.1: CTRestormer pipeline or architecture schematic.
2. Figure 3.2: Residual noise analysis workflow.
3. Figure 5.1: Quantitative metric comparison on L506.
4. Figure 5.2: Representative qualitative denoising comparison for L506_88.
5. Figure 5.3: CTRestormer or CTformer training loss curve.
6. Figure 6.1: Standardized prediction comparison on L506_88.
7. Figure 6.2: Removed residual maps with shared residual scale.
8. Figure 6.3: Prediction error maps relative to NDCT target.
9. Figure 6.4: Residual metrics bar chart.
10. Figure 6.5: Residual histogram.
11. Figure 6.6: Residual power spectrum.

### Recommended Final Tables

1. Table 3.1: Dataset and split summary.
2. Table 4.1: Training and inference configuration.
3. Table 5.1: Main quantitative denoising results.
4. Table 5.2: CTRestormer checkpoint comparison.
5. Table 6.1: Residual noise statistics for L506_88.
6. Table 7.1: Limitations and future-work mapping if needed.

## 6. Reference Expansion Targets

Add references in these categories:

1. LDCT denoising survey or representative medical image denoising paper.
2. Image quality metric paper, especially SSIM.
3. Original Transformer paper if Transformer background is discussed.
4. Vision Transformer or restoration Transformer paper.
5. Noise2Void/Noise2Self/self-supervised denoising reference if future work discusses Noise2Noise-style training.
6. CT reconstruction or iterative reconstruction background reference if Chapter 1 discusses dose/reconstruction.

Recommended target count: 12-18 references.

## 7. Immediate Fix List

1. Generate and insert standardized residual figures.
2. Rename all captions to chapter-based numbering.
3. Add dataset split and training configuration tables.
4. Expand Chapter 2 by at least 1500-2500 words.
5. Expand Chapter 3 by at least 1500-2000 words.
6. Expand Chapter 6 by at least 1000-1500 words.
7. Add at least 5 references, then cite them in text.
8. Replace placeholder personal information.
9. Export final Word and PDF versions.
