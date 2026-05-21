$ErrorActionPreference = "Stop"

$Root = Resolve-Path "."
$ReportDir = Join-Path $Root "report"
$SourceDocx = Join-Path $ReportDir "Appendix_6_filled_ST4001_senior_thesis_draft_v3_residual.docx"
$TargetDocx = Join-Path $ReportDir "Appendix_6_filled_ST4001_senior_thesis_draft_v4_standardized.docx"
$TargetPdf = Join-Path $ReportDir "Appendix_6_filled_ST4001_senior_thesis_draft_v4_standardized.pdf"
$FigureDir = Join-Path $ReportDir "figures\standardized"

Copy-Item -LiteralPath $SourceDocx -Destination $TargetDocx -Force

$word = $null
$doc = $null

function Replace-AllText {
    param(
        [Parameter(Mandatory = $true)][string]$FindText,
        [Parameter(Mandatory = $true)][string]$ReplaceText
    )

    $range = $script:doc.Content
    $find = $range.Find
    $find.ClearFormatting() | Out-Null
    $find.Replacement.ClearFormatting() | Out-Null
    $find.Text = $FindText
    $find.Replacement.Text = $ReplaceText
    $find.Forward = $true
    $find.Wrap = 1
    $find.Format = $false
    $find.MatchCase = $false
    $find.MatchWholeWord = $false
    $find.Execute($FindText, $false, $false, $false, $false, $false, $true, 1, $false, $ReplaceText, 2) | Out-Null
}

function Move-SelectionBeforeText {
    param([Parameter(Mandatory = $true)][string]$FindText)

    $range = $script:doc.Content
    $find = $range.Find
    $find.ClearFormatting() | Out-Null
    $find.Text = $FindText
    $find.Forward = $true
    $find.Wrap = 0
    $find.Format = $false
    $ok = $find.Execute()
    if (-not $ok) {
        throw "Could not find insertion point: $FindText"
    }

    $insertRange = $script:doc.Range($range.Start, $range.Start)
    $insertRange.Select()
}

function Type-Para {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [bool]$Bold = $false,
        [int]$Size = 12,
        [int]$Alignment = 3
    )

    $sel = $script:word.Selection
    $sel.Font.Name = "Times New Roman"
    $sel.Font.Size = $Size
    $sel.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $sel.ParagraphFormat.Alignment = $Alignment
    $sel.TypeText($Text)
    $sel.TypeParagraph()
    $sel.Font.Bold = 0
}

function Type-BodyBlock {
    param([Parameter(Mandatory = $true)][string[]]$Paragraphs)

    foreach ($p in $Paragraphs) {
        if ($p.Trim().Length -eq 0) {
            $script:word.Selection.TypeParagraph()
        }
        else {
            Type-Para -Text $p -Bold:$false -Size 12 -Alignment 3
        }
    }
}

function Add-SimpleTable {
    param(
        [Parameter(Mandatory = $true)][string]$Caption,
        [Parameter(Mandatory = $true)][string[]]$Headers,
        [Parameter(Mandatory = $true)][object[]]$Rows
    )

    Type-Para -Text $Caption -Bold:$true -Size 10 -Alignment 1

    $sel = $script:word.Selection
    $table = $script:doc.Tables.Add($sel.Range, $Rows.Count + 1, $Headers.Count)
    $table.Borders.Enable = 1
    $table.Range.Font.Name = "Times New Roman"
    $table.Range.Font.Size = 9

    for ($c = 0; $c -lt $Headers.Count; $c++) {
        $cell = $table.Cell(1, $c + 1).Range
        $cell.Text = $Headers[$c]
        $cell.Font.Bold = -1
        $cell.ParagraphFormat.Alignment = 1
    }

    for ($r = 0; $r -lt $Rows.Count; $r++) {
        $row = $Rows[$r]
        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $cell = $table.Cell($r + 2, $c + 1).Range
            $cell.Text = [string]$row[$c]
            $cell.Font.Bold = 0
            $cell.ParagraphFormat.Alignment = 0
        }
    }

    $table.AutoFitBehavior(2)
    $after = $table.Range
    $after.Collapse(0)
    $after.Select()
    $script:word.Selection.TypeParagraph()
}

function Add-Figure {
    param(
        [Parameter(Mandatory = $true)][string]$ImagePath,
        [Parameter(Mandatory = $true)][string]$Caption,
        [double]$MaxWidth = 430
    )

    $sel = $script:word.Selection
    $sel.ParagraphFormat.Alignment = 1
    $shape = $sel.InlineShapes.AddPicture($ImagePath, $false, $true)
    $shape.LockAspectRatio = -1
    if ($shape.Width -gt $MaxWidth) {
        $shape.Width = $MaxWidth
    }
    $sel.TypeParagraph()
    Type-Para -Text $Caption -Bold:$true -Size 10 -Alignment 1
}

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($TargetDocx)
    $script:word = $word
    $script:doc = $doc

    Replace-AllText "Table 1 shows the current main results" "Table 5.1 shows the current main results"
    Replace-AllText "Table 1. Quantitative results on the held-out L506 test patient." "Table 5.1. Quantitative results on the held-out L506 test patient."
    Replace-AllText "Figure 1. Metric comparison of low-dose input and denoising models on the held-out L506 patient." "Figure 5.1. Metric comparison of low-dose input and denoising models on the held-out L506 patient."
    Replace-AllText "Figure 2. Representative qualitative result for slice L506_88." "Figure 5.2. Representative qualitative result for slice L506_88."
    Replace-AllText "Figure 3. Training loss trend from the reproduced CTformer run." "Figure 5.3. Training loss trend from the reproduced CTformer run."
    Replace-AllText "Figure 4. Residual noise analysis for L506_88." "Figure 6.1. Original residual noise overview for L506_88."
    Replace-AllText "Table 2. Residual noise statistics for representative slice L506_88." "Table 6.1. Residual noise statistics for representative slice L506_88."
    Replace-AllText "Figure 5. Histogram of removed residual values for L506_88." "Figure 6.6. Histogram of removed residual values for L506_88."
    Replace-AllText "Figure 6. Radially averaged residual power spectrum for L506_88." "Figure 6.7. Radially averaged residual power spectrum for L506_88."

    Move-SelectionBeforeText "1.4 Thesis Organization"
    Type-BodyBlock @(
        "In this thesis, the contribution is deliberately framed as an engineering research contribution rather than a simple benchmark reproduction. The first contribution is a complete paired LDCT denoising pipeline: data preparation, patient-level splitting, model training, checkpoint management, tiled inference, visual comparison, and quantitative evaluation are connected into one reproducible workflow. The second contribution is the adaptation of a Restormer-style restoration Transformer, named CTRestormer in this thesis, to full-resolution CT denoising under limited local GPU memory. The third contribution is the residual noise analysis protocol, which compares the true low-dose residual with the residual removed by each model. This analysis helps identify whether a method wins mainly by suppressing noise-like texture or by over-smoothing anatomical edges.",
        "These contributions are important because LDCT denoising should not be judged only by global PSNR or SSIM. A model can obtain a strong pixel metric while still producing clinically undesirable smoothing or structure leakage. Therefore, this thesis evaluates both image fidelity and residual behavior. The final discussion treats CTRestormer not as an isolated neural network, but as a practical denoising system whose output, residual maps, and failure risks can be inspected."
    )

    Move-SelectionBeforeText "Chapter 3. Methodology"
    Type-Para -Text "2.5 Literature Gap and Thesis Position" -Bold:$true -Size 13 -Alignment 0
    Type-BodyBlock @(
        "The reviewed literature shows three limitations that motivate the design of this thesis. First, CNN-based LDCT denoising methods such as RED-CNN are effective and computationally stable, but their local convolutional operators may require deep stacks to model long-range anatomical context. Second, Transformer-based image restoration methods improve global context modeling, but their memory consumption can become problematic for 512 x 512 CT slices. Third, many LDCT denoising papers report PSNR, SSIM, and RMSE, while giving less attention to what signal is removed from the noisy image. SSIM itself was designed to measure structural similarity rather than clinical correctness [8], so it should be interpreted together with qualitative and residual evidence.",
        "This thesis is positioned at the intersection of these gaps. It keeps RED-CNN as a strong CNN baseline, uses CTformer as a Transformer-based LDCT reference, and adapts the Restormer restoration idea to CT denoising. The attention mechanism is motivated by the general Transformer framework [9] and later vision Transformer developments [10], while the residual analysis is motivated by the need to understand denoising behavior beyond aggregate metrics. The discussion of future Noise2Noise-style work is also connected to blind or self-supervised denoising methods such as Noise2Self and Noise2Void [11], [12]."
    )

    Move-SelectionBeforeText "Chapter 4. Experimental Setup"
    Type-Para -Text "3.5 Implementation and Reproducibility Details" -Bold:$true -Size 13 -Alignment 0
    Type-BodyBlock @(
        "The implementation is designed around paired supervised restoration. Each input slice is a low-dose CT image and each target slice is the corresponding normal-dose CT image. During training, patches are sampled from the paired slices to reduce GPU memory usage and to increase the diversity of local anatomical patterns observed by the network. During evaluation, the full 512 x 512 slice is reconstructed through tiled inference. This is necessary because Transformer-based restoration models may exceed local GPU memory when a complete CT slice is processed at once.",
        "For CTRestormer, the network predicts a restored image through a residual restoration design. The residual design is useful because the expected output should preserve most anatomical structures from the input while removing dose-induced noise. The training therefore encourages the model to learn a correction between LDCT and NDCT rather than hallucinating an entirely new image. In the evaluation stage, PSNR, SSIM, and RMSE are computed on the held-out L506 patient. The residual analysis additionally computes input minus target, input minus prediction, and prediction minus target, making it possible to compare the removed residual with the estimated true noise."
    )

    Move-SelectionBeforeText "Chapter 5. Experimental Results"
    Type-Para -Text "4.4 Dataset and Configuration Summary" -Bold:$true -Size 13 -Alignment 0
    Add-SimpleTable -Caption "Table 4.1. Dataset split and evaluation protocol." -Headers @("Item", "Setting", "Purpose") -Rows @(
        @("Dataset", "AAPM-Mayo 3mm B30 paired LDCT/NDCT slices", "Supervised low-dose to normal-dose CT restoration"),
        @("Patients", "L067, L096, L109, L143, L192, L286, L291, L310, L333, L506", "Patient-level organization of paired slices"),
        @("Test patient", "L506", "Held-out evaluation to avoid slice-level leakage"),
        @("Image size", "512 x 512", "Full CT slice denoising and tiled inference"),
        @("Metrics", "PSNR, SSIM, RMSE, residual correlation, edge leakage", "Image quality and residual behavior evaluation")
    )
    Add-SimpleTable -Caption "Table 4.2. Model and inference configuration summary." -Headers @("Component", "Configuration", "Reason") -Rows @(
        @("RED-CNN", "CNN denoising baseline", "Representative convolutional LDCT denoiser"),
        @("CTformer", "Transformer LDCT baseline checkpoint", "Comparison with a Transformer-specific CT model"),
        @("CTRestormer", "Restormer-style encoder-decoder with residual output", "Proposed practical restoration model for LDCT"),
        @("Training", "Patch-based training with memory-aware settings", "Allows training under local GPU limitations"),
        @("Inference", "Tiled full-slice inference", "Avoids out-of-memory errors for 512 x 512 slices")
    )
    Type-BodyBlock @(
        "These tables are included to make the experimental setup easier to audit. They also separate experimental design from experimental results, which improves thesis readability and makes the later metric tables easier to interpret."
    )

    Move-SelectionBeforeText "6.3 Noise Texture and Frequency Analysis"
    Type-Para -Text "6.2.2 Standardized Residual Figure Presentation" -Bold:$true -Size 13 -Alignment 0
    Type-BodyBlock @(
        "The original residual overview is useful for a quick inspection, but it is visually dense. For thesis presentation, the residual analysis is therefore split into standardized figures. The prediction comparison, removed residual maps, prediction error maps, and residual metric bars are separated so that each figure has one visual purpose. The same CT display window is used for prediction comparison, and shared residual scales are used for residual maps where appropriate.",
        "Figure 6.2 compares the low-dose input, normal-dose target, and denoised predictions. Figure 6.3 focuses only on what each model removes from the low-dose input. This distinction is important: a visually smooth denoised image is not necessarily better if the removed residual contains anatomical edges. Figure 6.4 therefore shows the prediction error relative to the normal-dose target, while Figure 6.5 summarizes residual correlation and edge leakage numerically."
    )
    Add-Figure -ImagePath (Join-Path $FigureDir "L506_88_prediction_comparison_standard.png") -Caption "Figure 6.2. Standardized prediction comparison for L506_88 using a shared CT display window." -MaxWidth 430
    Add-Figure -ImagePath (Join-Path $FigureDir "L506_88_removed_residuals_standard.png") -Caption "Figure 6.3. Removed residual maps computed as input minus prediction. The panels use a shared diverging residual scale to compare removal strength." -MaxWidth 430
    Add-Figure -ImagePath (Join-Path $FigureDir "L506_88_prediction_errors_standard.png") -Caption "Figure 6.4. Prediction error maps computed as prediction minus target for L506_88." -MaxWidth 430
    Add-Figure -ImagePath (Join-Path $FigureDir "L506_88_residual_metrics_bar_standard.png") -Caption "Figure 6.5. Residual correlation and edge leakage comparison for the representative L506_88 slice." -MaxWidth 390

    Move-SelectionBeforeText "Chapter 7. Conclusion"
    Type-Para -Text "6.5 Limitations of the Current Residual Study" -Bold:$true -Size 13 -Alignment 0
    Type-BodyBlock @(
        "The residual analysis strengthens the thesis because it checks model behavior beyond PSNR and SSIM. However, the current residual visual figures are based on a representative slice, L506_88. Therefore, they should be interpreted as a detailed case study rather than complete proof over the entire test patient. A stronger final version can extend the same residual metrics to all L506 slices and report the mean and standard deviation of residual correlation, removed residual standard deviation, and edge leakage.",
        "Another limitation is that residual similarity to input minus target is only an approximation of desirable denoising. The normal-dose target is treated as the reference, but clinical image quality also depends on lesion visibility, tissue boundary preservation, and reader confidence. Future work should therefore combine residual maps with ROI-based measurements and, if possible, clinical task-based evaluation."
    )

    Move-SelectionBeforeText "Appendix"
    Type-BodyBlock @(
        "[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, ""Image quality assessment: From error visibility to structural similarity,"" IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.",
        "[9] A. Vaswani et al., ""Attention is all you need,"" Advances in Neural Information Processing Systems, 2017, pp. 5998-6008.",
        "[10] A. Dosovitskiy et al., ""An image is worth 16x16 words: Transformers for image recognition at scale,"" International Conference on Learning Representations, 2021.",
        "[11] J. Batson and L. Royer, ""Noise2Self: Blind denoising by self-supervision,"" International Conference on Machine Learning, 2019.",
        "[12] A. Krull, T.-O. Buchholz, and F. Jug, ""Noise2Void: Learning denoising from single noisy images,"" Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.",
        "[13] E. Kang, W. Chang, J. Yoo, and J. C. Ye, ""Deep convolutional framelet denoising for low-dose CT via wavelet residual network,"" IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1358-1369, 2018."
    )

    $doc.Save()
    $doc.ExportAsFixedFormat($TargetPdf, 17)
    $doc.Close($false)
    $word.Quit()

    Write-Output "Updated thesis DOCX: $TargetDocx"
    Write-Output "Exported PDF: $TargetPdf"
}
catch {
    if ($doc -ne $null) {
        try { $doc.Close($false) | Out-Null } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() | Out-Null } catch {}
    }
    throw
}
