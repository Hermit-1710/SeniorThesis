# ZJUI LaTeX Thesis Draft

This folder is a complete LaTeX thesis project generated according to the template in:

`E:\a_ST\CTformer\Latex格式`

The project keeps the template structure:

- `main.tex`
- `zjuthesis.cls`
- `gbt7714-numerical.bst`
- `data/*.tex`
- `images/*`

Compile with XeLaTeX:

```powershell
xelatex main.tex
xelatex main.tex
```

BibTeX is not required for this draft because references are written directly in `data/ref.tex` using `thebibliography`, matching one of the options in the provided template.

Before final submission, replace the fields in `main.tex`:

- `name`
- `id`
- `supervisor`
- `major`
- `Academic title`
- `submission date`

MiKTeX was installed locally and the project was compiled successfully with XeLaTeX. The generated PDF is:

`main.pdf`

The current build is 39 pages. The log has no undefined citations or undefined references.

Recent additions:

- CTRestormer architecture figure: `images/ctrestormer_architecture.png`
- Model size table for RED-CNN, CTformer, and CTRestormer
- Same-iteration comparison between CTformer and CTRestormer at 21,500 iterations
- Expanded architecture discussion and comparison against RED-CNN and CTformer
- Updated final CTformer result at 100,000 iterations, plus refreshed metric and qualitative figures
