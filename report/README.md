# report/

Draft LaTeX sections and notes for the Overleaf report (Computer Vision template).
Overleaf is the master copy. Files here are drafts for each member to paste in, so the text stays reviewable in Git.

## Who writes what (team guide)
| Section | Author | File here |
|---|---|---|
| Introduction and problem statement | Abhishek | `sections/01_introduction.tex` |
| Dataset, preprocessing and baseline CNN | Abhishek | `sections/03_dataset_baseline.tex` |
| Related work (one paragraph each) | All four | `sections/related_work_abhishek.tex` (Abhishek's paragraph), `sections/related_work_ajit.tex` (Ajit's paragraph) |
| References | All four | `references_abhishek.bib` (Abhishek's entries), `references_ajit.bib` (Ajit's entries) |
| DenseNet-121 method, training, comparison | Ajit | `sections/04_densenet_comparison.tex` |
| Grad-CAM, lung masks, attention metric, shortcut findings | Aman | – |
| Masking experiment, final results, discussion, abstract, conclusion | Himanshu | – |

## Putting Abhishek's sections into Overleaf
1. Copy each `.tex` file's content under the matching template heading. Rename `\section{...}` to the template's own headings if they differ.
2. Add the entries from `references_abhishek.bib` to the shared `.bib` file, skipping keys a teammate already added.
3. Upload these figures to Overleaf at the paths used in the text. They come only from `results/`:

   | Overleaf path | Repo file |
   |---|---|
   | `figures/stage1/sample_grid.png` | `results/stage1_eda/sample_grid.png` |
   | `figures/stage1/image_sizes.png` | `results/stage1_eda/image_sizes.png` |
   | `figures/stage1/training_curves.png` | `results/stage1_baseline/training_curves.png` |
   | `figures/stage1/confusion_matrices.png` | `results/stage1_baseline/confusion_matrices.png` |

4. Packages used: `graphicx`, `booktabs` (`\toprule`), `cleveref` (`\cref`), `amsmath`. If the template has no `cleveref`, add `\usepackage[capitalize]{cleveref}` after `hyperref`, or replace `\cref{x}` with `Section~\ref{x}` / `Table~\ref{x}` / `Figure~\ref{x}`.
   `\etal` is defined by CVPR-style templates; otherwise use `et~al.`
5. `figure*` (the sample grid) spans both columns in a two-column template. Use `figure` if the template is one column.

## Rules for every section
- Every number must match a file in `results/` or `data/splits/` (the source file is named in a comment at the top of each section).
- Number every figure and table and refer to each one in the text.
- `% TODO` comments mark sentences that need another stage's results.
