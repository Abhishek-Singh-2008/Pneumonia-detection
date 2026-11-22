# Project report

The assembled report is maintained in [the Team 3 Overleaf project](https://www.overleaf.com/project/6b02a18da5a3f1c8ad57832d/share#6f94ea97aa960a91dbba342c538e075b273f2f39a382a43c). This folder is the local source mirror and upload bundle for the Stage 4.5 report assembly.

## Files

- `elcvia_report.tex` is the main document. It includes the introduction, all three related-work paragraphs, the Stage 1–4 sections, the conclusion, and the three reference databases.
- `sections/` contains the section sources. Stage 4 results and discussion are in `sections/06_masking_experiment.tex`.
- `references_*.bib` contains the team's references.
- `overleaf-report-upload.zip` contains the main document, section sources, bibliographies, and all figure assets at the paths used by the TeX source. Upload its contents to the existing Overleaf project to sync the local source mirror.

## Compile in Overleaf

1. Set `elcvia_report.tex` as the main document.
2. Keep the supplied `elcvia.sty` in the project root. It is part of the Overleaf template and is not duplicated in the upload bundle.
3. Compile with pdfLaTeX and BibTeX. The document uses `graphicx`, `booktabs`, `amsmath`, `hyperref`, and `cleveref`.

All reported experimental values are sourced from `results/` or `data/splits/`; each section identifies its source files in comments. The local workspace does not have a LaTeX compiler, so the latest PDF still needs to be compiled and reviewed in Overleaf.
