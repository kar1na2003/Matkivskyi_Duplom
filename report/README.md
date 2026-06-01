# Report Folder

This folder contains everything needed to build and maintain the report in one place.

## Structure

- `test.tex` - main report LaTeX source
- `all_results_tables.tex` - full appendix tables
- `latexforspecification.tex` - alternative/specification draft
- `data/` - CSV/XLSX datasets used in tables and analysis
- `plots/` - generated figures
- `images/` - static images used in the report
- `scripts/` - scripts that generate report CSVs and plots
- `build/` - LaTeX build artifacts

## Generate Plots

From repository root:

```bash
cd report
python scripts/generate_report_plots.py
python scripts/generate_family_plots.py
```

## Build PDF

From repository root:

```bash
cd report
pdflatex -interaction=nonstopmode test.tex
```

If references/tables are unresolved, run `pdflatex` one more time.
