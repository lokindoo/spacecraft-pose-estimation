# arXiv version: source and number ledger

The LaTeX source of the extended version as submitted to arXiv, with the
compiled [article.pdf](article.pdf), the script that writes its tables, and the
script that checks every number in it against [`results/`](../../results/).

## Check the numbers

```bash
python check_numbers.py            # from this directory
python check_numbers.py --verbose  # also list the claims that pass
```

Each claim names the exact phrase it checks and the formula that should
produce its numbers from the results CSVs and the per-image predictions. A
claim fails if a number is off by more than its stated precision allows, or if
the phrase is no longer in the text. On this source: 83 claims plus structural
checks, 0 failures.

## Rebuild

```bash
python make_tables.py              # regenerates tables/ from results/
pdflatex article && pdflatex article
```

`article.bbl` is included, so BibTeX is not needed. Only the five figures the
paper uses are here. `scripts/make_figures.py` regenerates Figures 1, 2, 4 and
5, and `scripts/rotation_errors.py` Figure 3, the rotation-error densities.

## What differs from the submitted source

- **`make_tables.py` and `check_numbers.py`:** only the line that locates
  `results/` changed, to fit this repository's layout.
