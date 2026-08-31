# Published results

The numbers in the paper, as data. `scripts/reproduce_table.py` diffs a fresh run
against these row by row and exits non-zero on any disagreement, so they are a
regression test rather than documentation.

## `results_2x2.csv`

24 rows — 8 arms × 3 domains. One row per (arm, seed, domain).

| Column | Meaning |
|---|---|
| `arm` | config name: `frozen_no_aug`, `frozen_aug`, `unfrozen_no_aug`, `unfrozen_aug` |
| `seed` | 42, 43 or 44. Frozen arms are seed 42 only (n=1). |
| `domain` | `synthetic` (validation), `lightbox`, `sunlamp` |
| `backbone` | `frozen` or `layer4` — the first factor of the 2×2 |
| `augmentation` | the second factor |
| `translation_relative` | E_T, range-normalised: ‖Δr‖ / ‖r_true‖ |
| `rotation` | E_R, geodesic angle in radians. **Chance level is 2.207.** |
| `speed_score` | S = E_T + E_R, the official SPEED+ score, unthresholded |

Table 1 of the paper is this file aggregated: mean ± s.d. over seeds for the
`layer4` rows, single values for the frozen ones.

## `conformal_coverage.csv`

96 rows — 8 arms × 4 domains × 3 metrics, at nominal 90%.

| Column | Meaning |
|---|---|
| `arm` | includes the seed, e.g. `unfrozen_aug_seed43` |
| `domain` | `in_domain_test`, `sunlamp`, `lightbox`, `synthetic` |
| `metric` | `translation_metres`, `translation_relative`, or `rotation` |
| `nominal` | 0.9 throughout |
| `q_hat` | the conformal radius, calibrated **once** on fresh synthetic data |
| `coverage` | fraction of errors actually inside that radius on this domain |
| `coverage_low`, `coverage_high` | Clopper–Pearson interval on the coverage |
| `inflation` | factor q̂ would need to grow by for honest coverage here |

Two things to know when reading it:

- **`domain = synthetic` is a sanity check, not a result.** That split was used
  to select checkpoints, so it is not exchangeable with calibration and should
  read slightly *above* nominal. `in_domain_test` is the clean control.
- **Rotation inflation is capped at π/q̂.** Rotation error cannot exceed π, so a
  small rotation inflation means either a genuinely robust model *or* a region
  that already fills most of SO(3) and has nowhere to grow. For the frozen arms
  it is the second: their 90% cone spans 2.30–2.40 rad, roughly half of all
  possible orientations. Read inflation next to the SO(3) share, never alone.

## `checkpoints.json`

Maps each arm to the checkpoint file behind its row, plus the selection rule.
Two fields are unfilled until release: `huggingface` (the uploaded path) and
`sha256` (so a reproducer can verify they have the same weights).

## Provenance

These were transcribed from `project_context/results_table.md`, which the
original Colab notebooks produced. They are the *published* values; a
reproduction run regenerates them independently and the comparison is what
matters.
