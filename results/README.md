# Published results

The numbers in the paper, as data. `scripts/reproduce_table.py` diffs a fresh
run against these row by row and exits non-zero on any disagreement, which makes
them a regression test rather than documentation.

## `results_2x2.csv`

24 rows: 8 arms × 3 domains, one per (arm, seed, domain).

| Column | Meaning |
|---|---|
| `arm` | config name: `frozen_no_aug`, `frozen_aug`, `unfrozen_no_aug`, `unfrozen_aug` |
| `seed` | 42, 43 or 44. Frozen arms are seed 42 only (n=1). |
| `domain` | `synthetic` (validation), `lightbox`, `sunlamp` |
| `backbone` | `frozen` or `layer4`, the first factor of the 2×2 |
| `augmentation` | the second factor |
| `translation_relative` | E_T, range-normalised: ‖Δr‖ / ‖r_true‖ |
| `rotation` | E_R, geodesic angle in radians. Chance level is 2.207. |
| `speed_score` | S = E_T + E_R, the official SPEED+ score, unthresholded |

Table 1 of the paper is this file aggregated: mean ± s.d. over seeds for the
`layer4` rows, single values for the frozen ones.

## `conformal_coverage.csv`

96 rows: 8 arms × 4 domains × 3 metrics, at nominal 90%.

| Column | Meaning |
|---|---|
| `arm` | includes the seed, e.g. `unfrozen_aug_seed43` |
| `domain` | `in_domain_test`, `sunlamp`, `lightbox`, `synthetic_old` |
| `metric` | `translation_metres`, `translation_relative`, or `rotation` |
| `nominal` | 0.9 throughout |
| `q_hat` | the conformal radius, calibrated once on fresh synthetic data |
| `coverage` | fraction of errors actually inside that radius on this domain |
| `coverage_low`, `coverage_high` | Clopper–Pearson interval on the coverage |
| `inflation` | factor q̂ would need to grow by for honest coverage here |

Two things to know when reading it:

- `domain = synthetic_old` is a sanity check rather than a result. That split
  was used to select checkpoints, so it is not exchangeable with calibration and
  should read slightly *above* nominal. `in_domain_test` is the clean control,
  and the one the paper quotes as "in-domain".
- Rotation inflation is capped at π/q̂. Rotation error cannot exceed π, so a
  small rotation inflation means either a genuinely robust model or a region
  that already fills most of SO(3) and has nowhere to grow. For the frozen arms
  it is the second: their 90% cone spans 2.30–2.40 rad, roughly half of all
  possible orientations. Read inflation next to the SO(3) share, never alone.

## `constant_baseline.csv`

Six rows: two constants × three domains. The trivial no-skill predictor is one
fixed pose, fitted on the 28,000 synthetic training labels only and scored on
every domain. Regenerate with `scripts/constant_baseline.py` (labels only; no
images, no checkpoint, no GPU).

| Column | Meaning |
|---|---|
| `constant` | `closed_form` (chordal-mean quaternion, arithmetic-mean position) or `best` (the constant that directly minimises the reported metric) |
| `domain` | `synthetic` (validation), `lightbox`, `sunlamp` |
| `translation_relative`, `rotation`, `speed_score` | scored exactly as the trained arms are |
| `n` | images in that domain |

This is what anchors "near chance" to something achievable rather than to a
distributional assumption. SPEED+ orientations are not uniform on SO(3), so the
2.207 rad uniform-guess level needs an empirical counterpart, and it lands
within 0.03 rad of it on all three domains, which is why 2.207 is used
throughout.

App. C of the paper reports this constant after five Karcher iterations on the
sphere, which refine the chordal mean this script computes: E_T 0.415 / 0.367 /
0.372, E_R 2.176 / 2.203 / 2.199. The `closed_form` rows here are that chordal
starting point, and they agree to 0.001 on every score (synthetic E_R 2.177
against the paper's 2.176); the Karcher step itself is not implemented here.
Either reads as an upper bound on the optimal constant, since both estimators
minimise squared error while the reported metric is mean unsquared error. The `best` rows are the tighter constant, and they matter for
translation, where the two differ substantially (sunlamp 0.330 against 0.372);
for rotation they agree to 0.001. The no-skill claim about sunlamp translation
holds against either, since two of the three augmented layer-4 seeds (0.405,
0.399) beat neither.

## `checkpoints.json`

Maps each arm to the checkpoint file behind its row, plus the selection rule.
Two fields are unfilled until release: `huggingface` (the uploaded path) and
`sha256` (so a reproducer can verify they have the same weights).

## Provenance

These were transcribed from `project_context/results_table.md`, which the
original Colab notebooks produced. They are the published values; a reproduction
run regenerates them independently, and the comparison is what matters.
