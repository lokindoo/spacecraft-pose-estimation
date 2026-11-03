# P1 arXiv extension: pre-registered predictions

**Status: LOCKED.**
- **Commits:**
  - the decision rules (the template) were committed as `ee9ff04`, Fri 30 Oct 11:31 UK;
  - the predictions were committed verbatim as `36eeb43`, Fri 30 Oct 11:54 UK.
- **This version** polishes formatting and wording only. The diff against `36eeb43` shows that no prediction changed.
- **No output** listed below had been opened when any of these commits was made.

**Provenance.**
- The template was written by Claude on Thu 29 Oct 2026, after the analyses had run but before anyone opened their outputs.
- The predictions were entered by Nikolay on Fri 30 Oct.
- **†** marks a threshold or reading Claude chose to make a prediction testable. All were left unchanged at lock.
- Fields left blank at lock read *no prediction*.
- Conventions (paired, resolved, unresolved) are as in [P1_revision_predictions.md](P1_revision_predictions.md).

**Locked at:** Fri 30 Oct 2026, 11:54 UK, commit `36eeb43`.

**State when the template was written (Thu 29 Oct, ~13:00 UK):**
- **Frozen + aug, seeds 43 and 44:** training on the M5 since 11:46, unopened. ETA about 17:00.
- **§2 and §3 below:** computed at ~12:30. Claude has seen only the reproduction checks: n and mean E_R per record, matching the published tables, and per-bin counts summing to the marginal. Files:
  - `~/spacecraft-data/artifacts/results/rotation_error_shape.csv`
  - `~/spacecraft-data/artifacts/results/rotation_error_histograms.csv`
  - `~/spacecraft-data/artifacts/figures/rotation_error_histograms.png`
  - `~/spacecraft-data/artifacts/results/coverage_by_range.csv`
- **§4:** `results/revision/conformal_ablation.csv` and `conformal_heads.csv` were computed on 28 Oct. Neither table is in the camera-ready. Claude has seen only their first two rows (in-domain translation for arm B seed 42 and one head).

**Update before lock (Fri 30 Oct, 11:08 UK):**
- Frozen + aug seeds 43 and 44 finished Thu 18:10.
- Their conformal coverage went to `~/spacecraft-data/artifacts/results/conformal_frozen_aug.csv`.
- The §2 and §3 files above were regenerated with the two seeds added, 17 records in all. Script: `~/spacecraft-data/step6_frozen_aug_analysis.sh`.
- Claude read only the exit codes, the in-domain gate and the bins-sum check.

**Cut-off:** results in hand by Mon 2 Nov, 12:00 UK.

---

## Summary (tracker format)

| Experiment | Prediction | Changes the wording if | Cut-off |
|---|---|---|---|
| Frozen + aug, seeds 43/44 | The seed-42 sunlamp improvement goes away as noise: seeds 43 and 44 give no conclusive evidence of a sunlamp gain. | see §1 | arXiv |
| Rotation-error shape | Unsure: no prediction. | see §2 | arXiv |
| Coverage by range bin | Unsure overall; per-item leanings in §3 (confidence 60–70%). | see §3 | arXiv |
| Coverage of arms A/B and the frozen heads | Coverage mirrors errors. | see §4 | arXiv |

---

## 1. Frozen backbone + full augmentation stack, seeds 43 and 44

**Why:** Table 1's frozen/aug row is the only remaining n = 1 row. Two claims rest on it:
- §3.2's closing sentence, "the gap can narrow merely because the reference degrades (frozen arms: 1.053 → 1.180 under augmentation)";
- Table 1 itself.

**Quantity:**
- E_R per domain, three-seed mean ± s.d.: published seed 42 (Colab), plus seeds 43 and 44 (M5, package).
- The augmentation effect for the frozen backbone, paired by seed against the frozen no-aug head retrains: aug − no-aug.

**Reference points (already seen):**

| | synthetic | lightbox | sunlamp |
|---|---|---|---|
| Frozen + aug, seed 42 (published) | 1.180 | 1.718 | 1.864 |
| Frozen no-aug, seed 42 (published) | 1.053 | 1.752 | 2.048 |
| Frozen no-aug head retrains, seeds 42 / 43 / 44 | 1.011 / 1.045 / 1.036 | 1.747 / 1.746 / 1.791 | 1.992 / 1.952 / 2.013 |
| Frozen no-aug head retrains, mean ± s.d. | 1.030 ± 0.017 | 1.761 ± 0.026 | 1.986 ± 0.031 |

**Shown before predicting:** at seed 42, augmentation helped the frozen backbone *more on sunlamp* (−0.184) than on lightbox (−0.034). That is the opposite direction to the layer-4 arms, where lightbox improved on every seed and sunlamp was unresolved. At n = 1 nobody has to explain it; at n = 3 it either goes away or qualifies the directional claim.

**Prediction:**
- **Overall:** the figures on seeds 43 and 44 are similar to seed 42's, aside from the sunlamp gain - fluctuations here are expected.
- **Sunlamp** (from the summary): the seed-42 improvement goes away as noise; seeds 43 and 44 give no conclusive evidence of a sunlamp gain.
- **Per domain,** aug − no-aug on synthetic, lightbox and sunlamp: no separate prediction.
- **Confidence:** 85–90%.

**Decision rules†:**
- **"The reference degrades":** keep the sentence if synthetic aug − no-aug is positive on 3/3 seeds. Otherwise cut the frozen example and keep the general point about gap ratios.
- **Directional claim:** if the frozen sunlamp effect is negative on 3/3 seeds *and* larger in size than the frozen lightbox effect on 3/3 seeds, the claim needs a qualifier in the arXiv version, e.g. "for the adapted backbone". The wording is Nikolay's. If the effect is mixed, report it as unresolved, as for layer-4 sunlamp.
- **Table 1:** the frozen/aug row becomes mean ± s.d. either way, with a note that seeds 43 and 44 ran on the M5.

**Pairing caveat:**
- **Two training paths.** Seeds 43/44 of the aug arm train end to end through `train.py`, while the no-aug retrains train on cached features through `train_head.py`. Both use the same head, loss, optimiser and selection rule, and the cache reproduces end-to-end evaluation to within 0.0005.
- **Two platforms.** Seed 42 of the aug arm is the Colab run. The layer-4 control showed a platform shift of up to 0.14 rad on sunlamp, so a single-seed difference of that size is not informative.

## 2. Rotation-error shape (M2Lg: "bimodal? symmetry?")

**Quantity:** per domain, for the six layer-4 checkpoints (the frozen arms, arms A and B and the control are in the CSV too):
- **(a) Near-π share:** the share of errors above π − 0.3 rad. The uniform-guessing reference is 0.19: chance alone puts that much mass near π, because the angle density (1 − cos θ)/π peaks at π.
- **(b) Shape:** whether the histogram has a second mode near π, separate from the main one.
- **(c) Error axes** among the near-π errors:
  - concentration in the **body** frame, the signature of a flip about a near-symmetry axis;
  - concentration in the **camera** frame, the signature of an in-plane (roll) confusion about the boresight.
  - Measured as the share of axes within 20° of the principal axis, against 0.06 for random axes.

**Reference points:** none. The camera-ready makes no claim about error shape.

**Prediction:** unsure, so none.
- (a) Near-π share on sunlamp and on lightbox, relative to 0.19: no prediction.
- (b) Bimodal on a real domain: no prediction.
- (c) Which frame concentrates, and about which axis: no prediction.

**Decision rules†:**
- **"Bimodal"** if, on a domain, all six layer-4 checkpoints have a near-π share at least 1.5× the uniform reference (≥ 0.28). Otherwise the text says real-domain errors are unimodal and broad, and the mass near π is no more than chance puts there.
- **"Symmetry flips"** if the body-frame share in cone is ≥ 0.18 (3× random) on all six checkpoints *and* the camera-frame share is not. Name the axis.
- **"In-plane confusion"** if the reverse holds.
- **Neither:** say so. That is an answer to M2Lg too.

**Frame convention:** taken from `scripts/image_utils.py`, where the quaternion's matrix maps body to camera. If that is backwards, "body" and "camera" swap labels and nothing else changes.

## 3. Coverage by range bin (kVr6; the Limitations claim)

**Quantity:** coverage of the fixed, marginal 90% q̂ within five range bins (quintiles of the calibration split's range), on `in_domain_test`, lightbox and sunlamp, for all three score families, with Clopper–Pearson intervals. In-domain bins hold about 200 images each, so each interval is roughly ±4–5 points.

**The claim under test** (camera-ready Limitations): "fixed metre-radius spheres under-cover distant targets even in-domain, and range-normalising overcorrects."

**Prediction** (summary: unsure overall; these are the leanings):
- **In-domain, metres score:** coverage falls with range, and the far bin's interval excludes 90%. **Agree.**
- **In-domain, range-normalised score:** coverage rises with range, and the near bin under-covers. **Agree.**
- **In-domain, rotation:** **worse at far range**, rather than flat.
- **Real domains:** the collapse is **uniform across range**, rather than concentrated in some bins.
- **Confidence:** 60–70%.

**Decision rules†:**
- **Both in-domain translation patterns hold**, with the extreme bin's interval excluding 90% on most arms: keep the Limitations sentence and cite the new appendix table.
- **Monotone, but intervals include 90%:** soften it to "tend to".
- **Not monotone:** cut the sentence.

## 4. Coverage for arms A and B and for the frozen heads

**Quantity:**
- Rotation coverage on lightbox and sunlamp:
  - arms A and B, per seed;
  - the synthetic-BN and tuned frozen heads, three seeds each, each with its own q̂.
- These were computed on 28 Oct but are in neither the camera-ready nor its appendix.

**Reference points (already seen):** layer-4 rotation coverage, published:
- no aug: lightbox 21–27%, sunlamp 9–13%;
- full stack: lightbox 33–38%, sunlamp 13–15%.

**Prediction:** coverage mirrors errors.
- **Arms A and B:** lightbox coverage drops. The question is whether coverage follows error per seed; arm A worsened lightbox error on all three seeds.
- **Synthetic-BN head:** its cone is not wider, although its error is worse everywhere.
- **Confidence:** 50–60%.

**Decision rule†:** if per-seed coverage moves opposite to per-seed error (lower error, higher coverage) on ≥ 5 of 6 arm-A/B checkpoints per domain, the body can say "coverage mirrors the error ordering across the ablation too". Otherwise report the numbers in the appendix without that sentence.

---

## Not pre-registered (already visible)

- **The transfer plot** (`make_figures.py --transfer`): every point is a number already in Table 1 or the appendix. Only the picture is new.
- **AdaBN translation:** already in camera-ready Table 4. Sunlamp E_T falls on all six checkpoints, by 0.07–0.20, to between 0.18 and 0.21, below the constant predictor's 0.372. Interpreting it is a wording decision, not a prediction.
