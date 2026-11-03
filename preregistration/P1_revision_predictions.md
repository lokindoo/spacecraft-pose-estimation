# P1 revision: pre-registered predictions

**Status: locked.** Nothing under `~/spacecraft-data/artifacts` was opened before this file was committed.

**Provenance.** Your rough predictions are from the chat on Wed 28 Oct. Claude expanded them into this file the same day.
- **†** marks a number or reading Claude chose to make your wording testable. Change any that isn't what you meant.
- **[you]** marks a field only you can fill. "No prediction" is a valid entry.

**Locked at:** Wed 28 Oct 2026, 17:41 GMT. The commit that adds this file is the timestamp of record (`git log -- project_context/P1_revision_predictions.md`).

**State of the runs when this was written:**
- Arm A seed 42 had finished training and evaluation (Tue 27 Oct, ~23:00). Seed 43 was at epoch 18/20.
- Neither of us has opened any arm A output. Claude has seen only file names and epoch numbers.
- The control, arm B, the frozen block, AdaBN and the Fig. 1 statistics had not been run.
- At locking (17:41): arm A had finished all three seeds, still unopened; the control was at epoch 6 of 20; arm B had not started.

---

## Conventions

- **Errors:** E_R is in radians; lower is better.
- **Paired:** the per-seed difference at the same seed number (42, 43, 44).
- **Resolved:** 3/3 seeds have the same sign (the §3.2 standard). A difference that straddles zero is reported as **unresolved** (checklist 4.4).
- **Cut-offs:**
  - **CR (camera-ready):** results in hand by the freeze, Thu 29 Oct 12:00.
  - **arXiv:** results in hand by Mon 2 Nov.

**Published reference points** (E_R; layer-4 rows are three-seed mean ± s.d.):

| | synthetic | lightbox | sunlamp |
|---|---|---|---|
| Legacy 4k / logits (n = 1) | 1.666 | 2.002 | 2.073 |
| Frozen, no aug, seed 42 | 1.053 | 1.752 | 2.048 |
| Layer 4, no aug | 0.317 ± 0.010 | 1.166 ± 0.018 | 1.534 ± 0.075 |
| Layer 4, full stack | 0.322 ± 0.027 | 0.987 ± 0.010 | 1.468 ± 0.119 |
| Layer 4, full stack, seed 42 | 0.352 | 0.986 | 1.592 |
| Fitted constant (no skill) | 2.177 | 2.203 | 2.199 |

**Other published scales:**
- **Layer-4 gain** (frozen → layer 4, no aug): synthetic 0.736, lightbox 0.586, sunlamp 0.514.
- **Full stack, paired by seed:**
  - lightbox −0.183 / −0.185 / −0.170;
  - sunlamp +0.116 / −0.265 / −0.047.
- **Sunlamp E_T:** full stack 0.355 ± 0.082; the constant scores 0.372.
- **Layer-4 rotation coverage:** sunlamp 9–15%, lightbox 21–38%.

---

## Summary (tracker format)

| Experiment | Prediction | Changes the wording if | Cut-off |
|---|---|---|---|
| Arm A (no blur/noise) | Sunlamp improves marginally vs the full stack | 3/3 seeds negative: keep "blur and noise work against sunlamp outright". Otherwise soften it. | CR |
| Arm B (jitter + gamma + erasing) | Sunlamp improves more than arm A | Arm B − arm A is 3/3 negative: the glare sentence gains evidence. Otherwise soften it. | CR if all 3 seeds finish, else arXiv |
| Control | Close to published seed 42 | Outside the 2 s.d. band on a real domain: compare arms with the control and add a platform caveat. | CR |
| Frozen seeds 43/44 | Spread ≤ layer 4's | Sunlamp s.d. ≥ 0.025: "only 0.03 rad" becomes "no detectable change". | CR |
| Decomposition | Features matter more than data | Either factor alone moves sunlamp by > 2 frozen s.d.: split the asymmetry claim by factor. | CR (appendix) |
| Head sweep | Unsure. Lean: small gain, because data is the bottleneck. | Recovers ≥ 50% of the layer-4 gain on a real domain: restate the transfer finding. | CR (appendix) |
| Synthetic-BN head | No prediction | ≥ 50% of the layer-4 gain on both real domains: "adapting normalisation statistics". | CR |
| AdaBN (layer-4 BN only) | Better on every metric, both domains | Adapted rotation coverage < 80%: "lowers error, coverage stays low". | CR |
| Fig. 1 statistics | The mechanism's implications 1–4 hold | Sunlamp not sharper or more clipped than the full-stack pool: weaken the mechanism paragraph. | CR |

---

## 1. Arm A: the full stack without blur and noise

**Quantity:** sunlamp E_R, arm A − full stack, paired by seed.
- Secondary: lightbox E_R, sunlamp E_T against the constant's 0.372, and rotation coverage.

**Expectation:** sunlamp improves marginally.
- The mean paired difference falls between −0.12 and 0 rad†, i.e. less than one seed s.d. (0.119).
- Under the resolved rule, the most likely outcome is therefore **unresolved**.

**Lightbox:** still improvement, but smaller.
- What the mechanism implies: roughly unchanged, since blur and noise aren't what models diffuse light.

**Confidence:** 60-70 %

**Decision rule:**
- 3/3 negative on sunlamp: keep "Blur and noise work *against* sunlamp outright", and cite arm A.
- Otherwise: rewrite it as the hypothesis arm A was built to test, and say the ablation didn't detect the effect at three seeds. Drop "outright".
- This comparison crosses platforms: the published full stack ran in the notebooks on Colab, arm A in the package on the Mac. It holds only if the control passes (§3).

**Basis†:**
- The §3.2 mechanism says blur and noise discard the sharp structure that survives harsh light.
- But the full stack's own sunlamp effect sat inside seed noise (+0.116, −0.265, −0.047), so removing two of its components is unlikely to move sunlamp much.

## 2. Arm B: jitter, gamma and random erasing

**Definition (decided Wed 28 Oct, before arm B started):** arm B keeps random erasing and drops the glare blob, blur and noise.
- Arm B − arm A therefore isolates the glare blob.

**Quantity:** sunlamp E_R, arm B − arm A, paired by seed.
- Both arms are trained on the Mac with the package, so this comparison doesn't depend on the control.
- Also arm B − full stack.

**Expectation:** arm B improves sunlamp more than arm A does.
- Arm B − arm A has a mean between −0.12 and 0 rad†.

**Lightbox:** slightly smaller improvement
- What the mechanism implies: arm B may lose some of the lightbox gain, because the soft blob is part of what models diffuse light.

**Confidence:** 70-80 %

**Decision rule:**
- Arm B − arm A 3/3 negative on sunlamp: the glare sentence ("teaches it to ignore soft bright regions") gains evidence.
- Otherwise: soften the glare sentence, as for arm A.
- In or out (4.4): the camera-ready gets arm B only if all three seeds finish by Thu 12:00. Otherwise it goes to arXiv.

**Basis†:** the mechanism's glare sentence. You also expect the effects of the removed components to add up.

## 3. Control: full stack, seed 42, retrained on the Mac

**Quantity:** control − published seed 42, E_R per domain.
- Published seed 42: synthetic 0.352, lightbox 0.986, sunlamp 1.592.

**Expectation:** close to the published run.
- "Close" means within 2 × the full stack's seed s.d.†: synthetic ±0.053, lightbox ±0.020, sunlamp ±0.24.

**Confidence:** 95 %

**Decision rule:**
- **Inside the band on both real domains:** the pairing with the published seeds holds, and arms A and B are compared with the published full stack.
- **Outside the band on a domain:** for that domain, arms A and B are compared with the control (seed 42 only), and the paper reports how far code and hardware moved the result.
- The lightbox band is narrow (±0.020) because the published lightbox seeds agreed so closely. A miss there is easier than on sunlamp.

**Basis†:**
- The Mac reproduced all 24 Table 1 rows from the published checkpoints (worst difference 0.0006).
- That tests evaluation, though, not training. GPU nondeterminism and data order differ across platforms, so a retrain behaves at best like a fresh draw near seed 42.

## 4. Frozen seeds 43 and 44

**Quantity:** E_R s.d. across the frozen no-aug head's three seeds, per domain, compared with the layer-4 no-aug s.d. (0.010 / 0.018 / 0.075).
- The three seeds are the published seed 42 plus new seeds 43 and 44, trained on cached features.
- The head trainer's own seed 42 is also run, to check that it reproduces the published one.

**Expectation:** the frozen spread is no larger than layer 4's on each domain, and sunlamp's s.d. does not reach 0.025 - no detectable change.

**Testability:** with three seeds, an s.d. ratio anywhere from about 0.23 to 4.4 is consistent with equal true spread (90% interval). This prediction can't fail convincingly, so record it and report it descriptively.

**Confidence:** 95 %

**Decision rule** (the one that matters):
- **Frozen sunlamp s.d. ≥ 0.025:** "moved sunlamp by only 0.03 rad" (abstract and §3.1) becomes "did not detectably move sunlamp (0.03 rad, within seed variation)".
- **Below 0.025:** keep the current wording.
- Either way, Table 1's frozen no-aug row becomes mean ± s.d. Frozen aug stays at n = 1, because augmented runs can't use the feature cache.

**Basis†:** a head on fixed features has fewer sources of randomness: initialisation and data order only, with no backbone drift.

## 5. Decomposition: {4k, 28k} images × {logits, pooled} features

**Quantity:** synthetic E_R in each of the four cells, three seeds each (cheap on cached features); real-domain E_R per cell.
- The 4k cells use one fixed draw from the 28k (subset seed 0). The 4k / logits cell therefore won't exactly reproduce the legacy 1.666, which used a different 4k set.

**Expectation:** features matter more than data.
- Cell order on synthetic, best first: pooled at either size, then 28k / logits, then 4k / logits†.
- The deciding pair is **4k / pooled vs 28k / logits**: your ordering implies 4k / pooled wins.

**Real domains:** no real change compared to published figures

**Confidence:** 90-95 %

**Decision rule:**
- Split the 0.61 rad synthetic gain by factor (main effects). If one factor carries ≥ 2/3 of it, name it in the text ("mostly from the features").
- If either factor alone moves sunlamp by more than 2 × the frozen sunlamp s.d. (either direction), restate the asymmetry claim per factor. Otherwise it stands as "neither factor moved sunlamp".

**Basis†:** your notebook 4k / pooled result of ≈ 1.31 rad sits between legacy 1.666 and 28k / pooled 1.053.
- At 4k, that makes the feature switch worth ~0.36 rad and the extra data ~0.26 rad afterwards.
- Check which validation split that 1.31 used before leaning on it.

## 6. Head sweep

**Quantity:** the best configuration's synthetic E_R at seed 42, against the published 1.053; then its real-domain E_R over three seeds.
- Grid: learning rate {3e-4, 1e-3, 3e-3} × epochs {20, 60}.
- Selection on synthetic validation only.

**Expectation (unsure):** the lean is a small gain.
- The best configuration improves synthetic by less than a quarter of the frozen → layer-4 synthetic gap: < 0.18 rad, so it stays above ~0.87†.
- The real domains move less than synthetic.

**Confidence:** low; 30-40 %

**Decision rule:**
- **"The head was under-trained"** (M2Lg): the best configuration beats published seed 42 on synthetic by more than 2 × the frozen synthetic s.d. If so, the tuned head goes into Table 1 or the appendix.
- **"Closes the gap"** (4.4): the tuned head recovers ≥ 50% of the layer-4 no-aug gain on a real domain (lightbox ≥ 0.29 rad, sunlamp ≥ 0.26 rad).
  - If so, restate the transfer finding: part of the frozen-vs-layer-4 contrast was head tuning.
  - If not, the finding stands and the sweep answers M2Lg.
- **Your data hypothesis isn't tested before the freeze.**
  - If the sweep is flat, state data as the remaining explanation: training used 28k of the ~60k synthetic images.
  - The decomposition's 4k → 28k change is the only evidence.
  - No 60k run; that's out of scope.

**Basis:** you think data, not optimisation, limits the head.
- Worth checking: was the published frozen run's validation curve still falling at epoch 20? If so, that argues against.

## 7. Synthetic-BN frozen head

**Quantity:** the share of the layer-4 gain recovered on each real domain, using three-seed means.

  share = (frozen ImageNet-BN E_R − synthetic-BN E_R) / (frozen ImageNet-BN E_R − layer-4 no-aug E_R)

- At the published seed-42 frozen value, the denominators are lightbox 0.586 and sunlamp 0.514. They'll be recomputed with the three-seed frozen mean.

**Expectation:** no directional prediction (unsure).

**Decision rule:**
- **"Most" = ≥ 50% on each real domain**† (the checklist 0.2 suggestion). If so, "adapting the representation" becomes "adapting normalisation statistics" (4.4).
- **0–50%:** "normalisation statistics account for part of it."
- **≤ 0:** "not explained by normalisation statistics alone."

**Caveat:** this isn't a clean twin of the layer-4 arm.
- This head re-estimates every BN layer and changes no weights.
- The layer-4 arm adapted layer 4's weights and BN statistics, and kept ImageNet statistics in layers 1–3.

## 8. AdaBN (test-time adaptation)

**Scope (decided Wed 28 Oct):** the primary analysis only.
- It re-estimates the BN layers that were fitted on synthetic data, i.e. layer 4's, in the six layer-4 checkpoints.
- The frozen arms are excluded: all their BN statistics are ImageNet's, so there's nothing synthetic to replace.
- No "all layers" secondary analysis.
- `adabn.py` needs a scope flag first; Claude adds it.

**Quantity:** per checkpoint and real domain, adapted − unadapted on the same held-out half (split seed 0):
- ΔE_R and ΔE_T;
- Δ coverage (rotation and translation), using the published q̂.

**Reading of "better than on Control"†:** better than the same checkpoint without AdaBN. If you meant the retrained control run, say so.

**Expectation:** better all round. E_R and E_T are lower and coverage is higher, on both domains, in both arms.
- Size: roughly 5-10% better

**Confidence:** 50-60 %

**Decision rule:**
- **"Improved"** on a domain and metric: all 6 checkpoints share the sign (3/3 within each arm).
- **Mixed:** report the counts, and write no "AdaBN helps" sentence for that domain.
- **Coverage "stays low"** (4.4): adapted rotation coverage < 80%† (published: sunlamp 9–15%, lightbox 21–38%). If so, the coverage finding says "AdaBN lowers error, but coverage stays low."
- Report AdaBN separately, as test-time adaptation.

**Basis†:**
- The AdaBN literature.
- §3.2's last mechanism sentence: layer 4's BN statistics track the augmented synthetic distribution, which fits neither real domain.

## 9. Fig. 1 statistics

**Quantity:** per pool, the edge density, clipped-pixel fraction, and gradient TV from each real domain.
- Pools: lightbox; sunlamp; synthetic under each of {full stack, arm A, arm B, no aug}.

**Expectation:** Close to what can be implied from the paper. What the mechanism paragraph implies (agree or disagree with each):
1. Clipped fraction: sunlamp > full stack. The paper already asserts this.
2. Edge density: sunlamp > lightbox, and sunlamp > full stack.
3. Gradient TV: the full stack sits farther from sunlamp than from lightbox, as it does for intensity (0.80 vs 0.63).
4. Arm A has a higher edge density than the full stack, because blur is removed.
   - Noise adds small gradients everywhere, so the full stack vs no aug could go either way.
   - Arm A vs the full stack isolates the blur-and-noise pair.

**Confidence:** 90 %

**Decision rule:** if 1 or 2 fails, two passages become hypotheses rather than observations:
- the mechanism paragraph's description of sunlamp (hard-edged, clipped);
- "The mechanism is visible in the images".

**Pipeline check, not a prediction:** the full stack's intensity TV should reproduce the published 0.63 / 0.80.
