# Pre-registration record

Predictions and decision rules for the experiments added after review, each
committed before the results it predicts were opened. Both files are copied
byte for byte from the author's private working repository, where they were
written; [history.patch](history.patch) reproduces every change ever made to
them there, with its commit hash and time.

| File | Covers | Used for |
|---|---|---|
| [P1_revision_predictions.md](P1_revision_predictions.md) | Augmentation ablation (arms A and B), the control rerun, frozen-head seeds, the {4k, 28k} × {logits, pooled} decomposition, the head sweep, the synthetic-BN head, AdaBN, and the image statistics | Sim2Science camera-ready, App. D |
| [P1_arxiv_predictions.md](P1_arxiv_predictions.md) | Frozen backbone + augmentation, seeds 43 and 44; rotation-error shape; coverage by range bin; coverage of the ablation arms and frozen heads | Sim2Science camera-ready (frozen + aug only, see Deviations) and the arXiv version |

## Timeline (UTC; the UK was on GMT throughout)

| Commit | Time | What it did |
|---|---|---|
| `213db98` | Wed 28 Oct 2026, 17:42 | Revision predictions locked. Arm A had finished training, unopened; the control was at epoch 6 of 20; nothing else had run. |
| `ee9ff04` | Fri 30 Oct, 11:31 | arXiv decision rules committed, with the prediction fields blank. |
| `36eeb43` | Fri 30 Oct, 11:54 | arXiv predictions entered verbatim, then locked. |
| `5ea79c3` | Fri 30 Oct, 11:58 | Formatting and wording polish. No prediction changed (compare against `36eeb43` in the patch). |
| `507292d` | Fri 30 Oct, 13:36 | The author's one-line clarification of the §1 "Overall" prediction. |

The arXiv-extension results were opened on Fri 30 Oct at 13:36, straight after
`507292d`. Each file lists, under "State when written" and "Reference points
(already seen)", exactly what had been run or seen when it was written.

## What this does and does not establish

- **Times are commit times in the author's own repository.** They are not a
  third-party registration, so they show the order of events as recorded, not
  proof that nothing was written later. The patch makes every edit visible,
  including the post-lock polish and clarification.
- **Some numbers were known before predicting,** and each file lists them,
  for example the frozen + aug seed-42 result that the arXiv §1 prediction is
  about.
- **Not pre-registered:** the items under "Not pre-registered" at the end of
  the arXiv file, and anything the paper labels exploratory.
- **Outcomes:** each decision rule's outcome is reported in the arXiv version's
  pre-registration appendix.

## Deviations

- **Frozen + aug seeds 43 and 44 in the camera-ready.** They were
  pre-registered for the arXiv version and finished after the camera-ready's
  results freeze (Thu 29 Oct, 12:00). Once their outcome was known, they were
  added to the camera-ready anyway, because they qualify one of its three
  findings: with the backbone frozen, augmentation helped sunlamp more than
  lightbox. Nothing else that finished after the freeze went in.
- **The coverage-by-range rule (arXiv §3)** says "monotone" without defining
  it, so it can be read strictly (every bin in order) or as a near-to-far
  trend, and the two readings lead to different wording. The arXiv version
  reports near- and far-bin coverage as numbers, which holds under either
  reading.
