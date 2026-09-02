<!--
DRAFT HuggingFace model card.

BLOCKING TODO BEFORE UPLOADING ANY WEIGHTS: read the SPEED+ licence on Zenodo
record 5588480 and confirm (a) that publishing weights derived from it is
permitted and (b) what the licence is actually called, then fill in the
`dataset_license` field below and the Dataset section. Do not guess this - it
gates the upload, not just the card. Every other TODO here is cosmetic; that one
is not.

Move this YAML block to the top of the file when publishing (HuggingFace parses
front matter only at position 0).

---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
  - pose-estimation
  - 6d-pose-estimation
  - satellite
  - space
  - sim-to-real
  - domain-gap
  - conformal-prediction
datasets:
  - speedplus            # TODO: confirm the dataset's tag/id on the Hub
metrics:
  - rotation_error
  - translation_error
model-index:
  - name: speedplus-pose-resnet50
    results:
      - task:
          type: pose-estimation
          name: 6DoF spacecraft pose estimation
        dataset:
          type: speedplus
          name: "SPEED+ (sunlamp, real HIL)"
        metrics:
          - type: rotation_error
            value: 1.468
            name: "Geodesic rotation error (rad), layer4+aug, mean over 3 seeds"
          - type: translation_error
            value: 0.355
            name: "Range-normalised translation error, layer4+aug"
      - task:
          type: pose-estimation
          name: 6DoF spacecraft pose estimation
        dataset:
          type: speedplus
          name: "SPEED+ (lightbox, real HIL)"
        metrics:
          - type: rotation_error
            value: 0.987
            name: "Geodesic rotation error (rad), layer4+aug, mean over 3 seeds"
          - type: translation_error
            value: 0.150
            name: "Range-normalised translation error, layer4+aug"
---
-->

# SPEED+ Pose Estimator — sim-to-real study checkpoints

Direct 6DoF pose regression (ResNet-50 + MLP head) trained **only on synthetic
SPEED+ imagery**, released as the eight checkpoints behind a study of what
transfers across the synthetic-to-real gap.

These are not competitive weights. They are the experimental arms of a study, and
they are published so its table can be regenerated rather than trusted.

Code, configs and the reproduction script:
**TODO: repository URL**

## Checkpoints

Each file corresponds to exactly one row of the results table. The selection
rule, applied uniformly: **the checkpoint with the best synthetic validation
rotation error** among those saved on even epochs. No real-domain measurement
informed the choice.

| File | Backbone | Augmentation | Seed |
|---|---|---|---|
| `frozen_no_aug_seed42.pt` | frozen | — | 42 |
| `frozen_aug_seed42.pt` | frozen | ✓ | 42 |
| `unfrozen_no_aug_seed42.pt` | layer 4 trainable | — | 42 |
| `unfrozen_no_aug_seed43.pt` | layer 4 trainable | — | 43 |
| `unfrozen_no_aug_seed44.pt` | layer 4 trainable | — | 44 |
| `unfrozen_aug_seed42.pt` | layer 4 trainable | ✓ | 42 |
| `unfrozen_aug_seed43.pt` | layer 4 trainable | ✓ | 43 |
| `unfrozen_aug_seed44.pt` | layer 4 trainable | ✓ | 44 |

## Results

E_T range-normalised; E_R geodesic, in radians. **Chance level for E_R is
2.207 rad** — a uniformly random orientation guess. Read the errors as distance
below that, not as absolute magnitudes.

| Backbone | Aug. | Synth E_T | Synth E_R | Lightbox E_T | Lightbox E_R | Sunlamp E_T | Sunlamp E_R |
|---|---|---|---|---|---|---|---|
| frozen | — | 0.066 | 1.053 | 0.151 | 1.752 | 0.253 | 2.048 |
| frozen | ✓ | 0.071 | 1.180 | 0.118 | 1.718 | 0.306 | 1.864 |
| layer 4 | — | 0.036±0.002 | 0.317±0.010 | 0.169±0.027 | 1.166±0.018 | 0.311±0.036 | 1.534±0.075 |
| layer 4 | ✓ | 0.042±0.002 | 0.322±0.027 | 0.150±0.017 | **0.987±0.010** | 0.355±0.081 | 1.468±0.119 |

Frozen arms are single-seed; layer 4 arms are mean ± s.d. over seeds 42/43/44.

## Inference

```python
import torch
from PIL import Image
from speedplus_pose.model import SpacecraftPoseEstimator
from speedplus_pose.normalisation import TranslationNormaliser
from speedplus_pose.transforms import build_eval_transform

checkpoint = torch.load("unfrozen_aug_seed42.pt", weights_only=False, map_location="cpu")

model = SpacecraftPoseEstimator()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Translation targets were z-scored during training; the constants travel with
# the checkpoint. Skipping this step silently produces nonsense positions.
normaliser = TranslationNormaliser(checkpoint["train_means"], checkpoint["train_stds"])

image = Image.open("img000001.jpg")           # SPEED+ grayscale, any resolution
tensor = build_eval_transform()(image)         # -> 3x320x512, ImageNet-normalised

with torch.no_grad():
    output = model(tensor.unsqueeze(0))

translation = normaliser.remove(output[:, :3])            # metres, camera frame
quaternion = output[:, 3:]
quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)  # scalar-first, unit

print(translation, quaternion)
```

Note the asymmetry: the **translation** half must be un-normalised, the
**quaternion** half must not — it is trained raw and unit-normalised. Passing the
quaternion through the de-normaliser mangles every orientation in a way that
still looks plausible.

## Intended use

**Intended for:** research on sim-to-real transfer, domain-gap benchmarking on
SPEED+, uncertainty-quantification studies under distribution shift, and as a
reproducible baseline for direct pose regression.

**Not intended for:** any operational use. Explicitly:

> **This is a research artifact, not flight software.** It has no verification,
> validation, qualification, or testing appropriate to a spaceflight system, and
> must not be used in a real rendezvous, proximity-operations, docking, or
> servicing system.

## Limitations

- **Real-domain accuracy is poor in absolute terms.** The best arm's rotation
  error on sunlamp (1.47 rad ≈ 84°) retains only ~36% of its synthetic skill.
- **The model does not know when it is wrong.** Conformal regions calibrated to
  90% on synthetic data cover 9–15% of rotation errors on real domains.
  Confidence is *not* informative under the shift, and predictions arrive as
  confidently wrong as they do correct.
- **Augmentation is directional.** The photometric stack is a poor model of
  both real domains and a markedly less poor one of the diffuse, lightbox-like
  case. It improves lightbox on every seed and does not move sunlamp's harsh
  directional light beyond seed-to-seed variation.
- **Trained on one target only** — the Tango spacecraft. Not a general-purpose
  spacecraft pose estimator.
- **Fixed input geometry**: 320×512 grayscale, one camera's intrinsics.
- Direct regression relies on coarse global shape, so it degrades less than
  keypoint+PnP pipelines across the gap. That is a property of the method class,
  not a sign that these weights are robust.

## Training data

SPEED+ synthetic split only: 28,000 training and 2,000 validation images,
downscaled to 320×512 grayscale. **No real image influenced training, tuning, or
checkpoint selection.** The lightbox (6,740) and sunlamp (2,791) sets are used
strictly as held-out tests.

## Dataset licence

The weights are derived from SPEED+, which is licensed separately by its authors
and is **not** covered by this repository's Apache-2.0 licence.

> **TODO — BLOCKING:** state the SPEED+ licence here verbatim after reading the
> Zenodo record (5588480) terms, and confirm they permit publishing derived model
> weights. Resolve this before uploading anything.

SPEED+ itself is not redistributed here, in any form.

## Citation

See `CITATION.cff` in the repository. Cite the SPEED+ dataset (Park et al., 2022)
alongside this work.
