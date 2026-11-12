# SPEED+ Pose Estimator — sim-to-real study checkpoints

Direct 6DoF pose regression (ResNet-50 + MLP head) trained only on synthetic
SPEED+ imagery: the checkpoints behind a study of what transfers across the
synthetic-to-real gap. Eight are the published arms; nine more are the
post-review runs. `results/checkpoints.json` in the repository lists all
seventeen with their sha256.

The study itself reproduces without the weights; see the repository.

These are not competitive weights. They are the experimental arms of a study,
and exist so that its table can be regenerated rather than taken on trust.

Weights, on Zenodo:
https://doi.org/10.5281/zenodo.24140385

Code, configs and the reproduction script:
https://github.com/lokindoo/spacecraft-pose-estimation

## Checkpoints

The selection rule, applied uniformly, is the checkpoint with the best
synthetic validation rotation error among those saved on even epochs. No
real-domain measurement informed the choice.

`SHA256SUMS`, next to the weights on Zenodo, lists each file's hash. The same
hashes appear in `results/checkpoints.json`, whose `checkpoint` field gives
each file's original name from training. The repository's scripts accept
either name.

### Results table

Ten files, behind the results table below.

| File | Backbone | Augmentation | Seed | Trained on |
|---|---|---|---|---|
| `frozen_no_aug_seed42.pt` | frozen | — | 42 | Colab |
| `frozen_aug_seed42.pt` | frozen | ✓ | 42 | Colab |
| `frozen_aug_seed43.pt` | frozen | ✓ | 43 | Apple M5 |
| `frozen_aug_seed44.pt` | frozen | ✓ | 44 | Apple M5 |
| `unfrozen_no_aug_seed42.pt` | layer 4 trainable | — | 42 | Colab |
| `unfrozen_no_aug_seed43.pt` | layer 4 trainable | — | 43 | Colab |
| `unfrozen_no_aug_seed44.pt` | layer 4 trainable | — | 44 | Colab |
| `unfrozen_aug_seed42.pt` | layer 4 trainable | ✓ | 42 | Colab |
| `unfrozen_aug_seed43.pt` | layer 4 trainable | ✓ | 43 | Colab |
| `unfrozen_aug_seed44.pt` | layer 4 trainable | ✓ | 44 | Colab |

### Augmentation ablation

Seven post-review files, all with layer 4 trainable, trained on an Apple M5.

| File | Augmentation | Seed |
|---|---|---|
| `unfrozen_aug_no_blur_noise_seed42.pt` | arm A: full stack minus blur and noise | 42 |
| `unfrozen_aug_no_blur_noise_seed43.pt` | arm A: full stack minus blur and noise | 43 |
| `unfrozen_aug_no_blur_noise_seed44.pt` | arm A: full stack minus blur and noise | 44 |
| `unfrozen_aug_jitter_gamma_seed42.pt` | arm B: jitter, gamma and erasing only | 42 |
| `unfrozen_aug_jitter_gamma_seed43.pt` | arm B: jitter, gamma and erasing only | 43 |
| `unfrozen_aug_jitter_gamma_seed44.pt` | arm B: jitter, gamma and erasing only | 44 |
| `control_unfrozen_aug_seed42.pt` | control: full stack, rerun on the ablation's hardware | 42 |

## Results

E_T range-normalised; E_R geodesic, in radians. The chance level for E_R is
2.207 rad, the error of a uniformly random orientation guess. Read the errors as
distance below that rather than as absolute magnitudes.

| Backbone | Aug. | Synth E_T | Synth E_R | Lightbox E_T | Lightbox E_R | Sunlamp E_T | Sunlamp E_R |
|---|---|---|---|---|---|---|---|
| frozen | — | 0.066 | 1.053 | 0.151 | 1.752 | 0.253 | 2.048 |
| frozen | ✓ | 0.073±0.002 | 1.185±0.019 | 0.121±0.005 | 1.708±0.031 | 0.304±0.024 | 1.876±0.010 |
| layer 4 | — | 0.036±0.002 | 0.317±0.010 | 0.169±0.027 | 1.166±0.018 | 0.311±0.036 | 1.534±0.075 |
| layer 4 | ✓ | 0.042±0.002 | 0.322±0.027 | 0.150±0.017 | **0.987±0.010** | 0.355±0.081 | 1.468±0.119 |

The frozen un-augmented arm is single-seed; the other arms are mean ± s.d. over
seeds 42/43/44. Frozen augmented seeds 43 and 44 were trained on an Apple M5,
the rest on Colab.

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

Note the asymmetry. The translation half must be un-normalised; the quaternion
half must not, since it is trained raw and unit-normalised. Passing the
quaternion through the de-normaliser mangles every orientation in a way that
still looks plausible.

One file stores no constants: `frozen_no_aug_seed42.pt` predates the code that
saved them, so the example raises `KeyError` on it. Every arm trained on the
same subset, and the other sixteen files store identical constants; take them
from any of those. `normalisation.constants_for_checkpoint` recomputes them
from the SPEED+ training labels instead.

## Intended use

Intended for research on sim-to-real transfer, domain-gap benchmarking on
SPEED+, uncertainty-quantification studies under distribution shift, and as a
reproducible baseline for direct pose regression.

Not intended for any operational use. Explicitly:

> **This is a research artifact, not flight software.** It has no verification,
> validation, qualification, or testing appropriate to a spaceflight system, and
> must not be used in a real rendezvous, proximity-operations, docking, or
> servicing system.

## Limitations

- Real-domain accuracy is poor in absolute terms. The best arm's rotation error
  on sunlamp (1.47 rad ≈ 84°) retains only ~36% of its synthetic skill.
- The model does not know when it is wrong. Conformal regions calibrated to 90%
  on synthetic data cover 9–15% of rotation errors on real domains, so
  confidence is not informative under the shift and predictions arrive as
  confidently wrong as they do correct.
- Augmentation is directional. The photometric stack is a poor model of both
  real domains and a markedly less poor one of the diffuse, lightbox-like case.
  It improves lightbox on every seed, and does not move sunlamp's harsh
  directional light beyond seed-to-seed variation.
- Trained on one target only, the Tango spacecraft. This is not a
  general-purpose spacecraft pose estimator.
- Fixed input geometry: 320×512 grayscale, one camera's intrinsics.
- Direct regression relies on coarse global shape, so it degrades less than
  keypoint+PnP pipelines across the gap. That is a property of the method class
  rather than a sign that these weights are robust.

## Training data

SPEED+ synthetic split only: 28,000 training and 2,000 validation images,
downscaled to 320×512 grayscale. No real image influenced training, tuning, or
checkpoint selection; the lightbox (6,740) and sunlamp (2,791) sets are used
strictly as held-out tests.

## Dataset licence

The weights are derived from SPEED+, which is licensed under Creative Commons Attribution 4.0 International.

The ResNet-50 backbone was initialised from torchvision's ImageNet-pretrained
weights (`IMAGENET1K_V2`), which may carry their own terms.

SPEED+ itself is not redistributed here, in any form.

## Citation

See `CITATION.cff` in the repository. Cite the SPEED+ dataset (Park et al., 2022)
alongside this work.
