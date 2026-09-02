#!/usr/bin/env python
"""
Re-evaluate the project's first baseline, which trained on 4,000 images with a
head reading the backbone's 1,000-d ImageNet logits, on the current 2,000-image
synthetic validation split.

    python scripts/legacy_baseline_eval.py \
        --checkpoint "<drive>/post_canonicalisation_model_checkpoints/checkpoint_20260705_1620.pt" \
        --legacy-train-labels "<drive>/data/synthetic_subset_20260624/training_subset_20260624.json" \
        --control-labels "<drive>/data/synthetic_subset_20260624/validation_subset_20260624.json" \
        --control-images "/content/legacy_synthetic"

The first finding's headline used to read 1.63 -> 1.05 rad, which bundled three
changes together: 4k -> 28k images, ImageNet logits -> pooled 2048-d features,
and a change of validation split, the earlier one being a 1,000-image split
drawn from training.json. A footnote conceded the third rather than removing it.
Scoring the old checkpoint on the current split removes it, which is what this
script did.

That run has happened. The control reproduced the published 1.634 exactly, and
on the current 2,000-image split the same checkpoint scores E_T 0.098, E_R
1.666, so the paper now reports the single-split comparison, 1.67 -> 1.05 =
0.61 rad, with no split footnote. Re-run this only to re-verify that number.

What remains in the headline is a two-way confound between data and features.
The 4k/2048-d intermediate point that would decompose it (1.31 rad) is still on
the old split, so the paper does not quote it; reinstating it would need a third
eval path for that architecture, which neither this script nor evaluate.py
provides. Inference only: minutes on one GPU, no training. If the checkpoint is
gone, say so and stop, and soften the number in the text rather than
reconstructing the model from memory.

The architecture here is not the current one, and is rebuilt deliberately. The
first baseline left ResNet-50's ``fc`` in place and fed its 1,000 ImageNet
logits to the head, and it z-scored all seven targets, quaternion components
included, rather than translation alone. Both are reproduced below, because
loading those weights under the current model's conventions would produce
plausible-looking numbers that are wrong.

Run the control. With --control-labels pointing at the old 1,000-image
validation split, and --control-images at its images, the script first
reproduces the published legacy numbers, 1.634 / 2.073 / 2.002 rad. If those do
not come back, the checkpoint, the normalisation constants, or the label file is
not the one that produced the paper's numbers, and nothing downstream is worth
reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import ImageConfig, TrainConfig
from speedplus_pose.data import SpeedPlusDataset, build_dataloader
from speedplus_pose.experiment import build_eval_dataloader
from speedplus_pose.labels import LABEL_COLUMNS, load_labels
from speedplus_pose.metrics import (
    CHANCE_LEVEL,
    rotation_error,
    translation_error_relative,
)
from speedplus_pose.model import resolve_device
from speedplus_pose.transforms import build_eval_transform

LEGACY_LOGITS = 1000
LEGACY_HEAD_HIDDEN = 1024
POSE_OUTPUTS = 7

# What the original notebook printed for this checkpoint, on the old
# 1,000-image split.
PUBLISHED_CONTROL = {"synthetic": 1.6335688885450363, "sunlamp": 2.073278134114125,
                     "lightbox": 2.002465371388489}


class LegacyPoseEstimator(nn.Module):
    """
    The first baseline, verbatim: ResNet-50 with its ImageNet classifier intact,
    then Linear(1000, 1024) -> ReLU -> Linear(1024, 7).

    Its "1,000-d features" were therefore the ImageNet logits, a thousand class
    scores, rather than pooled visual features, which is a sharper contrast with
    the 2048-d pooled version than the paper originally drew.
    """

    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fc = nn.Sequential(
            nn.Linear(LEGACY_LOGITS, LEGACY_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(LEGACY_HEAD_HIDDEN, POSE_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)
        return self.fc(features.view(features.size(0), -1))


def legacy_normalisation(train_labels_path: Path, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-component mean and s.d. over all seven targets of the 4k training split.

    ddof=0, matching the ``np.std`` default the original notebook used. These
    are the constants the model's outputs are expressed in, so taking them from
    the current 28k split instead would shift every prediction.
    """
    train = load_labels(train_labels_path)
    means = train[LABEL_COLUMNS].to_numpy(dtype=np.float64).mean(axis=0)
    stds = train[LABEL_COLUMNS].to_numpy(dtype=np.float64).std(axis=0, ddof=0)
    return (
        torch.tensor(means, dtype=torch.float32, device=device),
        torch.tensor(stds, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def evaluate(model, dataloader, means, stds, device, description: str) -> dict[str, float]:
    """Mean E_T and E_R over one domain, de-normalising all seven outputs."""
    model.eval()
    translation_batches, rotation_batches = [], []

    for inputs, labels in tqdm(dataloader, desc=description, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs) * stds + means

        r_pred = outputs[:, :3].cpu().numpy()
        q_pred = outputs[:, 3:].cpu().numpy()
        q_pred = q_pred / np.linalg.norm(q_pred, axis=1, keepdims=True)
        r_true = labels[:, :3].cpu().numpy()
        q_true = labels[:, 3:].cpu().numpy()

        translation_batches.append(translation_error_relative(r_pred, r_true))
        rotation_batches.append(rotation_error(q_pred, q_true))

    translation = float(np.mean(np.concatenate(translation_batches)))
    rotation = float(np.mean(np.concatenate(rotation_batches)))
    return {
        "translation_relative": translation,
        "rotation": rotation,
        "speed_score": translation + rotation,
        "n": len(dataloader.dataset),
    }


def report(name: str, row: dict[str, float]) -> None:
    print(
        f"{name:22} E_T {row['translation_relative']:.3f}  E_R {row['rotation']:.3f}  "
        f"S {row['speed_score']:.3f}  skill {CHANCE_LEVEL - row['rotation']:.3f}  "
        f"n={row['n']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--checkpoint", required=True, help="The legacy .pt checkpoint")
    parser.add_argument(
        "--legacy-train-labels",
        required=True,
        help="The 4,000-image training label JSON the checkpoint was normalised against",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["synthetic"],
        help="Current splits to score on (default: synthetic, the 2,000-image "
        "validation split the comparison needs). Add lightbox sunlamp to confirm "
        "the real-domain numbers are unchanged.",
    )
    parser.add_argument(
        "--control-labels",
        default=None,
        help="Old 1,000-image validation label JSON; reproduces the published "
        "1.634 rad before anything new is reported",
    )
    parser.add_argument("--control-images", default=None, help="Images for --control-labels")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="The legacy runs used 32")
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(
            f"no checkpoint at {checkpoint_path}.\n"
            "If it is genuinely gone, do not reconstruct it: soften the claim to "
            "'~0.5 rad on a comparable split' and keep the footnote. The stronger "
            "half of the finding, that the real domains barely moved on identical "
            "full test sets, does not depend on this number.",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(args.device)
    paths = paths_from_args(args)
    transform = build_eval_transform(ImageConfig())
    train_config = TrainConfig(batch_size=args.batch_size)

    means, stds = legacy_normalisation(Path(args.legacy_train_labels), device)
    print(f"normalisation from {Path(args.legacy_train_labels).name}")
    print(f"  means {np.round(means.cpu().numpy(), 4)}")
    print(f"  stds  {np.round(stds.cpu().numpy(), 4)}\n")

    model = LegacyPoseEstimator().to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"loaded {checkpoint_path.name} (epoch {checkpoint.get('epoch', '?')})\n")

    if args.control_labels:
        if not args.control_images:
            parser.error("--control-labels needs --control-images")
        control = SpeedPlusDataset(
            args.control_images, load_labels(args.control_labels), transform
        )
        row = evaluate(
            model,
            build_dataloader(control, train_config, shuffle=False),
            means, stds, device, "control (old split)",
        )
        report("control, old split", row)
        published = PUBLISHED_CONTROL["synthetic"]
        difference = abs(row["rotation"] - published)
        verdict = "matches" if difference < 0.01 else "DOES NOT MATCH"
        print(f"  published {published:.4f} rad -> {verdict} (difference {difference:.4f})\n")
        if difference >= 0.01:
            print(
                "Stop here. Something differs from the run that produced the paper's "
                "numbers: wrong checkpoint, wrong normalisation labels, or a "
                "re-prepared image subset. Nothing below is reportable.",
                file=sys.stderr,
            )
            return 1

    print("on the current splits:")
    for domain in args.domains:
        dataloader = build_eval_dataloader(
            domain, paths, transform, train_config, local_images=args.local_images
        )
        report(domain, evaluate(model, dataloader, means, stds, device, domain))

    print(
        "\nThe synthetic row is the de-confounded starting point for the first "
        "finding. The published\nrun gave E_R 1.666, which is why the paper quotes "
        "1.67 -> 1.05 rad (0.61) on one split, with no split footnote."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
