#!/usr/bin/env python
"""
Evaluate a trained checkpoint on the synthetic validation split and both real
domains, and write the prediction record every table and figure reads from.

    python scripts/evaluate.py --config unfrozen_aug --seed 42 \
        --checkpoint <path>/checkpoint_20260723_1342.pt

With --checkpoint omitted, the latest checkpoint in the arm's run directory is
used. That is right for a quick look and wrong for a reported number: published
results use the checkpoint with the best synthetic validation rotation error,
which is not generally the last one. Pass it explicitly, and record it in
results/checkpoints.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from speedplus_pose.cli import (
    add_config_arguments,
    add_local_images_argument,
    add_path_arguments,
    paths_from_args,
)
from speedplus_pose.config import load_config
from speedplus_pose.evaluate import predict, save_predictions, summarise
from speedplus_pose.experiment import (
    REPORT_DOMAINS,
    build_eval_dataloader,
    describe,
    load_split_labels,
)
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.metrics import CHANCE_LEVEL, skill_retained
from speedplus_pose.model import build_model, resolve_device
from speedplus_pose.normalisation import TranslationNormaliser, constants_for_checkpoint
from speedplus_pose.train import latest_checkpoint
from speedplus_pose.transforms import build_eval_transform


def resolve_checkpoint(args, paths, config) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    directory = paths.checkpoints(config.run_name)
    found = latest_checkpoint(directory)
    if found is None:
        raise FileNotFoundError(f"no checkpoints in {directory}")
    print(f"WARNING: using the latest checkpoint ({found.name}), not the selected one.")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(REPORT_DOMAINS),
        help=f"Splits to evaluate (default: {' '.join(REPORT_DOMAINS)})",
    )
    parser.add_argument("--output", default=None, help="Prediction pickle path")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.seed)
    paths = paths_from_args(args)
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args, paths, config)

    print(f"Evaluating {describe(config)}")
    print(f"  checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = build_model(config.model, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Prefer the constants stored in the checkpoint, but verify them against the
    # training subset: evaluating with the wrong constants mangles every
    # position prediction in a way that still looks plausible.
    fallback_means, fallback_stds = translation_normalisation(
        load_split_labels("train", paths)
    )
    means, stds = constants_for_checkpoint(
        checkpoint, fallback_means, fallback_stds, config.arm
    )
    normaliser = TranslationNormaliser(means, stds, device)

    transform = build_eval_transform(config.image)
    predictions = {}
    for domain in args.domains:
        dataloader = build_eval_dataloader(
            domain, paths, transform, config.train, local_images=args.local_images
        )
        predictions[domain] = predict(model, dataloader, normaliser, device, domain)

    output = Path(args.output) if args.output else paths.predictions / f"{config.arm}.pkl"
    save_predictions(output, predictions)
    print(f"\nsaved {output}")

    print(f"\n{'domain':16} {'E_T':>8} {'E_R':>8} {'S':>8} {'skill kept':>11}   n")
    reference = summarise(predictions["synthetic"])["rotation"] if "synthetic" in predictions else None
    for domain, record in predictions.items():
        row = summarise(record)
        kept = " " * 10
        if reference is not None:
            retained = skill_retained(reference, row["rotation"])
            # nan means the synthetic reference is itself at or below chance, so
            # there is no skill to retain and the ratio is meaningless.
            kept = "       n/a" if np.isnan(retained) else f"{retained:10.0%}"
        print(
            f"{domain:16} {row['translation_relative']:8.3f} {row['rotation']:8.3f} "
            f"{row['speed_score']:8.3f} {kept}   {row['n']}"
        )
    print(f"\nchance level for E_R is {CHANCE_LEVEL:.3f} rad; read distances from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
