#!/usr/bin/env python
"""
Prove the ImageNet feature cache is the published frozen arm's backbone.

    python scripts/check_cache.py --checkpoint <dir>/1-frozen_no_aug_checkpoint_20260716_0245.pt

Loads the published frozen-no-aug seed-42 head, runs it on the cached features,
and compares every number against results/results_2x2.csv. A head *retrained*
on the cache cannot be expected to land on the published numbers exactly, since
the published runs seeded torch only and their RNG stream is gone. The
published head on the cache can, and must: if it does not, the cache is not
what the frozen arm saw, and nothing trained on it means anything.

Exits non-zero on any disagreement beyond --tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from speedplus_pose.cli import add_path_arguments, paths_from_args
from speedplus_pose.evaluate import summarise
from speedplus_pose.experiment import REPORT_DOMAINS, load_split_labels
from speedplus_pose.features import load_cache
from speedplus_pose.heads import head_from_checkpoint, predict_head
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.normalisation import TranslationNormaliser, constants_for_checkpoint

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results_2x2.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="Published frozen_no_aug seed-42 checkpoint")
    parser.add_argument("--arm", default="frozen_no_aug")
    parser.add_argument("--seed", type=int, default=42)
    # The published numbers are rounded to three decimals.
    parser.add_argument("--tolerance", type=float, default=0.0015)
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    head = head_from_checkpoint(checkpoint)
    means, stds = constants_for_checkpoint(
        checkpoint, *translation_normalisation(load_split_labels("train", paths)), args.arm
    )
    normaliser = TranslationNormaliser(means, stds)

    cache = load_cache(paths.features("imagenet"), REPORT_DOMAINS)
    published = pd.read_csv(RESULTS)
    published = published[(published.arm == args.arm) & (published.seed == args.seed)]

    failures = 0
    print(f"{'domain':10} {'metric':22} {'published':>10} {'from cache':>11} {'diff':>8}")
    for domain in REPORT_DOMAINS:
        measured = summarise(predict_head(head, cache[domain], normaliser))
        expected = published[published.domain == domain].iloc[0]
        for metric in ("translation_relative", "rotation"):
            diff = measured[metric] - expected[metric]
            ok = abs(diff) <= args.tolerance
            failures += not ok
            print(
                f"{domain:10} {metric:22} {expected[metric]:10.3f} {measured[metric]:11.4f} "
                f"{diff:+8.4f} {'' if ok else '  <-- MISMATCH'}"
            )

    if failures:
        print(f"\n{failures} mismatch(es). The cache is not the published frozen backbone: stop and debug.")
        return 1
    print("\nCache reproduces the published frozen arm. Heads trained on it are comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
