#!/usr/bin/env python
"""
AdaBN: test-time BatchNorm adaptation on unlabelled real images.

    python scripts/adabn.py --checkpoint-dir <dir with the 8 published checkpoints>

For every arm in results/checkpoints.json and each real domain:

  1. split the domain 50/50 by filename with a fixed seed, the same split for
     every arm, so arms stay comparable;
  2. evaluate the model as published on the held-out half;
  3. reset the BatchNorm layers in scope and re-estimate their statistics from
     the *images* of the adaptation half, labels unused, weights untouched;
  4. evaluate the adapted model on the same held-out half.

Scope (--scope). AdaBN swaps source-domain statistics for target-domain ones,
so the source statistics have to be the ones the model was trained on:

  trained   (default, the pre-registered analysis) only the BatchNorm layers
            in the trainable stages, i.e. layer 4 of the layer-4 arms, whose
            statistics were fitted on synthetic data. Frozen arms keep
            ImageNet statistics everywhere, have nothing in scope, and are
            skipped.
  all       every BatchNorm layer, including the ImageNet ones in layers 1-3.

Coverage for both uses q_hat exactly as published in
results/conformal_coverage.csv, calibrated on synthetic data with the
unadapted model. Recalibrating would answer a different question.

This is test-time adaptation, it uses target-domain images, and the paper
reports it separately from the train-time arms for that reason.

Writes <artifacts>/adabn/<scope>/<arm>.pkl and <artifacts>/results/adabn_<scope>.csv.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from speedplus_pose.batchnorm import batchnorm_layers, recompute_bn_statistics
from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import TrainConfig, load_config
from speedplus_pose.conformal import clopper_pearson_interval, compute_coverage
from speedplus_pose.coverage import METRICS
from speedplus_pose.data import build_dataloader
from speedplus_pose.evaluate import predict, save_predictions, summarise
from speedplus_pose.experiment import build_dataset, load_split_labels
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.model import build_model, resolve_device
from speedplus_pose.normalisation import TranslationNormaliser, constants_for_checkpoint
from speedplus_pose.transforms import build_eval_transform

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REAL_DOMAINS = ("lightbox", "sunlamp")


def split_halves(labels: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(adapt, held-out): a fixed permutation of the sorted filenames, cut in half."""
    order = np.random.default_rng(seed).permutation(np.sort(labels["filename"].to_numpy()))
    adapt = set(order[: len(order) // 2])
    mask = labels["filename"].isin(adapt)
    return labels[mask].reset_index(drop=True), labels[~mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--arms", nargs="+", default=None, help="Subset of checkpoints.json arms")
    parser.add_argument("--scope", choices=("trained", "all"), default="trained")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="First N images per domain (smoke test)")
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    device = resolve_device(args.device)
    manifest = json.loads((RESULTS_DIR / "checkpoints.json").read_text(encoding="utf-8"))
    q_hats = pd.read_csv(RESULTS_DIR / "conformal_coverage.csv").groupby(["arm", "metric"]).q_hat.first()
    arms = args.arms or list(manifest["arms"])

    loader_config = TrainConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    transform = build_eval_transform()
    fallback = translation_normalisation(load_split_labels("train", paths))

    halves = {}
    for domain in REAL_DOMAINS:
        labels = load_split_labels(domain, paths)
        if args.limit:
            labels = labels.head(args.limit)
        halves[domain] = split_halves(labels, args.split_seed)

    def dataloader(domain: str, labels: pd.DataFrame):
        dataset = build_dataset(domain, paths, transform, labels=labels, local_images=args.local_images)
        return build_dataloader(dataset, loader_config, shuffle=False)

    rows = []
    for arm in arms:
        entry = manifest["arms"][arm]
        config = load_config(entry["config"], entry["seed"])
        if args.scope == "trained" and not config.model.trainable_stages:
            print(f"  {arm}: frozen backbone, no synthetic-fitted BatchNorm in scope; skipped")
            continue
        checkpoint = torch.load(Path(args.checkpoint_dir) / entry["checkpoint"], weights_only=False, map_location=device)
        model = build_model(config.model, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        normaliser = TranslationNormaliser(*constants_for_checkpoint(checkpoint, *fallback, arm), device)

        records = {}
        for domain, (adapt, held_out) in halves.items():
            adapted = copy.deepcopy(model)
            layers = None
            if args.scope == "trained":
                stages = [getattr(adapted.resnet, name) for name in config.model.trainable_stages]
                layers = [layer for stage in stages for layer in batchnorm_layers(stage)]
            seen = recompute_bn_statistics(
                adapted.resnet, dataloader(domain, adapt), device, f"{arm} BN <- {domain}", layers=layers
            )
            records[domain] = {
                "unadapted": predict(model, dataloader(domain, held_out), normaliser, device, f"{arm} {domain}"),
                "adapted": predict(adapted, dataloader(domain, held_out), normaliser, device, f"{arm} {domain} AdaBN"),
                "adapt_images": seen,
            }
            for condition in ("unadapted", "adapted"):
                record = records[domain][condition]
                row = {"arm": arm, "domain": domain, "condition": condition, "scope": args.scope,
                       "split_seed": args.split_seed} | summarise(record)
                for metric in METRICS:
                    q_hat = float(q_hats[(arm, metric)])
                    count, n = compute_coverage(record[metric], q_hat)
                    low, high = clopper_pearson_interval(count, n)
                    row |= {f"coverage_{metric}": count / n, f"coverage_{metric}_low": low, f"coverage_{metric}_high": high}
                rows.append(row)
                print(
                    f"  {arm:24} {domain:9} {condition:10} E_T {row['translation_relative']:.3f}  "
                    f"E_R {row['rotation']:.3f}  rot cov {row['coverage_rotation']:.1%}"
                )
        save_predictions(paths.artifacts / "adabn" / args.scope / f"{arm}.pkl", records)

    output = paths.results / f"adabn_{args.scope}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
