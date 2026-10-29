#!/usr/bin/env python
"""
Train frozen-arm pose heads on cached features. CPU, minutes per run.

    # frozen arm at the shared config, extra seeds
    python scripts/train_head.py --seeds 42 43 44

    # the decomposition: 4k vs 28k images x logits vs pooled features
    python scripts/train_head.py --features logits --train-subset 4000 --subset-seed 0

    # the head sweep, selected on synthetic validation only
    python scripts/train_head.py --lr 3e-4 --epochs 60

    # the synthetic-BN head
    python scripts/train_head.py --bn synthetic --seeds 42 43 44

Each run writes <artifacts>/heads/predictions/<name>.pkl, in the format
scripts/run_conformal.py reads (pass --predictions-dir <artifacts>/heads/predictions
--arms <name> ...), and a row per domain to <artifacts>/heads/heads.csv.

The selection rule is the published one: among even epochs, the best synthetic
validation rotation error. Real-domain numbers are computed for the record and
never consulted, here or in choosing a sweep winner.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from speedplus_pose.cli import add_path_arguments, paths_from_args
from speedplus_pose.evaluate import save_predictions, summarise
from speedplus_pose.features import CACHE_SPLITS, BN_REGIMES, load_cache, load_features, to_logits
from speedplus_pose.heads import HeadConfig, predict_head, subsample, train_head
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.normalisation import TranslationNormaliser
from speedplus_pose.experiment import load_split_labels


def run_name(args, train_images: int, seed: int) -> str:
    size = f"{train_images // 1000}k" if train_images >= 1000 else f"{train_images}img"
    return (
        f"head_{args.bn}bn_{args.features}_{size}"
        f"_lr{args.lr:g}_ep{args.epochs}_seed{seed}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--bn", choices=BN_REGIMES, default="imagenet")
    parser.add_argument("--features", choices=("pooled", "logits"), default="pooled")
    parser.add_argument("--train-subset", type=int, default=None, help="e.g. 4000 of the 28k")
    parser.add_argument("--subset-seed", type=int, default=0, help="Seed for drawing --train-subset; record it")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    cache = load_cache(paths.features(args.bn), CACHE_SPLITS)

    if args.features == "logits":
        fc = load_features(paths.features(args.bn).parent / "imagenet_fc.npz")
        for record in cache.values():
            record["features"] = to_logits(record["features"], fc["weight"], fc["bias"])

    train = cache["train"]
    if args.train_subset:
        train = subsample(train, args.train_subset, args.subset_seed)

    # Normalisation constants from the rows the head actually trains on, the
    # same rule train.py applies, computed from the label file rather than the
    # float32 cache so the 28k case matches the published constants exactly.
    train_frame = load_split_labels("train", paths).set_index("filename").loc[train["filename"]]
    means, stds = translation_normalisation(train_frame.reset_index())
    normaliser = TranslationNormaliser(means, stds)

    rows = []
    for seed in args.seeds:
        name = run_name(args, len(train["filename"]), seed)
        config = HeadConfig(learning_rate=args.lr, epochs=args.epochs, seed=seed)
        result = train_head(train, cache["synthetic"], normaliser, config)

        predictions = {split: predict_head(result.head, cache[split], normaliser) for split in CACHE_SPLITS if split != "train"}
        save_predictions(paths.heads / "predictions" / f"{name}.pkl", predictions)
        summary = {
            "name": name,
            "bn": args.bn,
            "features": args.features,
            "train_images": int(len(train["filename"])),
            "subset_seed": args.subset_seed if args.train_subset else None,
            "lr": args.lr,
            "epochs": args.epochs,
            "seed": seed,
            "selected_epoch": result.selected_epoch + 1,
            "validation_rotation_errors": result.validation_rotation_errors,
        }
        (paths.heads / "runs").mkdir(parents=True, exist_ok=True)
        (paths.heads / "runs" / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n{name}: selected epoch {result.selected_epoch + 1} on synthetic val")
        for domain in ("synthetic", "lightbox", "sunlamp"):
            row = summarise(predictions[domain])
            print(f"  {domain:10} E_T {row['translation_relative']:.3f}  E_R {row['rotation']:.3f}")
            rows.append({k: summary[k] for k in ("name", "bn", "features", "train_images", "subset_seed", "lr", "epochs", "seed", "selected_epoch")}
                        | {"domain": domain} | row)

    table = paths.heads / "heads.csv"
    frame = pd.DataFrame(rows)
    if table.exists():
        previous = pd.read_csv(table)
        frame = pd.concat([previous[~previous.name.isin(frame.name)], frame], ignore_index=True)
    frame.to_csv(table, index=False)
    print(f"\nwrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
