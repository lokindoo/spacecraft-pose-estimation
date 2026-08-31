#!/usr/bin/env python
"""
Regenerate the paper's headline table and check it against the committed
results.

    python scripts/reproduce_table.py --from-predictions
    python scripts/reproduce_table.py --from-checkpoints <dir>

--from-predictions reads the prediction records scripts/evaluate.py wrote and
recomputes the table from them. No GPU, seconds.

--from-checkpoints is the real thing: it loads each released checkpoint, runs
inference over every domain, and rebuilds the table from raw model outputs. Use
this one to verify a set of downloaded weights actually produces the published
numbers.

Either way the result is compared row by row against results/results_2x2.csv and
the script exits non-zero on any disagreement beyond --tolerance. That comparison
is the point: a table you can regenerate is a table someone can trust.

NOTE ON RETRAINING: re-running scripts/train.py will NOT reproduce these numbers
exactly. The published runs seeded torch only, leaving numpy, random and the
DataLoader workers unseeded, so the runs are not bit-reproducible. Reproduce from
the checkpoints, not from a retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import load_config
from speedplus_pose.evaluate import load_predictions, predict, summarise
from speedplus_pose.experiment import REPORT_DOMAINS, build_eval_dataloader, load_split_labels
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.metrics import CHANCE_LEVEL, skill_retained
from speedplus_pose.model import build_model, resolve_device
from speedplus_pose.normalisation import TranslationNormaliser, constants_for_checkpoint
from speedplus_pose.transforms import build_eval_transform

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Table 1 groups the eight arms into four rows. Unfrozen rows aggregate three
# seeds as mean +- s.d.; frozen rows are single-seed and marked as such.
TABLE_ROWS = [
    ("frozen", "--", "frozen_no_aug"),
    ("frozen", "yes", "frozen_aug"),
    ("layer 4", "--", "unfrozen_no_aug"),
    ("layer 4", "yes", "unfrozen_aug"),
]


def measured_from_predictions(manifest: dict, predictions_path: Path) -> pd.DataFrame:
    rows = []
    for arm, entry in manifest["arms"].items():
        records = load_predictions(predictions_path / f"{arm}.pkl")
        for domain in REPORT_DOMAINS:
            summary = summarise(records[domain])
            rows.append(
                {
                    "arm": entry["config"],
                    "seed": entry["seed"],
                    "domain": domain,
                    "translation_relative": summary["translation_relative"],
                    "rotation": summary["rotation"],
                }
            )
    return pd.DataFrame(rows)


def measured_from_checkpoints(
    manifest: dict, checkpoint_dir: Path, paths, device, local_images
) -> pd.DataFrame:
    fallback_means, fallback_stds = translation_normalisation(load_split_labels("train", paths))
    rows = []

    for arm, entry in manifest["arms"].items():
        config = load_config(entry["config"], entry["seed"])
        path = checkpoint_dir / entry["checkpoint"]
        if not path.exists():
            raise FileNotFoundError(f"{arm}: no checkpoint at {path}")

        print(f"\n{arm}")
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        model = build_model(config.model, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        means, stds = constants_for_checkpoint(checkpoint, fallback_means, fallback_stds, arm)
        normaliser = TranslationNormaliser(means, stds, device)

        transform = build_eval_transform(config.image)
        for domain in REPORT_DOMAINS:
            dataloader = build_eval_dataloader(
                domain, paths, transform, config.train, local_images=local_images
            )
            summary = summarise(predict(model, dataloader, normaliser, device, f"{arm}/{domain}"))
            rows.append(
                {
                    "arm": entry["config"],
                    "seed": entry["seed"],
                    "domain": domain,
                    "translation_relative": summary["translation_relative"],
                    "rotation": summary["rotation"],
                }
            )
            print(f"  {domain:10} E_T {summary['translation_relative']:.3f}  E_R {summary['rotation']:.3f}")

        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def print_table(measured: pd.DataFrame) -> None:
    """Table 1 in the paper's layout."""
    print("\n" + "=" * 92)
    print("Table 1: mean +- s.d. over seeds {42, 43, 44}; frozen arms are single-seed (n=1)")
    print(f"E_T range-normalised, E_R in rad (chance level {CHANCE_LEVEL:.3f})")
    print("=" * 92)
    header = f"{'Backbone':10} {'Aug':5}"
    for domain in REPORT_DOMAINS:
        header += f" {domain[:9]:>19}"
    print(header)
    print(f"{'':10} {'':5}" + f" {'E_T':>9}{'E_R':>10}" * len(REPORT_DOMAINS))
    print("-" * 92)

    for backbone, aug, arm in TABLE_ROWS:
        cells = ""
        for domain in REPORT_DOMAINS:
            group = measured[(measured["arm"] == arm) & (measured["domain"] == domain)]
            if group.empty:
                cells += f" {'-':>9}{'-':>10}"
                continue
            if len(group) == 1:
                cells += f" {group['translation_relative'].iloc[0]:9.3f}{group['rotation'].iloc[0]:10.3f}"
            else:
                cells += (
                    f" {group['translation_relative'].mean():9.3f}"
                    f"{group['rotation'].mean():10.3f}"
                )
        print(f"{backbone:10} {aug:5}{cells}")

    print("-" * 92)
    for backbone, aug, arm in TABLE_ROWS:
        group = measured[measured["arm"] == arm]
        if group["seed"].nunique() < 2:
            continue
        parts = []
        for domain in REPORT_DOMAINS:
            values = group[group["domain"] == domain].sort_values("seed")["rotation"]
            parts.append(f"{domain} " + "/".join(f"{v:.3f}" for v in values))
        print(f"  {arm} per-seed E_R: " + ", ".join(parts))

    # The headline sentence of the study, computed rather than asserted.
    best = measured[measured["arm"] == "unfrozen_aug"]
    if not best.empty:
        synthetic = best[best["domain"] == "synthetic"]["rotation"].mean()
        print("\nSkill retained by the best arm (rotation, relative to its own synthetic score):")
        for domain in ("lightbox", "sunlamp"):
            domain_error = best[best["domain"] == domain]["rotation"].mean()
            print(f"  {domain:10} {skill_retained(synthetic, domain_error):.0%}")


def compare(measured: pd.DataFrame, tolerance: float) -> bool:
    """Row-by-row check against the committed results."""
    expected = pd.read_csv(RESULTS_DIR / "results_2x2.csv")
    keys = ["arm", "seed", "domain"]
    joined = expected.merge(measured, on=keys, suffixes=("_published", "_measured"), how="left")

    print("\n" + "=" * 92)
    print(f"Comparison against results/results_2x2.csv (tolerance {tolerance})")
    print("=" * 92)

    worst, failures = 0.0, 0
    for _, row in joined.iterrows():
        if row.isna().any():
            print(f"  MISSING  {row['arm']:18} seed {row['seed']} {row['domain']:10}")
            failures += 1
            continue
        d_t = abs(row["translation_relative_measured"] - row["translation_relative_published"])
        d_r = abs(row["rotation_measured"] - row["rotation_published"])
        worst = max(worst, d_t, d_r)
        if max(d_t, d_r) > tolerance:
            print(
                f"  MISMATCH {row['arm']:18} seed {row['seed']} {row['domain']:10} "
                f"E_T {row['translation_relative_published']:.3f} vs "
                f"{row['translation_relative_measured']:.3f}, "
                f"E_R {row['rotation_published']:.3f} vs {row['rotation_measured']:.3f}"
            )
            failures += 1

    if failures:
        print(f"\n{failures} row(s) disagree. Worst difference {worst:.4f}.")
        return False
    print(f"\nAll {len(joined)} rows reproduce. Worst difference {worst:.4f}.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-predictions",
        action="store_true",
        help="Recompute from stored prediction records (no GPU)",
    )
    source.add_argument(
        "--from-checkpoints",
        metavar="DIR",
        help="Re-run inference from the released checkpoints in DIR",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Published values are rounded to 3 decimals, so the default matches "
        "that rounding rather than pretending to more precision",
    )
    parser.add_argument("--checkpoints-manifest", default=RESULTS_DIR / "checkpoints.json")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.checkpoints_manifest).read_text(encoding="utf-8"))

    if args.from_predictions:
        paths = paths_from_args(args)
        measured = measured_from_predictions(manifest, paths.predictions)
    else:
        paths = paths_from_args(args)
        measured = measured_from_checkpoints(
            manifest,
            Path(args.from_checkpoints),
            paths,
            resolve_device(args.device),
            args.local_images,
        )

    print_table(measured)
    return 0 if compare(measured, args.tolerance) else 1


if __name__ == "__main__":
    sys.exit(main())
