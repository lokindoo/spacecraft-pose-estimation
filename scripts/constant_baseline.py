#!/usr/bin/env python
"""
The trivial baseline: one fixed pose, fitted on the synthetic training labels
and scored on every domain.

    python scripts/constant_baseline.py

Labels only. No images, no checkpoint, no GPU: a couple of seconds on a CPU, and
on Colab it needs nothing mounted beyond the label JSONs.

The study reads rotation errors as distances below the chance level, pi/2 + 2/pi
~ 2.207 rad. That is the error of a uniformly random guess, and SPEED+ attitudes
are not uniform on SO(3), so a predictor that always emits the mean training
pose ought to score better than 2.207. This script measures how much better, on
each domain, which tells you whether the chance level is a fair anchor or
whether an arm scoring near it is in fact worse than doing nothing.

Two constants are reported per quantity: the closed-form one (chordal-mean
quaternion, arithmetic-mean position) and the one that directly minimises the
reported metric. Quote the stronger of the two, since the claim being anchored
is that no constant pose does better. See baselines.py for why they differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from speedplus_pose.baselines import (
    QUATERNION_COLUMNS,
    TRANSLATION_COLUMNS,
    best_constant_quaternion,
    best_constant_translation,
    chordal_mean_quaternion,
    constant_predictor_errors,
    mean_translation,
)
from speedplus_pose.cli import add_path_arguments
from speedplus_pose.experiment import REPORT_DOMAINS, load_split_labels
from speedplus_pose.metrics import CHANCE_LEVEL
from speedplus_pose.paths import Paths

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_path_arguments(parser)
    parser.add_argument(
        "--fit-split",
        default="train",
        help="Split the constant is fitted on (default: train, the 28k synthetic "
        "training labels, which are the only labels a deployed system would have)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(REPORT_DOMAINS),
        help=f"Splits to score it on (default: {' '.join(REPORT_DOMAINS)})",
    )
    parser.add_argument(
        "--output",
        default=RESULTS_DIR / "constant_baseline.csv",
        help="Where to write the per-domain table",
    )
    args = parser.parse_args(argv)

    # Nothing is read from or written to the artifacts root here, so it is
    # defaulted rather than demanded.
    paths = Paths(args.data_root, args.artifacts_root or Path("."))

    train = load_split_labels(args.fit_split, paths)
    print(f"fitting the constant on {args.fit_split}: {len(train)} labels\n")

    q_train = train[QUATERNION_COLUMNS].to_numpy(dtype=float)
    r_train = train[TRANSLATION_COLUMNS].to_numpy(dtype=float)

    constants = {
        "closed_form": (mean_translation(r_train), chordal_mean_quaternion(q_train)),
    }
    constants["best"] = (
        best_constant_translation(r_train, start=constants["closed_form"][0]),
        best_constant_quaternion(q_train, start=constants["closed_form"][1]),
    )

    for name, (r_constant, q_constant) in constants.items():
        print(f"{name:11} t = [{', '.join(f'{v:.3f}' for v in r_constant)}] m")
        print(f"{'':11} q = [{', '.join(f'{v:.4f}' for v in q_constant)}]")

    rows = []
    for name, (r_constant, q_constant) in constants.items():
        for domain in args.domains:
            labels = load_split_labels(domain, paths)
            row = constant_predictor_errors(r_constant, q_constant, labels)
            rows.append({"constant": name, "domain": domain, **row})

    table = pd.DataFrame(rows)

    print(f"\n{'constant':11} {'domain':11} {'E_T':>8} {'E_R':>8} {'S':>8} "
          f"{'skill':>8}      n")
    for _, row in table.iterrows():
        print(
            f"{row['constant']:11} {row['domain']:11} {row['translation_relative']:8.3f} "
            f"{row['rotation']:8.3f} {row['speed_score']:8.3f} "
            f"{CHANCE_LEVEL - row['rotation']:8.3f}  {row['n']:>6}"
        )

    print(
        f"\nchance level (uniform guess) is {CHANCE_LEVEL:.3f} rad. Read the E_R "
        "column against it:\nthe gap is how much skill above chance comes for "
        "free from the pose prior alone. Any arm\nin Table 1 scoring above the "
        "corresponding 'best' row is worse than predicting a single\nfixed pose."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
