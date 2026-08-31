#!/usr/bin/env python
"""
The trivial baseline: one fixed pose, fitted on the synthetic training labels
and scored on every domain.

    python scripts/constant_baseline.py

Labels only. No images, no checkpoint, no GPU - a couple of seconds on a CPU,
and on Colab it needs nothing mounted beyond the label JSONs.

WHY THIS EXISTS. The paper reads rotation errors as distances below the chance
level, pi/2 + 2/pi ~ 2.207 rad. That is the error of a uniformly random guess,
and SPEED+ attitudes are not uniform on SO(3), so a predictor that always emits
the mean training pose will score better than 2.207. This script measures how
much better, on each domain, which settles two things at once:

  * if the constant predictor lands near 2.207, the chance level is the right
    anchor and the framing is bulletproofed for the cost of one table row;
  * if it lands well below, then any arm scoring above it - the frozen sunlamp
    arm at 2.048 rad is the candidate - is worse than doing nothing, and the
    "retains 14% of its skill" line in section 3.1 is generous rather than
    damning.

Both outcomes are usable. The number is what decides which.

Two constants are reported per quantity: the closed-form one (chordal-mean
quaternion, arithmetic-mean position) and the one that directly minimises the
reported metric. Quote the *best* row in the paper - the claim being anchored is
"no constant pose does better than this", so the strongest constant is the
honest comparison. See baselines.py for why they differ.
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
        help="Split the constant is fitted on (default: train - the 28k "
        "synthetic training labels, the only labels a deployed system would have)",
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

    # The artifacts root is irrelevant here - nothing is read from or written to
    # it - so it is defaulted rather than demanded.
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
        "column against it:\nthe gap is how much of the paper's 'skill above "
        "chance' is available for free, from\nthe pose prior alone. Any arm in "
        "Table 1 scoring ABOVE the corresponding 'best' row is\nworse than "
        "predicting a single fixed pose."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
