#!/usr/bin/env python
"""
Split-conformal coverage across the domain gap.

    python scripts/run_conformal.py

Reads the prediction records written by scripts/evaluate.py, one per arm,
covering the calibration, in_domain_test, lightbox, sunlamp and synthetic
splits. For each arm it then

  1. calibrates q_hat once on the fresh synthetic calibration split,
  2. checks that in-domain coverage lands on nominal (the gate),
  3. holds q_hat fixed and measures coverage on the real domains.

Step 2 is load-bearing. Fresh synthetic data is exchangeable with calibration by
construction, so the guarantee has to hold there; if it does not, the pipeline is
broken and no real-domain number below it means anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from speedplus_pose.cli import add_path_arguments, paths_from_args
from speedplus_pose.conformal import so3_ball_fraction
from speedplus_pose.coverage import (
    ALPHA,
    METRIC_UNITS,
    METRICS,
    calibrate,
    gate,
    measure,
    scores_from_record,
)
from speedplus_pose.evaluate import load_predictions
from speedplus_pose.experiment import COVERAGE_DOMAINS
from speedplus_pose.metrics import CHANCE_LEVEL

REQUIRED_SPLITS = ("calibration", *COVERAGE_DOMAINS)

# Reference points for reading an attitude cone, from F(a) = (a - sin a)/pi.
HALF_OF_SO3 = 2.310  # exactly half of all possible orientations


def load_arm_scores(arm: str, predictions_path: Path) -> dict[str, dict[str, np.ndarray]]:
    path = predictions_path / f"{arm}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions for {arm} at {path}. Run:\n"
            f"  python scripts/evaluate.py --config <arm> --seed <n> "
            f"--checkpoint <ckpt> --domains {' '.join(REQUIRED_SPLITS)}"
        )
    records = load_predictions(path)
    missing = [split for split in REQUIRED_SPLITS if split not in records]
    if missing:
        raise ValueError(f"{path} is missing split(s) {missing}")
    return {split: scores_from_record(records[split]) for split in REQUIRED_SPLITS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument(
        "--checkpoints",
        default=Path(__file__).resolve().parents[1] / "results" / "checkpoints.json",
        help="Manifest naming the arms to analyse",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="1 - nominal coverage")
    parser.add_argument("--output", default=None, help="Write the coverage table as CSV")
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    manifest = json.loads(Path(args.checkpoints).read_text(encoding="utf-8"))
    arms = list(manifest["arms"])

    scores = {arm: load_arm_scores(arm, paths.predictions) for arm in arms}
    n_calibration = len(scores[arms[0]]["calibration"]["rotation"])

    results = {}
    for arm in arms:
        q_hat = calibrate(scores[arm]["calibration"], args.alpha)
        results[arm] = {
            "q_hat": q_hat,
            "coverage": {
                domain: measure(q_hat, scores[arm][domain], domain, args.alpha)
                for domain in COVERAGE_DOMAINS
            },
        }

    # --- 1. the gate -------------------------------------------------------
    print(f"In-domain control: does the {1 - args.alpha:.0%} promise hold on fresh synthetic?")
    print(f"n_calibration = {n_calibration}\n")

    in_domain = [
        results[arm]["coverage"]["in_domain_test"][metric] for arm in arms for metric in METRICS
    ]
    passed, z_scores = gate(in_domain, n_calibration, args.alpha)
    for arm in arms:
        for metric in METRICS:
            entry = results[arm]["coverage"]["in_domain_test"][metric]
            print(
                f"  {arm:24} {metric:21} {entry.coverage:7.2%}  "
                f"[{entry.low:6.2%}, {entry.high:6.2%}]"
            )
    z = np.array(z_scores)
    print(
        f"\n  mean z {z.mean():+.2f}   sd {z.std(ddof=1):.2f}   "
        f"range [{z.min():+.2f}, {z.max():+.2f}]"
        "   (a correct pipeline gives ~0.0, ~1.0, within about +-2)"
    )
    if not passed:
        print(
            "\nGATE FAILED. In-domain coverage is off nominal, or every arm is "
            "offset the same way. Debug the pipeline before reading anything below."
        )
        return 1
    print("\nGate passed. Real-domain coverage below is now interpretable.\n")

    # --- 2. what the regions actually mean ---------------------------------
    print(f"Rotation region size at alpha={args.alpha}\n")
    for arm in arms:
        q_r = results[arm]["q_hat"]["rotation"]
        print(
            f"  {arm:24} q_hat = {q_r:.3f} rad ({np.degrees(q_r):5.1f} deg)"
            f"  -> {so3_ball_fraction(q_r):6.1%} of SO(3)"
        )
    print(
        f"\n  anchors: {CHANCE_LEVEL:.3f} rad = mean angle between two random orientations"
        f"\n           {HALF_OF_SO3:.3f} rad = exactly half of all possible orientations"
    )

    # --- 3. the collapse ---------------------------------------------------
    #
    # Rotation inflation has to be read next to the SO(3) share. Rotation error
    # is bounded by pi, so inflation can never exceed pi/q_hat however badly the
    # model degrades. A small rotation inflation therefore means either that the
    # model is genuinely robust or that its region already fills most of SO(3)
    # and has nowhere left to grow. The last columns tell those two apart.
    print(f"\n\nInflation needed for genuine {1 - args.alpha:.0%} coverage\n")
    print(f"  {'arm':24} {'domain':10} {'trans':>7} {'rot':>7} {'rot cap':>8}   honest rotation region")
    for arm in arms:
        q_r = results[arm]["q_hat"]["rotation"]
        cap = np.pi / q_r
        for domain in ("sunlamp", "lightbox"):
            entry = results[arm]["coverage"][domain]
            inflation_r = entry["rotation"].inflation
            honest = min(q_r * inflation_r, np.pi)
            print(
                f"  {arm:24} {domain:10} "
                f"{entry['translation_metres'].inflation:6.2f}x {inflation_r:6.2f}x "
                f"{cap:7.2f}x   {honest:.2f} rad = {so3_ball_fraction(honest):5.1%} of SO(3)"
            )

    # --- 4. the table ------------------------------------------------------
    print("\n\n| Arm | Domain | Nominal | Coverage (translation) | q-hat trans (m) | Inflation "
          "| Coverage (rotation) | q-hat rot (rad) | Inflation |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for arm in arms:
        q_hat = results[arm]["q_hat"]
        for domain in COVERAGE_DOMAINS:
            entry = results[arm]["coverage"][domain]
            t, r = entry["translation_metres"], entry["rotation"]
            print(
                f"| {arm} | {domain} | {1 - args.alpha:.0%} "
                f"| {t.coverage:.1%} ({t.low:.1%}-{t.high:.1%}) "
                f"| {q_hat['translation_metres']:.3f} | {t.inflation:.2f}x "
                f"| {r.coverage:.1%} ({r.low:.1%}-{r.high:.1%}) "
                f"| {q_hat['rotation']:.3f} | {r.inflation:.2f}x |"
            )
            for metric in METRICS:
                cell = entry[metric]
                rows.append(
                    {
                        "arm": arm,
                        "domain": domain,
                        "metric": metric,
                        "unit": METRIC_UNITS[metric],
                        "nominal": 1 - args.alpha,
                        "q_hat": q_hat[metric],
                        "coverage": cell.coverage,
                        "coverage_low": cell.low,
                        "coverage_high": cell.high,
                        "inflation": cell.inflation,
                        "n": cell.n,
                    }
                )

    if args.output:
        import pandas as pd

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
