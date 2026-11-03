#!/usr/bin/env python
"""
The shape of the rotation errors: histograms, mass near pi, and error axes.

    python scripts/rotation_errors.py --records LABEL=PATH [LABEL=PATH ...]

Answers reviewer M2Lg's question (are real-domain failures bimodal, and is that
symmetry?) from the prediction records, without any new inference. For every
record and domain it writes

  - the histogram of rotation errors on [0, pi], next to the density uniform
    guessing would give, (1 - cos t)/pi;
  - the share of errors within NEAR_PI_WIDTH of pi, against its uniform
    reference (about 0.19), since chance alone piles errors towards pi;
  - the concentration of the error axes in the body frame and in the camera
    frame, for all non-trivial errors and for the near-pi ones alone. A
    symmetry flip concentrates in the body frame, an in-plane confusion in the
    camera frame (see speedplus_pose.error_analysis).

A bare path labels its record by the file stem. LABEL=PATH is for records whose
stems collide, such as the published full stack seed 42 and its Mac control.

Writes <artifacts>/results/rotation_error_shape.csv,
<artifacts>/results/rotation_error_histograms.csv and, given --figure-arms,
<artifacts>/figures/rotation_error_histograms.png. Prints only n and mean E_R
per row, which reproduce published tables, so running it does not reveal the
new summaries.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from speedplus_pose.cli import add_path_arguments, parse_records, paths_from_args
from speedplus_pose.error_analysis import (
    AXIS_CONE_DEGREES,
    MIN_AXIS_ANGLE,
    NEAR_PI_WIDTH,
    axial_concentration,
    error_rotations,
    near_pi_share,
    uniform_cone_share,
    uniform_tail_share,
)
from speedplus_pose.evaluate import load_predictions
from speedplus_pose.figures import rotation_error_histograms

DOMAINS = ("synthetic", "lightbox", "sunlamp")
BINS = np.linspace(0, np.pi, 37)  # 5-degree bins



def shape_rows(label: str, domain: str, record: dict) -> list[dict]:
    errors = np.asarray(record["rotation"], dtype=float)
    base = {
        "record": label,
        "domain": domain,
        "n": len(errors),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "share_above_half_pi": float(np.mean(errors > np.pi / 2)),
        "near_pi_share": near_pi_share(errors),
        "near_pi_uniform": uniform_tail_share(),
    }
    rows = []
    for frame in ("body", "camera"):
        angle, axes = error_rotations(record["q_pred"], record["q_true"], frame)
        if not np.allclose(angle, errors, atol=1e-5):
            raise AssertionError(f"{label}/{domain}: error angles disagree with the record")
        for subset, mask in (
            ("nontrivial", angle >= MIN_AXIS_ANGLE),
            ("near_pi", angle > np.pi - NEAR_PI_WIDTH),
        ):
            rows.append(
                base
                | {"frame": frame, "subset": subset, "cone_uniform": uniform_cone_share()}
                | axial_concentration(axes[mask], AXIS_CONE_DEGREES)
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--records", nargs="+", required=True, help="Prediction pickles, PATH or LABEL=PATH")
    parser.add_argument(
        "--figure-arms",
        nargs="*",
        default=None,
        help="Figure panels as NAME=LABEL,LABEL,... (records pooled per panel), "
        "e.g. 'no aug=unfrozen_no_aug_seed42,unfrozen_no_aug_seed43'",
    )
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    records = {label: load_predictions(path) for label, path in parse_records(args.records).items()}

    shape, histograms = [], []
    print(f"{'record':36} {'domain':10} {'n':>6} {'mean E_R':>9}")
    for label, splits in records.items():
        for domain in DOMAINS:
            if domain not in splits:
                continue
            errors = np.asarray(splits[domain]["rotation"], dtype=float)
            print(f"{label:36} {domain:10} {len(errors):6d} {errors.mean():9.3f}")
            shape += shape_rows(label, domain, splits[domain])
            counts, _ = np.histogram(errors, bins=BINS)
            histograms += [
                {"record": label, "domain": domain, "bin_low": BINS[i], "bin_high": BINS[i + 1], "count": int(c)}
                for i, c in enumerate(counts)
            ]

    results = paths.results
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shape).to_csv(results / "rotation_error_shape.csv", index=False)
    pd.DataFrame(histograms).to_csv(results / "rotation_error_histograms.csv", index=False)
    print(f"\nwrote {results / 'rotation_error_shape.csv'}\nwrote {results / 'rotation_error_histograms.csv'}")

    if args.figure_arms:
        panels = {}
        for spec in args.figure_arms:
            name, _, labels = spec.partition("=")
            panels[name] = {
                domain: np.concatenate([records[label][domain]["rotation"] for label in labels.split(",")])
                for domain in DOMAINS
            }
        rotation_error_histograms(panels, BINS, paths.artifacts / "figures" / "rotation_error_histograms.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
