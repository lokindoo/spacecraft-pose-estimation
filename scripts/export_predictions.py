#!/usr/bin/env python
"""
Export prediction records as plain CSV, for release.

    python scripts/export_predictions.py --records LABEL=PATH [LABEL=PATH ...]

The pickles that scripts/evaluate.py writes are convenient here but a poor
release format: unpickling runs code, and they repeat the SPEED+ labels in
every record. This writes

  - labels.csv.gz: one row per (domain, filename) with the true pose and range,
    taken once and checked identical across every record;
  - <label>.csv.gz per record: the predicted pose and the three per-image
    errors, in the same row order as labels.csv.gz.

Then it reads both back, recomputes every error from the CSV poses with
speedplus_pose.metrics, and fails if any differs from the record by more than
the export's 7 significant digits allow. Means per domain are printed so they
can be checked against results/.

Quaternions are scalar-first, (w, x, y, z), and their matrix maps body vectors
into the camera frame, as in the SPEED+ devkit. Positions are in metres.
SPEED+ is CC BY 4.0 (Park et al., Zenodo record 5588480); the labels here are
redistributed under it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from speedplus_pose.cli import parse_records
from speedplus_pose.evaluate import load_predictions
from speedplus_pose.metrics import rotation_error, translation_error_metres, translation_error_relative

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOMAINS = ("synthetic", "calibration", "in_domain_test", "lightbox", "sunlamp")
Q = ["qw", "qx", "qy", "qz"]
R = ["x", "y", "z"]
ERRORS = ["rotation", "translation_relative", "translation_metres"]
FLOAT_FORMAT = "%.7g"
# No timestamp in the gzip header, so a rerun writes identical bytes.
GZIP = {"method": "gzip", "mtime": 0}


def frame(record: dict, domain: str, which: str) -> pd.DataFrame:
    split = record[domain]
    columns = {"domain": domain, "filename": np.asarray(split["filename"]).astype(str)}
    columns |= {f"{c}_{which}": split[f"q_{which}"][:, i] for i, c in enumerate(Q)}
    columns |= {f"{c}_{which}": split[f"r_{which}"][:, i] for i, c in enumerate(R)}
    return pd.DataFrame(columns)


def export(records: dict[str, dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(records.values()))
    labels = pd.concat(
        [frame(first, d, "true").assign(range=first[d]["range"]) for d in DOMAINS], ignore_index=True
    )
    for label, record in records.items():
        mine = pd.concat([frame(record, d, "true") for d in DOMAINS], ignore_index=True)
        if not mine[["domain", "filename"]].equals(labels[["domain", "filename"]]):
            raise AssertionError(f"{label}: images or their order differ from the first record")
        if not np.allclose(mine.filter(like="_true").to_numpy(), labels.filter(like="_true").to_numpy()):
            raise AssertionError(f"{label}: true poses differ from the first record")
    labels.to_csv(output_dir / "labels.csv.gz", index=False, float_format=FLOAT_FORMAT, compression=GZIP)

    for label, record in records.items():
        predictions = pd.concat(
            [frame(record, d, "pred").assign(**{e: record[d][e] for e in ERRORS}) for d in DOMAINS],
            ignore_index=True,
        )
        predictions.to_csv(output_dir / f"{label}.csv.gz", index=False, float_format=FLOAT_FORMAT, compression=GZIP)


def verify(labels: list[str], output_dir: Path) -> None:
    truth = pd.read_csv(output_dir / "labels.csv.gz")
    q_true, r_true = truth[[f"{c}_true" for c in Q]].to_numpy(), truth[[f"{c}_true" for c in R]].to_numpy()
    print(f"{'record':48} " + " ".join(f"{d:>14}" for d in DOMAINS) + "   (mean E_R)")
    for label in labels:
        pred = pd.read_csv(output_dir / f"{label}.csv.gz")
        q_pred, r_pred = pred[[f"{c}_pred" for c in Q]].to_numpy(), pred[[f"{c}_pred" for c in R]].to_numpy()
        recomputed = {
            "rotation": rotation_error(q_pred, q_true),
            "translation_relative": translation_error_relative(r_pred, r_true),
            "translation_metres": translation_error_metres(r_pred, r_true),
        }
        for name, values in recomputed.items():
            worst = np.max(np.abs(values - pred[name].to_numpy()))
            if worst > 1e-4:
                raise AssertionError(f"{label}: {name} recomputed from the CSV is off by {worst:.2e}")
        means = pred.groupby("domain", sort=False)["rotation"].mean()
        print(f"{label:48} " + " ".join(f"{means[d]:14.3f}" for d in DOMAINS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+", required=True, help="Prediction pickles, PATH or LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "predictions")
    args = parser.parse_args(argv)

    records = {label: load_predictions(path) for label, path in parse_records(args.records).items()}
    export(records, args.output_dir)
    verify(list(records), args.output_dir)
    print(f"\nwrote {len(records)} records and labels.csv.gz to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
