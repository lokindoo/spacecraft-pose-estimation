#!/usr/bin/env python
"""
Conformal coverage by range bin: where the marginal 90% actually lands.

    python scripts/coverage_by_range.py --records LABEL=PATH [LABEL=PATH ...]

Split conformal promises marginal coverage, averaged over every pose; reviewer
kVr6 asked for the conditional picture, and the paper's limitations already
claim that a fixed metre-radius sphere under-covers distant targets in-domain
while range-normalising overcorrects. This measures it.

For each record it calibrates q_hat on the calibration split exactly as
scripts/run_conformal.py does, holds it fixed, and reports coverage within
range bins on in_domain_test, lightbox and sunlamp, for all three score
families. Bin edges are quantiles of the calibration split's range, so no test
data shapes them.

Before writing anything it checks that the per-bin counts add up to the
marginal coverage, the number run_conformal.py reports. Prints only that check,
so running it does not reveal the stratified numbers.

Writes <artifacts>/results/coverage_by_range.csv.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from speedplus_pose.cli import add_path_arguments, parse_records, paths_from_args
from speedplus_pose.conformal import compute_coverage
from speedplus_pose.coverage import ALPHA, METRICS, calibrate, scores_from_record
from speedplus_pose.error_analysis import coverage_by_bin, range_bin_edges
from speedplus_pose.evaluate import load_predictions
from speedplus_pose.experiment import COVERAGE_DOMAINS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--records", nargs="+", required=True, help="Prediction pickles, PATH or LABEL=PATH")
    parser.add_argument("--bins", type=int, default=5, help="Range bins (calibration quantiles)")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    rows = []
    for label, path in parse_records(args.records).items():
        records = load_predictions(path)
        calibration = scores_from_record(records["calibration"])
        q_hat = calibrate(calibration, args.alpha)
        edges = range_bin_edges(calibration["range"], args.bins)

        for domain in COVERAGE_DOMAINS:
            scores = scores_from_record(records[domain])
            for metric in METRICS:
                bins = coverage_by_bin(scores[metric], scores["range"], q_hat[metric], edges)
                marginal, n = compute_coverage(scores[metric], q_hat[metric])
                if sum(b["count"] for b in bins) != marginal or sum(b["n"] for b in bins) != n:
                    raise AssertionError(f"{label}/{domain}/{metric}: bins do not add up to the marginal")
                rows += [
                    {"record": label, "domain": domain, "metric": metric, "q_hat": q_hat[metric]} | b
                    for b in bins
                ]
        print(f"{label:36} bins add up to the marginal on {', '.join(COVERAGE_DOMAINS)}")

    output = paths.results / "coverage_by_range.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
