"""
Split-conformal coverage across the domain gap: calibrate q_hat once on fresh
synthetic images, then hold it fixed and ask what fraction of errors it covers
on real ones.

Two guards make the answer trustworthy, and both matter more than the headline
number. The calibration split must be disjoint from training and from the
validation split used to select checkpoints, or the "fresh synthetic" control is
not fresh (labels.assert_disjoint). And in-domain coverage must land on nominal
before any real-domain number is read; ``gate`` below is that check, sized
against the right null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conformal import (
    clopper_pearson_interval,
    compute_coverage,
    compute_q_hat,
    coverage_null_interval,
    coverage_null_z,
    inflation_factor,
)

# Nominal 90% regions throughout the study.
ALPHA = 0.1

# The three score families reported. "translation" is metres; see metrics.py for
# why both translation metrics are carried rather than unified.
METRICS = ("translation_metres", "translation_relative", "rotation")
METRIC_UNITS = {"translation_metres": "m", "translation_relative": "", "rotation": "rad"}


@dataclass
class CoverageEntry:
    metric: str
    domain: str
    count: int
    n: int
    low: float
    high: float
    inflation: float

    @property
    def coverage(self) -> float:
        return self.count / self.n


def scores_from_record(record: dict) -> dict[str, np.ndarray]:
    """Pull the conformal score families out of a prediction record."""
    return {metric: np.asarray(record[metric], dtype=float) for metric in METRICS} | {
        "range": np.asarray(record["range"], dtype=float)
    }


def calibrate(calibration_scores: dict[str, np.ndarray], alpha: float = ALPHA) -> dict[str, float]:
    """
    q_hat per metric, from the calibration split only. Computed once and never
    recomputed per domain, which is the whole experiment.
    """
    return {
        metric: float(compute_q_hat(calibration_scores[metric], alpha))
        for metric in METRICS
    }


def measure(
    q_hat: dict[str, float],
    domain_scores: dict[str, np.ndarray],
    domain: str,
    alpha: float = ALPHA,
) -> dict[str, CoverageEntry]:
    """Coverage of the fixed q_hat on one domain, per metric."""
    entries = {}
    for metric in METRICS:
        scores = domain_scores[metric]
        count, nobs = compute_coverage(scores, q_hat[metric])
        low, high = clopper_pearson_interval(count, nobs)
        entries[metric] = CoverageEntry(
            metric=metric,
            domain=domain,
            count=int(count),
            n=int(nobs),
            low=float(low),
            high=float(high),
            inflation=float(inflation_factor(scores, q_hat[metric], alpha)),
        )
    return entries


def gate(
    entries: list[CoverageEntry],
    n_calibration: int,
    alpha: float = ALPHA,
    level: float = 0.999,
    max_mean_z: float = 3.0,
) -> tuple[bool, list[float]]:
    """
    In-domain control: does the 90% promise hold on fresh synthetic data?

    Sizing this correctly matters, because two things are random rather than
    one. q_hat is the k-th smallest of n_calibration scores, so the coverage it
    delivers is itself Beta(k, n_cal + 1 - k); hits among the test points are
    then binomial given that coverage. Compounded, the honest spread is wider
    than either alone.

    A Clopper-Pearson interval on the test proportion models only the second
    source, treating q_hat as an exact known threshold. Simulated on a sound
    pipeline, such a gate rejects often enough that a dozen measurements will
    almost certainly produce a false alarm, so Clopper-Pearson stays for
    reporting each number's uncertainty and the pass/fail decision uses the
    compounded beta-binomial null.

    The mean z-score is the second half of the check. Per-measurement noise can
    hide a systematic fault, whereas a real pipeline bug drags every arm the
    same way, so a mean z far from zero is informative even when no single arm
    trips the gate. The arms share calibration and test images, so these
    z-scores are correlated by an unknown amount; in the worst case of perfect
    correlation the mean is itself standard normal, which makes 3.0
    conservative.

    Returns ``(passed, z_scores)``.
    """
    passed = True
    z_scores = []
    for entry in entries:
        low, high = coverage_null_interval(n_calibration, entry.n, alpha, level)
        z_scores.append(coverage_null_z(entry.count, n_calibration, entry.n, alpha))
        passed = passed and (low <= entry.count <= high)

    if z_scores and abs(float(np.mean(z_scores))) >= max_mean_z:
        passed = False
    return passed, z_scores


def sweep_reliability(
    calibration_scores: dict[str, np.ndarray],
    domain_scores: dict[str, dict[str, np.ndarray]],
    metric: str,
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Measured coverage as nominal coverage is swept: the reliability curve.

    A perfectly honest model tracks the diagonal. Curves must be non-decreasing
    in nominal coverage, since relaxing alpha can only admit more points, and
    because the maths guarantees that, the assertion below is a sharp check on
    the code.
    """
    curves = {domain: [] for domain in domain_scores}
    for alpha in alphas:
        q = compute_q_hat(calibration_scores[metric], alpha)
        for domain, scores in domain_scores.items():
            count, nobs = compute_coverage(scores[metric], q)
            curves[domain].append(count / nobs)

    result = {domain: np.array(values) for domain, values in curves.items()}
    for domain, values in result.items():
        order = np.argsort(1 - alphas)
        drop = float(np.min(np.diff(values[order])))
        if drop < -1e-12:
            raise AssertionError(
                f"{metric}/{domain}: coverage falls by {-drop:.4f} as nominal rises"
            )
    return result
