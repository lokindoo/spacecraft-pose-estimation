"""
Split-conformal prediction, and the machinery for deciding whether a coverage
measurement means anything.

Calibrate q_hat once on held-out synthetic data, then hold it fixed while
measuring coverage everywhere else. Recomputing q_hat per domain would answer a
different question; the one being asked is what happens to a guarantee
calibrated in simulation when the images stop being simulated.

The scores are Euclidean metres for translation, where q_hat reads as a keep-out
sphere radius, and geodesic angle for rotation, where it is an attitude cone.
Both come from metrics.py, so the conformal analysis and the results table
cannot drift apart.
"""

import numpy as np
from scipy.stats import betabinom
from statsmodels.stats.proportion import proportion_confint

from .metrics import rotation_error, translation_error_metres

__all__ = [
    "clopper_pearson_interval",
    "compute_coverage",
    "compute_q_hat",
    "coverage_null",
    "coverage_null_interval",
    "coverage_null_z",
    "inflation_factor",
    "rotation_error",
    "so3_ball_fraction",
    "translation_error_metres",
]


def compute_q_hat(scores: list, alpha: float):
    """The conformal quantile of the calibration scores, with the +1 correction."""
    scores = np.asarray(scores)
    assert np.isfinite(scores).all()

    n = len(scores)
    if alpha < 1 / (n + 1):
        return np.inf

    return np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="inverted_cdf")


def compute_coverage(scores: list, q_hat: float):
    scores = np.asarray(scores)
    assert np.isfinite(scores).all()

    correct = np.sum(scores <= q_hat)
    return correct, len(scores)


def clopper_pearson_interval(count: int, nobs: int):
    """Exact binomial confidence interval on an observed proportion."""
    return proportion_confint(count, nobs, method="beta")


def so3_ball_fraction(theta):
    theta = np.clip(theta, 0.0, np.pi)
    return (theta - np.sin(theta)) / np.pi


def inflation_factor(real_scores: list, q_hat_synth: float, alpha: float):
    return compute_q_hat(real_scores, alpha) / q_hat_synth


def coverage_null(n_calibration: int, n_test: int, alpha: float):
    """
    Distribution of the observed hit count when the pipeline is correct.

    Two independent things are random here, and only one of them is a binomial.
    First, q_hat is the k-th smallest of n_calibration scores, so the coverage
    it actually delivers is itself random: conditional on the calibration draw,
    that coverage is Beta(k, n_calibration + 1 - k). Second, given that
    coverage, the number of test points landing inside is Binomial(n_test,
    coverage).

    Compounding the two gives a beta-binomial. Treating q_hat as a fixed, known
    threshold, by using a plain binomial or a Clopper-Pearson interval on the
    test proportion alone, discards the first source and produces an interval
    far too narrow to use as a pass/fail gate.
    """
    k = int(np.ceil((n_calibration + 1) * (1 - alpha)))
    assert k <= n_calibration, (
        f"alpha={alpha} is too small for n_calibration={n_calibration}: "
        f"q_hat is infinite and coverage is trivially 100%"
    )
    return betabinom(n_test, k, n_calibration + 1 - k)


def coverage_null_interval(n_calibration: int, n_test: int, alpha: float, level: float = 0.999):
    """Central acceptance region for the hit count, as (low, high) counts."""
    distribution = coverage_null(n_calibration, n_test, alpha)
    tail = (1 - level) / 2
    return int(distribution.ppf(tail)), int(distribution.isf(tail))


def coverage_null_z(count: int, n_calibration: int, n_test: int, alpha: float):
    """How many standard deviations the observed hit count sits from nominal."""
    distribution = coverage_null(n_calibration, n_test, alpha)
    return float((count - distribution.mean()) / distribution.std())
