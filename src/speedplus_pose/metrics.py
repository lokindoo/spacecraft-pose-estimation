"""
Pose error metrics.

Two translation errors live here on purpose, and they are not interchangeable:

  translation_error_metres    ||r_pred - r_true||
  translation_error_relative  ||r_pred - r_true|| / ||r_true||

The relative one is the official SPEED+ normalised translation error and is
what Table 1 reports. The absolute one is what the conformal analysis
calibrates on, since q_hat there has to read as a keep-out sphere radius in
metres. The published tables carry both: q_hat = 0.410 m and q_hat = 0.0787 are
the same arm under the two metrics. They keep distinct names so that collapsing
them into one "translation_error" cannot quietly move half the conformal
numbers.
"""

from __future__ import annotations

import numpy as np

# A uniformly random orientation guess. The geodesic angle between two random
# rotations has density (1 - cos(theta))/pi on [0, pi], so its mean is
# pi/2 + 2/pi. Rotation errors are read as distance below this rather than as
# raw magnitudes: 1.67 and 2.07 rad look close, but as skill they are 0.54 and
# 0.14.
CHANCE_LEVEL = np.pi / 2 + 2 / np.pi  # ~2.207 rad


def _to_numpy(values) -> np.ndarray:
    """Accept torch tensors (possibly on GPU, possibly requiring grad) or arrays."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def _check(values: np.ndarray, width: int, name: str) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"expected {name} of shape (N, {width}), got {values.shape}")
    return values


def translation_error_metres(r_pred, r_true) -> np.ndarray:
    """Euclidean position error in metres. Used as the conformal score."""
    r_pred = _check(_to_numpy(r_pred), 3, "r_pred")
    r_true = _check(_to_numpy(r_true), 3, "r_true")
    if r_pred.shape != r_true.shape:
        raise ValueError(f"shape mismatch: {r_pred.shape} vs {r_true.shape}")
    return np.linalg.norm(r_pred - r_true, axis=1)


def translation_error_relative(r_pred, r_true) -> np.ndarray:
    """
    Range-normalised position error: the official SPEED+ E_T, and the
    translation column of Table 1. Dividing by range keeps a 10 cm miss at 5 m
    from scoring the same as a 10 cm miss at 40 m.
    """
    r_true_array = _check(_to_numpy(r_true), 3, "r_true")
    return translation_error_metres(r_pred, r_true_array) / np.linalg.norm(
        r_true_array, axis=1
    )


def rotation_error(q_pred, q_true) -> np.ndarray:
    """
    Geodesic angle between two orientations, in radians:
    2*arccos|<q_pred, q_true>|. The absolute value makes it sign-invariant, so
    the double cover cannot inflate the error.
    """
    q_pred = _check(_to_numpy(q_pred), 4, "q_pred")
    q_true = _check(_to_numpy(q_true), 4, "q_true")
    if q_pred.shape != q_true.shape:
        raise ValueError(f"shape mismatch: {q_pred.shape} vs {q_true.shape}")

    q_pred = q_pred / np.linalg.norm(q_pred, axis=1, keepdims=True)
    q_true = q_true / np.linalg.norm(q_true, axis=1, keepdims=True)
    dot = np.sum(q_pred * q_true, axis=1)
    return 2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def skill(rotation_errors) -> np.ndarray:
    """
    Rotation skill above chance: CHANCE_LEVEL - E_R.

    Positive means better than guessing; near zero means the model has learned
    essentially nothing about orientation, however respectable the raw error
    looks. The paper reports errors and skill side by side for that reason.
    """
    return CHANCE_LEVEL - np.asarray(rotation_errors, dtype=float)


def skill_retained(reference_error: float, domain_error: float) -> float:
    """
    Fraction of a model's synthetic rotation skill that survives on a real
    domain.
    """
    reference_skill = CHANCE_LEVEL - reference_error
    if reference_skill <= 0:
        return float("nan")
    return float((CHANCE_LEVEL - domain_error) / reference_skill)


def speed_score(translation_relative, rotation) -> np.ndarray:
    """
    The official SPEED+ pose score S = E_T + E_R, unthresholded.

    SPEC2021 additionally zeroes errors below a calibration threshold; we
    report the plain sum.
    """
    return np.asarray(translation_relative, dtype=float) + np.asarray(
        rotation, dtype=float
    )
