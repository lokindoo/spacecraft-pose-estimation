"""
Constant-predictor tests.

The constant baseline is only useful if it is hard to argue with, so these pin
the two properties a reviewer would poke at: that the double cover cannot move
the mean orientation, and that the refined constants optimise the metric the
paper reports rather than a convenient proxy.
"""

import numpy as np
import pandas as pd
import pytest

from speedplus_pose.baselines import (
    best_constant_quaternion,
    best_constant_translation,
    chordal_mean_quaternion,
    constant_predictor_errors,
    mean_translation,
)
from speedplus_pose.metrics import rotation_error, translation_error_relative


def _random_quaternions(n: int, spread: float, seed: int = 0) -> np.ndarray:
    """n orientations clustered around the identity, standing in for a pose prior."""
    rng = np.random.default_rng(seed)
    q = np.concatenate([np.ones((n, 1)), spread * rng.standard_normal((n, 3))], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _mean_rotation(q_constant, q_true) -> float:
    return float(np.mean(rotation_error(np.tile(q_constant, (len(q_true), 1)), q_true)))


def test_chordal_mean_of_identical_quaternions():
    q = np.tile([0.5, 0.5, 0.5, 0.5], (10, 1))
    assert np.allclose(np.abs(chordal_mean_quaternion(q)), [0.5, 0.5, 0.5, 0.5])


def test_chordal_mean_ignores_the_double_cover():
    """
    q and -q are the same rotation, so flipping half the labels must not move
    the mean. Hence the eigenvector of sum q q^T rather than an arithmetic mean
    of the quaternion components, which would collapse toward zero on a
    sign-flipped set.
    """
    q = _random_quaternions(200, spread=0.3)
    flipped = q.copy()
    flipped[::2] *= -1

    plain = chordal_mean_quaternion(q)
    mixed = chordal_mean_quaternion(flipped)
    assert rotation_error(plain[None, :], mixed[None, :])[0] < 1e-8


def test_best_constant_quaternion_beats_the_closed_form():
    """
    The closed form minimises a squared-chordal loss; Table 1 reports mean
    geodesic angle. The refinement must not be worse under the reported metric.
    """
    q = _random_quaternions(500, spread=0.6, seed=3)
    closed_form = chordal_mean_quaternion(q)
    refined = best_constant_quaternion(q, start=closed_form)
    assert _mean_rotation(refined, q) <= _mean_rotation(closed_form, q) + 1e-12


def test_best_constant_translation_beats_the_centroid():
    """
    E_T divides by range, so the optimal constant is pulled toward close
    targets. Over a spread of ranges it must beat the arithmetic mean.
    """
    rng = np.random.default_rng(1)
    r = np.column_stack(
        [rng.normal(0, 0.5, 400), rng.normal(0, 0.5, 400), rng.uniform(4, 40, 400)]
    )
    centroid = mean_translation(r)
    refined = best_constant_translation(r, start=centroid)

    def mean_relative(candidate):
        return float(np.mean(translation_error_relative(np.tile(candidate, (len(r), 1)), r)))

    assert mean_relative(refined) < mean_relative(centroid)


def test_constant_predictor_errors_are_zero_on_a_constant_label_set():
    labels = pd.DataFrame(
        {
            "filename": ["a.jpg", "b.jpg"],
            "r1": [0.0, 0.0], "r2": [0.0, 0.0], "r3": [10.0, 10.0],
            "q1": [1.0, 1.0], "q2": [0.0, 0.0], "q3": [0.0, 0.0], "q4": [0.0, 0.0],
        }
    )
    errors = constant_predictor_errors([0.0, 0.0, 10.0], [1.0, 0.0, 0.0, 0.0], labels)
    assert errors["translation_relative"] == pytest.approx(0.0)
    assert errors["rotation"] == pytest.approx(0.0, abs=1e-7)
    assert errors["n"] == 2
