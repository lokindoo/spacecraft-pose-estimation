"""
Metric tests.

These pin down the two things a refactor is most likely to break quietly: that
the two translation metrics stay distinct, and that rotation error stays
sign-invariant.
"""

import numpy as np
import pytest

from speedplus_pose.metrics import (
    CHANCE_LEVEL,
    rotation_error,
    skill,
    skill_retained,
    speed_score,
    translation_error_metres,
    translation_error_relative,
)


def test_translation_error_metres():
    assert translation_error_metres([[0, 0, 0]], [[3, 4, 0]]) == 5.0


def test_translation_metrics_are_different():
    """
    The two translation metrics must not be interchangeable.

    A 5 m miss on a target 10 m away is 0.5 relative; the same miss at 50 m is
    0.1. The conformal analysis calibrates on metres, where q_hat is a keep-out
    sphere radius, while Table 1 reports the relative one, so swapping them
    would move half the published numbers.
    """
    r_pred = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 45.0]])
    r_true = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 50.0]])

    assert np.allclose(translation_error_metres(r_pred, r_true), [5.0, 5.0])
    assert np.allclose(translation_error_relative(r_pred, r_true), [0.5, 0.1])


def test_rotation_error_is_sign_invariant():
    """q and -q are the same rotation, so the error between them must be zero."""
    q = np.array([[1.0, -0.2, 0.5, 0.0]])
    assert rotation_error(q, q) == 0
    assert rotation_error(q, -q) == 0


def test_rotation_error_known_angles():
    assert rotation_error([[1, 0, 0, 0]], [[0, 1, 0, 0]]) == np.pi
    half = np.sqrt(0.5)
    assert np.isclose(rotation_error([[half, half, 0, 0]], [[1, 0, 0, 0]]), np.pi / 2)


def test_rotation_error_normalises_input():
    """Un-normalised predictions must score the same as normalised ones."""
    q_pred = np.array([[2.0, 0.0, 0.0, 0.0]])
    q_true = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.isclose(rotation_error(q_pred, q_true), 0.0)


def test_chance_level():
    """
    The mean geodesic angle between two uniformly random orientations, checked
    against the closed form pi/2 + 2/pi by sampling. The density of the angle is
    (1 - cos(theta))/pi on [0, pi].
    """
    assert np.isclose(CHANCE_LEVEL, np.pi / 2 + 2 / np.pi)

    rng = np.random.default_rng(0)
    q = rng.standard_normal((200_000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    angles = rotation_error(q, np.roll(q, 1, axis=0))
    assert abs(angles.mean() - CHANCE_LEVEL) < 0.01


def test_skill_is_distance_below_chance():
    """1.67 and 2.07 rad look close; as skill they are 0.54 and 0.14."""
    assert np.allclose(skill([1.67, 2.07]), [CHANCE_LEVEL - 1.67, CHANCE_LEVEL - 2.07])
    assert np.isclose(skill([1.67])[0], 0.537, atol=1e-3)
    assert np.isclose(skill([2.07])[0], 0.137, atol=1e-3)


def test_skill_retained():
    """The scaled baseline keeps ~14% of its synthetic skill on sunlamp."""
    assert np.isclose(skill_retained(1.053, 2.048), 0.138, atol=1e-3)
    # A model already at chance has no skill to retain.
    assert np.isnan(skill_retained(CHANCE_LEVEL, 1.0))


def test_speed_score_is_the_plain_sum():
    assert np.allclose(speed_score([0.15], [1.19]), [1.34])


def test_wrong_shaped_input_raises():
    """
    Transposed input is the dangerous case: (3, N) instead of (N, 3) would
    otherwise return a plausible-looking array of the wrong length.
    """
    with pytest.raises(ValueError):
        translation_error_metres(np.zeros((3, 10)), np.zeros((3, 10)))
    with pytest.raises(ValueError):
        translation_error_metres([0, 0, 0], [3, 4, 0])
    with pytest.raises(ValueError):
        translation_error_metres(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        rotation_error(np.zeros((4, 10)), np.zeros((4, 10)))
