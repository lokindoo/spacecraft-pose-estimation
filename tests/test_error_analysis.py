"""
Tests for the error-shape and range-stratified coverage analyses.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from speedplus_pose.conformal import so3_ball_fraction
from speedplus_pose.error_analysis import (
    axial_concentration,
    coverage_by_bin,
    error_rotations,
    near_pi_share,
    range_bin_edges,
    uniform_cone_share,
    uniform_tail_share,
)
from speedplus_pose.metrics import rotation_error


def _random_quaternions(n, seed):
    return Rotation.random(n, random_state=seed).as_quat(scalar_first=True)


def test_uniform_tail_share_matches_so3_fraction():
    assert uniform_tail_share(0.3) == pytest.approx(1 - so3_ball_fraction(np.pi - 0.3))
    assert uniform_tail_share(0.3) == pytest.approx(0.1896, abs=1e-4)


def test_uniform_rotations_hit_the_tail_reference():
    q_true = _random_quaternions(200_000, seed=0)
    q_pred = _random_quaternions(200_000, seed=1)
    errors = rotation_error(q_pred, q_true)
    assert near_pi_share(errors) == pytest.approx(uniform_tail_share(), abs=0.004)


def test_body_symmetry_flip_shows_in_body_frame_only():
    # Every prediction is the truth flipped by pi about body z, the signature of
    # a two-fold symmetry. The body-frame axis is always z; the camera-frame
    # axis wanders with the pose.
    q_true = _random_quaternions(2000, seed=2)
    flip = Rotation.from_rotvec([0.0, 0.0, np.pi])
    q_pred = (Rotation.from_quat(q_true, scalar_first=True) * flip).as_quat(scalar_first=True)

    angle_body, axes_body = error_rotations(q_pred, q_true, "body")
    angle_camera, axes_camera = error_rotations(q_pred, q_true, "camera")

    np.testing.assert_allclose(angle_body, np.pi, atol=1e-6)
    np.testing.assert_allclose(angle_camera, rotation_error(q_pred, q_true), atol=1e-6)

    body = axial_concentration(axes_body)
    camera = axial_concentration(axes_camera)
    assert body["eigenvalue"] == pytest.approx(1.0, abs=1e-6)
    assert abs(body["axis_z"]) == pytest.approx(1.0, abs=1e-6)
    assert body["share_in_cone"] == 1.0
    assert camera["eigenvalue"] < 0.4
    assert camera["share_in_cone"] == pytest.approx(uniform_cone_share(), abs=0.02)


def test_in_plane_confusion_shows_in_camera_frame_only():
    q_true = _random_quaternions(2000, seed=3)
    roll = Rotation.from_rotvec([0.0, 0.0, 2.5])  # about the camera boresight
    q_pred = (roll * Rotation.from_quat(q_true, scalar_first=True)).as_quat(scalar_first=True)

    camera = axial_concentration(error_rotations(q_pred, q_true, "camera")[1])
    body = axial_concentration(error_rotations(q_pred, q_true, "body")[1])
    assert abs(camera["axis_z"]) == pytest.approx(1.0, abs=1e-6)
    assert body["eigenvalue"] < 0.4


def test_axial_concentration_ignores_axis_sign():
    axes = np.array([[0, 0, 1.0], [0, 0, -1.0]] * 50)
    result = axial_concentration(axes)
    assert result["eigenvalue"] == pytest.approx(1.0)
    assert result["share_in_cone"] == 1.0


def test_range_bins_partition_and_recover_marginal_coverage():
    rng = np.random.default_rng(0)
    calibration_range = rng.uniform(2, 10, 1000)
    edges = range_bin_edges(calibration_range, bins=5)
    assert len(edges) == 6 and np.isinf(edges[0]) and np.isinf(edges[-1])

    ranges = rng.uniform(1, 12, 3000)  # wider than calibration: the open edges catch it
    scores = rng.exponential(1.0, 3000) * ranges / 5
    q_hat = 1.5
    rows = coverage_by_bin(scores, ranges, q_hat, edges)

    assert sum(row["n"] for row in rows) == 3000
    assert sum(row["count"] for row in rows) == int(np.sum(scores <= q_hat))
    # Scores grow with range, so a single threshold over-covers near and
    # under-covers far: the effect the stratification exists to expose.
    assert rows[0]["coverage"] > rows[-1]["coverage"]
