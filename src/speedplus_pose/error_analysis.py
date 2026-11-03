"""
What the errors look like, not only how large they are on average.

Reviewer M2Lg asked whether the real-domain rotation errors are bimodal, and
whether they come from symmetry. Two summaries answer that from the prediction
records alone:

  - the share of errors near pi, set against the share uniform guessing would
    put there. A flip about a near-symmetry axis lands close to pi, but so does
    a lot of chance: under uniform rotations the angle has density
    (1 - cos t)/pi, which peaks at pi, so the raw share means nothing without
    the reference;
  - the axis of each error rotation, in the body frame and in the camera frame.
    A flip about a body symmetry axis repeats one body-frame axis however the
    target is posed; an in-plane confusion repeats the camera boresight. Both
    frames are reported because which one concentrates is the finding.

Frames follow scripts/image_utils.py, which draws the pose overlays: the label
quaternion is scalar-first and its matrix R maps body vectors into the camera
frame. A prediction R_pred = R_true S with S a body symmetry therefore gives a
body-frame error S = R_true^T R_pred. Were the convention the other way round,
the two frames would swap names and nothing else.

The conformal half of this module is coverage conditional on range, the
stratified version of the marginal coverage in coverage.py (reviewer kVr6).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .conformal import clopper_pearson_interval, compute_coverage

# Errors within this distance of pi count as "near pi" (radians).
NEAR_PI_WIDTH = 0.3

# Axes within this angle of the principal axis count as concentrated on it.
AXIS_CONE_DEGREES = 20.0

# Below this angle the axis of an error rotation is too noisy to mean anything.
MIN_AXIS_ANGLE = 0.5


def uniform_tail_share(width: float = NEAR_PI_WIDTH) -> float:
    """
    Share of uniformly random rotations whose angle exceeds pi - width.

    From F(a) = (a - sin a)/pi: 1 - F(pi - w) = (w + sin w)/pi, about 0.19 for
    w = 0.3. Chance alone puts this much mass near pi.
    """
    return float((width + np.sin(width)) / np.pi)


def uniform_angle_density(theta) -> np.ndarray:
    """Density of the rotation angle under uniformly random orientations."""
    return (1 - np.cos(np.asarray(theta, dtype=float))) / np.pi


def near_pi_share(errors, width: float = NEAR_PI_WIDTH) -> float:
    errors = np.asarray(errors, dtype=float)
    return float(np.mean(errors > np.pi - width))


def error_rotations(q_pred, q_true, frame: str = "body") -> tuple[np.ndarray, np.ndarray]:
    """
    Angle and unit axis of each error rotation, expressed in ``frame``.

    body:   R_true^T R_pred, the rotation that would turn the true body
            orientation into the predicted one about body axes;
    camera: R_pred R_true^T, the same error about camera axes.

    The angle equals metrics.rotation_error either way; only the axis differs.
    """
    true = Rotation.from_quat(np.asarray(q_true, dtype=float), scalar_first=True)
    pred = Rotation.from_quat(np.asarray(q_pred, dtype=float), scalar_first=True)
    if frame == "body":
        delta = true.inv() * pred
    elif frame == "camera":
        delta = pred * true.inv()
    else:
        raise ValueError(f"frame must be 'body' or 'camera', not {frame!r}")

    rotvec = delta.as_rotvec()
    angle = np.linalg.norm(rotvec, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = rotvec / angle[:, None]
    return angle, axis


def axial_concentration(axes, cone_degrees: float = AXIS_CONE_DEGREES) -> dict[str, float]:
    """
    How strongly a set of axes clusters around one direction.

    Axes are axial: at an angle of pi, a and -a are the same rotation, so a mean
    direction is meaningless. The scatter matrix T = mean(a a^T) is the right
    summary. Its largest eigenvalue runs from 1/3 (uniform) to 1 (every axis
    identical), and its eigenvector is the principal axis. The share of axes
    within ``cone_degrees`` of that axis is the readable version, with
    1 - cos(cone) as its uniform reference (0.06 at 20 degrees).
    """
    axes = np.asarray(axes, dtype=float)
    axes = axes[np.all(np.isfinite(axes), axis=1)]
    if len(axes) == 0:
        return {"n_axes": 0, "eigenvalue": np.nan, "axis_x": np.nan, "axis_y": np.nan,
                "axis_z": np.nan, "share_in_cone": np.nan}

    scatter = axes.T @ axes / len(axes)
    values, vectors = np.linalg.eigh(scatter)
    principal = vectors[:, -1]
    # The eigenvector's sign is arbitrary; fix it so reruns print the same axis.
    principal = principal * np.sign(principal[np.argmax(np.abs(principal))])
    share = np.mean(np.abs(axes @ principal) >= np.cos(np.radians(cone_degrees)))
    return {
        "n_axes": int(len(axes)),
        "eigenvalue": float(values[-1]),
        "axis_x": float(principal[0]),
        "axis_y": float(principal[1]),
        "axis_z": float(principal[2]),
        "share_in_cone": float(share),
    }


def uniform_cone_share(cone_degrees: float = AXIS_CONE_DEGREES) -> float:
    """Share of uniformly random axes within ``cone_degrees`` of a fixed axis."""
    return float(1 - np.cos(np.radians(cone_degrees)))


def range_bin_edges(calibration_range, bins: int) -> np.ndarray:
    """
    Quantile edges of the calibration split's range, with open outer edges.

    The bins come from calibration data only, so no test distribution shapes
    them, and the real domains (whose ranges match synthetic) land in bins of
    comparable size.
    """
    inner = np.quantile(np.asarray(calibration_range, dtype=float), np.linspace(0, 1, bins + 1)[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def coverage_by_bin(scores, ranges, q_hat: float, edges: np.ndarray) -> list[dict]:
    """
    Coverage of a fixed q_hat within each range bin, with Clopper-Pearson
    intervals. q_hat is the marginal one: this measures how the marginal
    guarantee distributes over range, not a per-bin recalibration.
    """
    scores = np.asarray(scores, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    which = np.digitize(ranges, edges[1:-1])
    rows = []
    for index in range(len(edges) - 1):
        mask = which == index
        count, nobs = compute_coverage(scores[mask], q_hat)
        low, high = clopper_pearson_interval(count, nobs) if nobs else (np.nan, np.nan)
        rows.append({
            "bin": index,
            "range_low": float(edges[index]),
            "range_high": float(edges[index + 1]),
            "range_median": float(np.median(ranges[mask])) if nobs else np.nan,
            "count": int(count),
            "n": int(nobs),
            "coverage": float(count / nobs) if nobs else np.nan,
            "coverage_low": float(low),
            "coverage_high": float(high),
        })
    return rows
