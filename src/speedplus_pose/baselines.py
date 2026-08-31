"""
Trivial constant predictors - the anchor the "barely better than chance" claim
needs.

CHANCE_LEVEL (pi/2 + 2/pi ~ 2.207 rad) is the error of a *uniformly random*
orientation guess. That is correct arithmetic for a uniform prior, and SPEED+
poses are not uniform on SO(3): the Tango mockup is imaged over a constrained
range of attitudes, so a predictor that ignores the image entirely and always
emits one fixed pose should beat 2.207. How far it beats it is the number that
decides whether an arm at 2.048 rad has learned anything at all, or is in fact
worse than doing nothing.

Nothing here touches an image, a checkpoint, or a GPU. Labels only.

TWO estimators per quantity, and the distinction is the point:

  chordal_mean_quaternion    closed form, the top eigenvector of sum q q^T. It
                             maximises sum cos^2(theta_i / 2), i.e. minimises a
                             *squared-chordal* loss. Sign-invariant, so the
                             double cover cannot move it. This is the estimator
                             the mock review asked for.
  best_constant_quaternion   the same thing refined to minimise the metric we
                             actually report, mean geodesic angle. The paper's
                             claim is about mean E_R, so the honest anchor is the
                             constant pose that minimises mean E_R, not one that
                             minimises a different loss and is quoted under this
                             one. It starts from the closed form and can only
                             improve on it.

The same split applies to translation: the arithmetic mean minimises squared
metres, while Table 1 reports ||dr|| / ||r||, whose optimum is pulled toward
close-range poses. best_constant_translation optimises the reported metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .metrics import rotation_error, translation_error_relative

QUATERNION_COLUMNS = ["q1", "q2", "q3", "q4"]
TRANSLATION_COLUMNS = ["r1", "r2", "r3"]


def _unit(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def chordal_mean_quaternion(quaternions) -> np.ndarray:
    """
    Closed-form mean orientation: the top eigenvector of sum_i q_i q_i^T.

    Exact and iteration-free. Because each q enters as an outer product, q and
    -q contribute identically - which is what makes this safe to run on labels
    whether or not they have been canonicalised.
    """
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ q)
    return _unit(eigenvectors[:, int(np.argmax(eigenvalues))])


def mean_translation(translations) -> np.ndarray:
    """Arithmetic mean position - the least-squares constant in metres."""
    return np.asarray(translations, dtype=float).mean(axis=0)


def best_constant_quaternion(quaternions, start: np.ndarray | None = None) -> np.ndarray:
    """
    The constant orientation minimising *mean geodesic error* over the labels.

    Refines ``chordal_mean_quaternion`` by Nelder-Mead over a three-parameter
    perturbation applied in the tangent space of the starting quaternion, then
    renormalising. Four parameters would be one too many - the norm constraint
    removes a degree of freedom - and a derivative-free search is ample for
    three unknowns fitted once, offline.

    Never returns something worse than its start: the refined point is accepted
    only if its mean error is lower.
    """
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q0 = _unit(np.asarray(start, dtype=float)) if start is not None else chordal_mean_quaternion(q)

    # An orthonormal basis of the tangent space at q0, so the search directions
    # move the orientation rather than its scale.
    basis = np.linalg.svd(q0[None, :])[2][1:]

    def mean_error(delta: np.ndarray) -> float:
        candidate = _unit(q0 + delta @ basis)
        return float(np.mean(rotation_error(np.tile(candidate, (len(q), 1)), q)))

    result = minimize(mean_error, np.zeros(3), method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    refined = _unit(q0 + result.x @ basis)
    return refined if mean_error(result.x) < mean_error(np.zeros(3)) else q0


def best_constant_translation(translations, start: np.ndarray | None = None) -> np.ndarray:
    """
    The constant position minimising *mean range-normalised error*, the E_T of
    Table 1.

    Not the arithmetic mean: dividing by ||r_true|| weights near targets more
    heavily, so the optimum sits closer to the camera than the centroid does.
    Nelder-Mead from the centroid, accepted only if it improves.
    """
    r = np.asarray(translations, dtype=float)
    r0 = np.asarray(start, dtype=float) if start is not None else mean_translation(r)

    def mean_error(candidate: np.ndarray) -> float:
        return float(np.mean(translation_error_relative(np.tile(candidate, (len(r), 1)), r)))

    result = minimize(mean_error, r0, method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    return result.x if mean_error(result.x) < mean_error(r0) else r0


def constant_predictor_errors(r_constant, q_constant, labels) -> dict[str, float]:
    """
    Score one constant pose against a label frame (``labels.load_labels``
    output), in exactly the metrics Table 1 reports.
    """
    r_true = labels[TRANSLATION_COLUMNS].to_numpy(dtype=float)
    q_true = labels[QUATERNION_COLUMNS].to_numpy(dtype=float)

    r_pred = np.tile(np.asarray(r_constant, dtype=float), (len(labels), 1))
    q_pred = np.tile(_unit(np.asarray(q_constant, dtype=float)), (len(labels), 1))

    translation = float(np.mean(translation_error_relative(r_pred, r_true)))
    rotation = float(np.mean(rotation_error(q_pred, q_true)))
    return {
        "translation_relative": translation,
        "rotation": rotation,
        "speed_score": translation + rotation,
        "n": int(len(labels)),
    }
