"""
The split pose loss.

    L = MSE(t_hat, t) + lambda * (1 - <q_hat, q>^2)

Split rather than one MSE over all seven outputs, because the translation
targets are z-scored and the quaternion components are not: a single MSE lets
translation dominate and rotation barely learns. Separating the two terms took
synthetic rotation error from ~1.6 rad into the tens of degrees.

Squaring the inner product makes the rotation term sign-invariant, so q and -q
give the same loss. With canonicalised labels (labels.py), the double cover
cannot hand the network contradictory targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_mse = nn.MSELoss()


def rotation_loss(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """1 - <q_hat, q>^2, averaged over the batch. Zero iff the rotations agree."""
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    dot = (q_pred * q_true).sum(dim=1)
    return (1.0 - dot**2).mean()


def translation_loss(t_pred: torch.Tensor, t_true: torch.Tensor) -> torch.Tensor:
    """MSE on z-scored translation targets."""
    return _mse(t_pred, t_true)


def pose_loss(
    outputs: torch.Tensor,
    translation_targets: torch.Tensor,
    rotation_targets: torch.Tensor,
    lambda_rotation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ``(total, translation_term, rotation_term)``, so the two halves can
    be logged separately when diagnosing one drowning the other.
    """
    t_loss = translation_loss(outputs[:, :3], translation_targets)
    r_loss = rotation_loss(outputs[:, 3:], rotation_targets)
    return t_loss + lambda_rotation * r_loss, t_loss, r_loss
