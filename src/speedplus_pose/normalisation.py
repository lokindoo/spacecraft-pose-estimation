"""
Translation target normalisation.

Targets are z-scored per component before the MSE term, and predictions are
un-scored before any error is computed. The constants are properties of the
training subset, so they travel with the checkpoint: evaluating with different
ones mangles every position prediction while still looking plausible.

One published checkpoint, the seed-44 unfrozen no-aug arm, predates the code
that stored them. ``constants_for_checkpoint`` recomputes from the training
subset in that case, and checks the recomputed values against the checkpoints
that do carry them rather than assuming they agree.
"""

from __future__ import annotations

import torch


class TranslationNormaliser:
    """Per-component z-scoring of translation targets."""

    def __init__(self, means, stds, device: torch.device | str | None = None):
        self.means = torch.as_tensor(means, dtype=torch.float32)
        self.stds = torch.as_tensor(stds, dtype=torch.float32)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device | str) -> "TranslationNormaliser":
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
        return self

    def apply(self, targets: torch.Tensor) -> torch.Tensor:
        """Metres -> z-scores. Used on labels before the loss."""
        return (targets - self.means) / self.stds

    def remove(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Z-scores -> metres, for the translation half of the model output only.

        The quaternion half is trained raw and unit-normalised, so passing it
        through here would corrupt every rotation prediction.
        """
        return (targets * self.stds) + self.means


def constants_for_checkpoint(
    checkpoint: dict,
    fallback_means,
    fallback_stds,
    arm_name: str = "",
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalisation constants for one checkpoint: the stored values where
    present, checked against constants recomputed from the training subset.

    Every arm trained on the same subset, so the two must agree. A disagreement
    means the checkpoint came from a different subset than you think, which
    would invalidate its translation numbers, so it warns rather than
    proceeding quietly.
    """
    fallback_means = torch.as_tensor(fallback_means, dtype=torch.float32)
    fallback_stds = torch.as_tensor(fallback_stds, dtype=torch.float32)

    if "train_means" not in checkpoint or "train_stds" not in checkpoint:
        print(f"  {arm_name}: checkpoint stores no constants, using recomputed fallback")
        return fallback_means.clone(), fallback_stds.clone()

    means = torch.as_tensor(checkpoint["train_means"]).detach().cpu().float()
    stds = torch.as_tensor(checkpoint["train_stds"]).detach().cpu().float()

    agrees = torch.allclose(
        means, fallback_means, rtol=tolerance, atol=tolerance
    ) and torch.allclose(stds, fallback_stds, rtol=tolerance, atol=tolerance)
    if not agrees:
        print(f"  {arm_name}: WARNING stored constants disagree with the training subset")
        print(f"    stored:     means={means.tolist()} stds={stds.tolist()}")
        print(f"    recomputed: means={fallback_means.tolist()} stds={fallback_stds.tolist()}")

    return means, stds
