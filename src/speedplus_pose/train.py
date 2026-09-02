"""
The training loop, with resume.

Resume is not a nicety here: the study ran on free-tier Colab, where a session
can vanish mid-run. Checkpoints are written on even epochs to a directory that
outlives the session, and a restart picks up the latest one.

Which saved epoch becomes "the model" is a separate decision, taken later: the
checkpoint with the best synthetic validation rotation error, applied uniformly
to every arm, with no real-domain number involved.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .losses import pose_loss
from .metrics import rotation_error, translation_error_relative
from .model import SpacecraftPoseEstimator, count_parameters, set_train_mode
from .normalisation import TranslationNormaliser


@dataclass
class TrainingHistory:
    """What the loop accumulates, carried through checkpoints."""

    train_losses: list[float] = field(default_factory=list)
    validation_translation_errors: list[float] = field(default_factory=list)
    validation_rotation_errors: list[float] = field(default_factory=list)

    @property
    def epochs_completed(self) -> int:
        return len(self.train_losses)

    def best_epoch(self) -> int:
        """Index of the lowest validation rotation error, i.e. the selection rule."""
        return int(np.argmin(self.validation_rotation_errors))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    normaliser: TranslationNormaliser,
    history: TrainingHistory,
    config: ExperimentConfig,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "epoch": epoch,
            # Named train_means/train_stds for compatibility with the checkpoints
            # the published runs wrote.
            "train_means": normaliser.means.detach().cpu(),
            "train_stds": normaliser.stds.detach().cpu(),
            "train_losses": history.train_losses,
            "validation_translation_errors": history.validation_translation_errors,
            "validation_rotation_errors": history.validation_rotation_errors,
            "config": asdict(config),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def latest_checkpoint(directory: Path) -> Path | None:
    """
    Most recent checkpoint in a run directory. Filenames are UTC timestamps, so
    lexical order is chronological order.
    """
    checkpoints = sorted(directory.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _checkpoint_name() -> str:
    return f"checkpoint_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pt"


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader,
    normaliser: TranslationNormaliser,
    device: torch.device,
) -> tuple[float, float]:
    """Mean range-normalised translation error and mean rotation error."""
    model.eval()
    translation_errors, rotation_errors = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        r_pred = normaliser.remove(outputs[:, :3])
        # The quaternion half is not un-normalised; it is trained raw and unit.
        q_pred = outputs[:, 3:]
        q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)

        translation_errors.extend(translation_error_relative(r_pred, labels[:, :3]))
        rotation_errors.extend(rotation_error(q_pred, labels[:, 3:]))

    return float(np.mean(translation_errors)), float(np.mean(rotation_errors))


def train(
    model: SpacecraftPoseEstimator,
    config: ExperimentConfig,
    train_dataloader,
    validation_dataloader,
    normaliser: TranslationNormaliser,
    checkpoint_dir: Path,
    device: torch.device,
    resume: bool = True,
) -> TrainingHistory:
    """Train one arm. Returns the full history."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    history = TrainingHistory()
    start_epoch = 0

    if resume and (previous := latest_checkpoint(checkpoint_dir)) is not None:
        print(f"Resuming from {previous}")
        checkpoint = torch.load(previous, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        normaliser = TranslationNormaliser(
            checkpoint["train_means"], checkpoint["train_stds"], device
        )
        history = TrainingHistory(
            train_losses=checkpoint["train_losses"],
            validation_translation_errors=checkpoint["validation_translation_errors"],
            validation_rotation_errors=checkpoint["validation_rotation_errors"],
        )

    total, trainable = count_parameters(model)
    print(f"{config.arm}: {trainable:,} trainable of {total:,} parameters")

    for epoch in range(start_epoch, config.train.epochs):
        started = time.time()
        # Sets the per-arm BatchNorm regime, which is what distinguishes the
        # frozen arms from the unfrozen ones. See model.set_train_mode.
        set_train_mode(model, config.model)

        running_loss = 0.0
        for inputs, labels in tqdm(
            train_dataloader, desc=f"epoch {epoch + 1}/{config.train.epochs}", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            translation_targets = normaliser.apply(labels[:, :3])
            rotation_targets = labels[:, 3:]

            optimiser.zero_grad()
            outputs = model(inputs)
            loss, _, _ = pose_loss(
                outputs, translation_targets, rotation_targets, config.train.lambda_rotation
            )
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        val_t, val_r = evaluate_epoch(model, validation_dataloader, normaliser, device)
        history.train_losses.append(running_loss / len(train_dataloader))
        history.validation_translation_errors.append(val_t)
        history.validation_rotation_errors.append(val_r)

        print(
            f"Epoch {epoch + 1}, {time.time() - started:.1f} s. "
            f"Train loss: {history.train_losses[-1]:.6f} | "
            f"val_t: {val_t:.4f} | val_r: {val_r:.4f}"
        )

        if (epoch + 1) % config.train.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / _checkpoint_name(),
                model,
                optimiser,
                epoch,
                normaliser,
                history,
                config,
            )

    return history
