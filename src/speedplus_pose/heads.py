"""
Training a pose head on cached features: the frozen arm without the backbone.

The head, loss, optimiser and selection rule are the published frozen arm's.
Only the input changes: rows of a feature cache instead of images through a
frozen ResNet-50, which is the same function computed once. That is what makes
the reviewers' two questions about the frozen arm cheap to answer:

  - is 1.05 rad a property of frozen features, or an under-trained head?
    (the learning-rate x epochs sweep)
  - which of the two bundled changes, 4k -> 28k images or 1000-d logits ->
    2048-d pooled features, produced the synthetic gain? (the decomposition)

Predictions come out in the record format evaluate.predict writes, so the
conformal analysis and the figures read them unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .losses import pose_loss
from .metrics import rotation_error, translation_error_metres, translation_error_relative
from .model import POSE_OUTPUTS
from .normalisation import TranslationNormaliser


def build_head(in_features: int, hidden: int = 1024) -> nn.Sequential:
    """
    Layer for layer the published ``model.fc``, so a published head's
    ``fc.*`` weights load straight into it.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, POSE_OUTPUTS),
    )


def head_from_checkpoint(checkpoint: dict, hidden: int = 1024) -> nn.Sequential:
    """The head of a published end-to-end checkpoint, without its backbone."""
    state = {
        key.removeprefix("fc."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("fc.")
    }
    head = build_head(state["0.weight"].shape[1], hidden)
    head.load_state_dict(state)
    return head


@dataclass(frozen=True)
class HeadConfig:
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 16
    lambda_rotation: float = 1.0
    checkpoint_every: int = 2
    hidden: int = 1024
    seed: int = 42


@dataclass
class HeadResult:
    head: nn.Sequential
    selected_epoch: int  # 0-indexed
    validation_rotation_errors: list[float]
    validation_translation_errors: list[float]
    train_losses: list[float]


@torch.no_grad()
def _outputs(head: nn.Module, features: np.ndarray, batch_size: int = 4096) -> torch.Tensor:
    head.eval()
    x = torch.from_numpy(features)
    return torch.cat([head(x[i : i + batch_size]) for i in range(0, len(x), batch_size)])


def predict_head(
    head: nn.Module, record: dict[str, np.ndarray], normaliser: TranslationNormaliser
) -> dict[str, np.ndarray]:
    """A prediction record in evaluate.predict's format, from cached features."""
    outputs = _outputs(head, record["features"])
    r_pred = normaliser.remove(outputs[:, :3]).numpy()
    q_pred = outputs[:, 3:].numpy()
    q_pred = q_pred / np.linalg.norm(q_pred, axis=1, keepdims=True)
    r_true = record["labels"][:, :3]
    q_true = record["labels"][:, 3:]
    return {
        "filename": record["filename"],
        "r_pred": r_pred,
        "q_pred": q_pred,
        "r_true": r_true,
        "q_true": q_true,
        "translation_relative": translation_error_relative(r_pred, r_true),
        "translation_metres": translation_error_metres(r_pred, r_true),
        "rotation": rotation_error(q_pred, q_true),
        "range": np.linalg.norm(r_true, axis=1),
    }


def train_head(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    normaliser: TranslationNormaliser,
    config: HeadConfig,
) -> HeadResult:
    """
    Train one head and apply the selection rule: among the epochs that would
    have been checkpointed, the best synthetic validation rotation error.
    """
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    x = torch.from_numpy(train["features"])
    y = torch.from_numpy(train["labels"])
    head = build_head(x.shape[1], config.hidden)
    optimiser = torch.optim.Adam(head.parameters(), lr=config.learning_rate)

    snapshots: dict[int, dict] = {}
    losses, val_r, val_t = [], [], []
    for epoch in range(config.epochs):
        head.train()
        order = torch.randperm(len(x), generator=generator)
        running, batches = 0.0, 0
        for start in range(0, len(x), config.batch_size):
            index = order[start : start + config.batch_size]
            labels = y[index]
            optimiser.zero_grad()
            loss, _, _ = pose_loss(
                head(x[index]),
                normaliser.apply(labels[:, :3]),
                labels[:, 3:],
                config.lambda_rotation,
            )
            loss.backward()
            optimiser.step()
            running += loss.item()
            batches += 1

        record = predict_head(head, validation, normaliser)
        losses.append(running / batches)
        val_r.append(float(np.mean(record["rotation"])))
        val_t.append(float(np.mean(record["translation_relative"])))
        if (epoch + 1) % config.checkpoint_every == 0:
            snapshots[epoch] = copy.deepcopy(head.state_dict())

    if not snapshots:
        raise ValueError(
            f"{config.epochs} epochs with checkpoint_every={config.checkpoint_every} "
            "saves nothing to select from"
        )
    selected = min(snapshots, key=lambda epoch: (val_r[epoch], epoch))
    head.load_state_dict(snapshots[selected])
    return HeadResult(head, selected, val_r, val_t, losses)


def subsample(record: dict[str, np.ndarray], size: int, seed: int) -> dict[str, np.ndarray]:
    """A fixed random subset of a cached split, e.g. 4k of the 28k training rows."""
    if size > len(record["filename"]):
        raise ValueError(f"cannot draw {size} rows from {len(record['filename'])}")
    index = np.sort(np.random.default_rng(seed).choice(len(record["filename"]), size, replace=False))
    return {key: value[index] for key, value in record.items()}
