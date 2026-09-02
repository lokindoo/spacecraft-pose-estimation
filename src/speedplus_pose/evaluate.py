"""
Inference over one or more domains, producing the prediction records every
table and figure is computed from.

Predictions are stored rather than recomputed because they are the join point:
the results table, the conformal analysis and the figures all read the same
arrays. Each record holds filenames, the predicted translation in metres, the
predicted unit quaternion, and both error families, so the conformal stage can
rescore under a different metric without a second GPU pass.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .metrics import (
    rotation_error,
    translation_error_metres,
    translation_error_relative,
)
from .normalisation import TranslationNormaliser


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader,
    normaliser: TranslationNormaliser,
    device: torch.device,
    description: str = "",
) -> dict[str, np.ndarray]:
    """Run one domain through the model and return a prediction record."""
    model.eval()
    r_pred_batches, q_pred_batches = [], []
    r_true_batches, q_true_batches = [], []

    for inputs, labels in tqdm(dataloader, desc=description or "predicting", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        r_pred = normaliser.remove(outputs[:, :3]).detach().cpu().numpy()
        q_pred = outputs[:, 3:].detach().cpu().numpy()
        q_pred = q_pred / np.linalg.norm(q_pred, axis=1, keepdims=True)

        r_pred_batches.append(r_pred)
        q_pred_batches.append(q_pred)
        r_true_batches.append(labels[:, :3].detach().cpu().numpy())
        q_true_batches.append(labels[:, 3:].detach().cpu().numpy())

    r_pred = np.concatenate(r_pred_batches)
    q_pred = np.concatenate(q_pred_batches)
    r_true = np.concatenate(r_true_batches)
    q_true = np.concatenate(q_true_batches)

    filenames = dataloader.dataset.filenames
    if len(filenames) != len(r_pred):
        raise AssertionError(
            f"{description}: {len(filenames)} filenames vs {len(r_pred)} predictions"
        )

    return {
        "filename": filenames,
        "r_pred": r_pred,
        "q_pred": q_pred,
        "r_true": r_true,
        "q_true": q_true,
        "translation_relative": translation_error_relative(r_pred, r_true),
        "translation_metres": translation_error_metres(r_pred, r_true),
        "rotation": rotation_error(q_pred, q_true),
        # True distance to the target, for the range-bin slice of the conformal
        # analysis and for confirming the domains pose matched difficulty.
        "range": np.linalg.norm(r_true, axis=1),
    }


def summarise(record: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean errors for one (arm, domain), i.e. one row of the results table."""
    translation = float(np.mean(record["translation_relative"]))
    rotation = float(np.mean(record["rotation"]))
    return {
        "translation_relative": translation,
        "translation_metres": float(np.mean(record["translation_metres"])),
        "rotation": rotation,
        "speed_score": translation + rotation,
        "n": int(len(record["rotation"])),
    }


def save_predictions(path: str | Path, predictions: dict) -> Path:
    """
    Write predictions as a pickle, after every arm rather than once at the end,
    so a disconnected Colab session costs one arm and not the whole sweep.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(predictions, handle)
    return path


def load_predictions(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)
