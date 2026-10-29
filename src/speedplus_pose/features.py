"""
Pooled-feature caching for the frozen arms.

A frozen arm's backbone is a fixed function of the image: eval mode, fixed
BatchNorm statistics, no augmentation. Its 2048-d pooled features can therefore
be computed once, and every head experiment (extra seeds, the learning-rate x
epochs sweep, the 4k-vs-28k and logits-vs-pooled decomposition) then runs on
CPU in minutes rather than costing a GPU pass each.

Only the no-augmentation frozen arm is cacheable. With augmentation on, the
features change every epoch.

The 1000-d ImageNet logits the project's first baseline read are fc(pooled), so
they come from the cache and the ImageNet ``fc`` weights rather than from a
second pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

#: Every split a head experiment needs: fit, select, calibrate, and the three
#: coverage domains. in_domain_test is the conformal in-domain control.
CACHE_SPLITS = ("train", "synthetic", "calibration", "in_domain_test", "lightbox", "sunlamp")

#: The two BatchNorm regimes a frozen backbone can run with.
BN_REGIMES = ("imagenet", "synthetic")


@torch.no_grad()
def extract_features(backbone, dataloader, device, description: str = "") -> dict[str, np.ndarray]:
    """
    Pooled features, labels and filenames for one split, in dataloader order.
    ``backbone`` is a ResNet-50 with ``fc`` replaced by identity.
    """
    backbone.eval()
    features, labels = [], []
    for inputs, batch_labels in tqdm(dataloader, desc=description or "features", leave=False):
        features.append(backbone(inputs.to(device)).float().cpu().numpy())
        labels.append(np.asarray(batch_labels, dtype=np.float32))

    record = {
        "features": np.concatenate(features).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.float32),
        # A fixed-width string array, not pandas' object dtype, so the cache
        # loads without allow_pickle.
        "filename": np.asarray(dataloader.dataset.filenames).astype(str),
    }
    if len(record["filename"]) != len(record["features"]):
        raise AssertionError(
            f"{description}: {len(record['filename'])} filenames vs "
            f"{len(record['features'])} feature rows"
        )
    return record


def save_features(path: str | Path, record: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **record)
    return path


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_cache(directory: str | Path, splits=CACHE_SPLITS) -> dict[str, dict[str, np.ndarray]]:
    directory = Path(directory)
    missing = [s for s in splits if not (directory / f"{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing cached split(s) {missing}; run scripts/cache_features.py"
        )
    return {split: load_features(directory / f"{split}.npz") for split in splits}


def imagenet_fc() -> tuple[np.ndarray, np.ndarray]:
    """
    The ImageNet classifier the backbone was pretrained with: weight (1000,
    2048) and bias (1000,). The same IMAGENET1K_V2 weights the model loads, so
    fc(pooled) here equals the logits a full ResNet-50 would emit.
    """
    state = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=False)
    return state["fc.weight"].numpy(), state["fc.bias"].numpy()


def to_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (features @ weight.T + bias).astype(np.float32)
