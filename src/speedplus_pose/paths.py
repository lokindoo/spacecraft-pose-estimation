"""
Where things live.

Two roots are resolved once, from the environment or an explicit argument, and
every other location hangs off them:

    SPEEDPLUS_DATA_ROOT   raw SPEED+ download and the derived subsets
    SPEEDPLUS_ARTIFACTS   checkpoints, predictions, figures

``DATASET_PATH`` is accepted as a fallback for the data root, since that is the
name the project's existing ``.env`` uses.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_VARS = ("SPEEDPLUS_DATA_ROOT", "DATASET_PATH")
ARTIFACTS_ROOT_VARS = ("SPEEDPLUS_ARTIFACTS",)

# Subdirectories of the raw SPEED+ download, as shipped on Zenodo record 5588480.
RAW_DOMAINS = ("synthetic", "lightbox", "sunlamp")

# Subset directory names produced by scripts/prepare_data.py. These match the
# names the published runs used, so an existing Drive copy drops straight in.
SUBSET_SYNTHETIC = "synthetic_28k_train_2k_val_2026-07-15"
SUBSET_CONFORMAL = "synthetic_split_conformal_coverage"
SUBSET_SUNLAMP = "sunlamp_downscaled"
SUBSET_LIGHTBOX = "lightbox_downscaled"


def _from_env(names: tuple[str, ...]) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            # tolerate quoted values, which is how the project's .env writes them
            return Path(value.strip().strip('"').strip("'")).expanduser()
    return None


def data_root(override: str | os.PathLike | None = None) -> Path:
    """Root holding the raw SPEED+ download and the derived ``subsets/`` tree."""
    root = Path(override).expanduser() if override else _from_env(DATA_ROOT_VARS)
    if root is None:
        raise RuntimeError(
            "No SPEED+ data root. Set SPEEDPLUS_DATA_ROOT (or DATASET_PATH), or pass "
            "--data-root. See README.md for how to obtain SPEED+ from Zenodo record "
            "5588480 - this project does not redistribute the dataset."
        )
    return root


def artifacts_root(override: str | os.PathLike | None = None) -> Path:
    """Root for everything runs produce: checkpoints, predictions, figures."""
    root = Path(override).expanduser() if override else _from_env(ARTIFACTS_ROOT_VARS)
    if root is None:
        raise RuntimeError(
            "No artifacts root. Set SPEEDPLUS_ARTIFACTS or pass --artifacts-root. "
            "On Colab this is typically a directory on the mounted Drive so a "
            "disconnect does not cost the run."
        )
    return root


class Paths:
    """Resolved locations for one invocation."""

    def __init__(
        self,
        data: str | os.PathLike | None = None,
        artifacts: str | os.PathLike | None = None,
    ):
        self.data = data_root(data)
        self.artifacts = artifacts_root(artifacts)

    # --- raw download -----------------------------------------------------
    def raw_images(self, domain: str) -> Path:
        self._check_domain(domain)
        return self.data / domain / "images"

    def raw_labels(self, domain: str, split: str) -> Path:
        """e.g. raw_labels("synthetic", "train") -> <root>/synthetic/train.json"""
        self._check_domain(domain)
        return self.data / domain / f"{split}.json"

    def camera(self) -> Path:
        return self.data / "camera.json"

    # --- derived subsets --------------------------------------------------
    @property
    def subsets(self) -> Path:
        return self.data / "subsets"

    def subset(self, name: str) -> Path:
        return self.subsets / name

    # --- run outputs ------------------------------------------------------
    def checkpoints(self, run_name: str) -> Path:
        return self.artifacts / "model_checkpoints" / run_name

    @property
    def predictions(self) -> Path:
        return self.artifacts / "predictions"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @staticmethod
    def _check_domain(domain: str) -> None:
        if domain not in RAW_DOMAINS:
            raise ValueError(f"unknown domain {domain!r}, expected one of {RAW_DOMAINS}")

    def __repr__(self) -> str:
        return f"Paths(data={self.data!s}, artifacts={self.artifacts!s})"
