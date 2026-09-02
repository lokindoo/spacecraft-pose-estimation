"""
The named splits, and how to turn one into a dataloader.

Training, evaluation, conformal calibration and the reproduction script all
agree on the split names defined here, so exactly one place knows which label
file backs "lightbox" or "calibration". Legacy filenames are accepted alongside
the canonical ones, so an existing copy of the prepared subsets (under the
dated names the published runs wrote) drops in without being regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, TrainConfig
from .data import SpeedPlusDataset, build_dataloader, materialise_images
from .labels import load_labels
from .paths import (
    SUBSET_CONFORMAL,
    SUBSET_LIGHTBOX,
    SUBSET_SUNLAMP,
    SUBSET_SYNTHETIC,
    Paths,
)


@dataclass(frozen=True)
class Split:
    """One named split: which subset directory it lives in and its label file."""

    subset: str
    labels: str
    legacy_labels: tuple[str, ...] = ()
    #: Was this split ever used to fit or select a model? Anything true here is
    #: disqualified from conformal calibration.
    seen_in_training: bool = False


SPLITS: dict[str, Split] = {
    # --- synthetic, used for fitting and selection --------------------------
    "train": Split(
        SUBSET_SYNTHETIC,
        "train_labels.json",
        ("training_subset_2026-07-15_scaled_data.json",),
        seen_in_training=True,
    ),
    # The split every checkpoint was SELECTED on. Reported as "Synthetic (val)"
    # in Table 1, and deliberately not used for conformal calibration.
    "synthetic": Split(
        SUBSET_SYNTHETIC,
        "validation_labels.json",
        ("validation_subset_2026-07-15_scaled_data.json",),
        seen_in_training=True,
    ),
    # --- fresh synthetic, for conformal -------------------------------------
    "calibration": Split(
        SUBSET_CONFORMAL, "calibration_data.json"
    ),
    "in_domain_test": Split(
        SUBSET_CONFORMAL, "domain_test_data.json"
    ),
    # --- real hardware-in-the-loop test domains -----------------------------
    "lightbox": Split(
        SUBSET_LIGHTBOX, "test_labels.json", ("testing_subset_20260704.json",)
    ),
    "sunlamp": Split(
        SUBSET_SUNLAMP, "test_labels.json", ("testing_subset_20260704.json",)
    ),
}

#: Domains the results table reports, in the order Table 1 lists them.
REPORT_DOMAINS = ("synthetic", "lightbox", "sunlamp")

#: Domains the conformal stage measures coverage on, given a q_hat calibrated
#: on "calibration". "synthetic" is a sanity check only: it selected the
#: checkpoints, so it should read slightly above nominal, and a clean reading
#: there is not evidence of anything.
COVERAGE_DOMAINS = ("in_domain_test", "sunlamp", "lightbox", "synthetic")


def split_labels_path(split_name: str, paths: Paths) -> Path:
    """Resolve a split's label file, falling back to the legacy filename."""
    split = SPLITS[split_name]
    directory = paths.subset(split.subset)

    for candidate in (split.labels, *split.legacy_labels):
        path = directory / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"no label file for split {split_name!r} in {directory} "
        f"(looked for {split.labels} and {list(split.legacy_labels)}). "
        "Run scripts/prepare_data.py to build the subsets."
    )


def load_split_labels(split_name: str, paths: Paths) -> pd.DataFrame:
    return load_labels(split_labels_path(split_name, paths))


def build_dataset(
    split_name: str,
    paths: Paths,
    transform,
    labels: pd.DataFrame | None = None,
    local_images: str | Path | None = None,
) -> SpeedPlusDataset:
    """Materialise the images for a split and wrap it in a dataset."""
    split = SPLITS[split_name]
    if labels is None:
        labels = load_split_labels(split_name, paths)
    images_path = materialise_images(paths.subset(split.subset), local_images)
    return SpeedPlusDataset(images_path, labels, transform)


def build_eval_dataloader(
    split_name: str,
    paths: Paths,
    transform,
    train_config: TrainConfig | None = None,
    labels: pd.DataFrame | None = None,
    local_images: str | Path | None = None,
):
    """
    Dataloader for a split being measured, never shuffled, so predictions can
    be joined back to labels by position. The join is still verified on
    filename downstream.
    """
    dataset = build_dataset(split_name, paths, transform, labels, local_images)
    return build_dataloader(dataset, train_config or TrainConfig(), shuffle=False)


def describe(config: ExperimentConfig) -> str:
    backbone = "frozen" if config.model.backbone_frozen else "+".join(config.model.trainable_stages)
    augmentation = "aug" if config.augmentation.enabled else "no aug"
    return f"{config.arm} (backbone {backbone}, {augmentation}, seed {config.seed})"
