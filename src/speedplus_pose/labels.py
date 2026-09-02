"""
SPEED+ label handling: read a label JSON, unpack the translation and quaternion
lists into scalar columns, and canonicalise the quaternion sign.

Canonicalisation matters. A rotation is represented by both q and -q (the
double cover), so an uncanonicalised label set hands the network contradictory
targets for identical rotations. The training loss is separately sign-invariant
because it squares the inner product, but canonicalising removes the ambiguity
rather than merely tolerating it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILENAME_COLUMN = "filename"
TRANSLATION_COLUMN = "r_Vo2To_vbs_true"
QUATERNION_COLUMN = "q_vbs2tango_true"

TRANSLATION_COLUMNS = ["r1", "r2", "r3"]
QUATERNION_COLUMNS = ["q1", "q2", "q3", "q4"]
LABEL_COLUMNS = TRANSLATION_COLUMNS + QUATERNION_COLUMNS

# ImageDataset reads the filename positionally (column 0) and the seven label
# values as columns 1:, so the order is part of the contract.
COLUMN_ORDER = [FILENAME_COLUMN] + LABEL_COLUMNS


def canonicalise_quaternions(data: pd.DataFrame) -> pd.DataFrame:
    """Flip every quaternion into the q1 >= 0 hemisphere, in place."""
    mask = data["q1"] < 0
    data.loc[mask, QUATERNION_COLUMNS] *= -1
    return data


def load_labels(path: str | Path) -> pd.DataFrame:
    """
    Load a SPEED+ label file (raw ``train.json``/``test.json``, or one of the
    derived subset files) as columns ``filename, r1..r3, q1..q4``, with the
    quaternions canonicalised on the way out.
    """
    data = pd.read_json(path)
    missing = [
        c
        for c in (FILENAME_COLUMN, TRANSLATION_COLUMN, QUATERNION_COLUMN)
        if c not in data.columns
    ]
    if missing:
        raise ValueError(f"{path}: label file is missing column(s) {missing}")

    data = data.rename(columns={TRANSLATION_COLUMN: "r", QUATERNION_COLUMN: "q"})
    data[TRANSLATION_COLUMNS] = pd.DataFrame(data["r"].tolist(), index=data.index)
    data[QUATERNION_COLUMNS] = pd.DataFrame(data["q"].tolist(), index=data.index)
    data = canonicalise_quaternions(data)
    data = data.drop(columns=["r", "q"])

    return data[COLUMN_ORDER].reset_index(drop=True)


def translation_normalisation(train_labels: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Per-component mean and standard deviation of the translation targets.

    Targets are z-scored with these before the MSE term and predictions are
    un-scored before any error is computed, so they are saved into every
    checkpoint.
    """
    means = [float(train_labels[c].mean()) for c in TRANSLATION_COLUMNS]
    # ddof=0, matching numpy's default, which is what the published runs used.
    stds = [float(train_labels[c].std(ddof=0)) for c in TRANSLATION_COLUMNS]
    return means, stds


def assert_disjoint(**named_filenames: set[str]) -> None:
    """
    Fail loudly if any two named filename sets overlap.

    The conformal calibration split only means anything if it shares no image
    with training, with the validation split used to select checkpoints, or
    with the in-domain test split.
    """
    names = list(named_filenames)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = named_filenames[left] & named_filenames[right]
            if shared:
                raise AssertionError(
                    f"{left} and {right} share {len(shared)} filenames, "
                    f"e.g. {sorted(shared)[:3]}; the split is contaminated"
                )
