"""
Label handling tests.

The quaternion canonicalisation is the one to watch: it is what lets the paper
say the double cover is removed rather than merely tolerated by the loss.
"""

import json

import numpy as np
import pandas as pd
import pytest

from speedplus_pose.labels import (
    COLUMN_ORDER,
    assert_disjoint,
    canonicalise_quaternions,
    load_labels,
    translation_normalisation,
)
from speedplus_pose.metrics import rotation_error


def _write_labels(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_load_labels_unpacks_and_orders_columns(tmp_path):
    """
    Column ORDER is part of the contract - the dataset reads the filename at
    position 0 and the seven pose values as columns 1:, so a reordering would
    feed the model quaternions as translations without raising anything.
    """
    path = _write_labels(
        tmp_path / "labels.json",
        [
            {
                "filename": "img000001.jpg",
                "q_vbs2tango_true": [0.5, 0.5, 0.5, 0.5],
                "r_Vo2To_vbs_true": [1.0, 2.0, 30.0],
            }
        ],
    )
    data = load_labels(path)
    assert list(data.columns) == COLUMN_ORDER
    assert data.loc[0, "r3"] == 30.0
    assert data.loc[0, "q1"] == 0.5


def test_load_labels_rejects_wrong_schema(tmp_path):
    path = _write_labels(tmp_path / "bad.json", [{"filename": "a.jpg", "pose": [1, 2, 3]}])
    with pytest.raises(ValueError, match="missing column"):
        load_labels(path)


def test_canonicalisation_flips_only_the_negative_hemisphere():
    data = pd.DataFrame(
        {
            "q1": [0.5, -0.5],
            "q2": [0.5, -0.5],
            "q3": [0.5, 0.5],
            "q4": [0.5, -0.5],
        }
    )
    canonicalise_quaternions(data)
    assert (data["q1"] >= 0).all()
    # The already-canonical row is untouched.
    assert data.loc[0].tolist() == [0.5, 0.5, 0.5, 0.5]
    # The flipped row negated every component, not just the first.
    assert data.loc[1].tolist() == [0.5, 0.5, -0.5, 0.5]


def test_canonicalisation_preserves_the_rotation():
    """
    Flipping the sign must not change the orientation the label encodes - if it
    did, canonicalisation would be corrupting the dataset rather than
    disambiguating it.
    """
    original = np.array([[-0.5, -0.5, 0.5, -0.5]])
    data = pd.DataFrame(original, columns=["q1", "q2", "q3", "q4"])
    canonicalise_quaternions(data)
    flipped = data[["q1", "q2", "q3", "q4"]].to_numpy()

    assert flipped[0, 0] >= 0
    assert np.isclose(rotation_error(flipped, original), 0.0)


def test_translation_normalisation_uses_population_std():
    """
    ddof=0, matching numpy's default - which is what the published runs used and
    what the constants stored in every checkpoint were computed with.
    """
    data = pd.DataFrame({"r1": [1.0, 3.0], "r2": [0.0, 0.0], "r3": [10.0, 20.0]})
    means, stds = translation_normalisation(data)
    assert means == [2.0, 0.0, 15.0]
    assert stds == [1.0, 0.0, 5.0]


def test_assert_disjoint_catches_contamination():
    """
    The conformal calibration split only means anything if it shares no image
    with training or with the split used to pick checkpoints.
    """
    assert_disjoint(train={"a.jpg"}, calibration={"b.jpg"}, test={"c.jpg"})

    with pytest.raises(AssertionError, match="contaminated"):
        assert_disjoint(train={"a.jpg", "b.jpg"}, calibration={"b.jpg"})
