"""
The revised Figure 1 statistics have one job: separate a soft bright region
from a hard edge, which the pooled intensity density cannot. These tests build
exactly that pair and check the statistics tell them apart.
"""

import numpy as np

from speedplus_pose.image_stats import (
    EDGE_THRESHOLD,
    accumulate,
    empty_accumulator,
    gradient_magnitude,
    image_statistics,
    summarise_pool,
    total_variation,
)

H, W = 64, 96


def _hard_step():
    plane = np.zeros((H, W))
    plane[:, W // 2 :] = 1.0
    return plane


def _soft_blob():
    ys, xs = np.mgrid[:H, :W]
    return np.exp(-((xs - W / 2) ** 2 + (ys - H / 2) ** 2) / (2 * 20.0**2))


def test_a_unit_step_reads_one():
    assert np.isclose(gradient_magnitude(_hard_step()).max(), 1.0)


def test_hard_edges_and_soft_blobs_separate_on_gradients():
    step, blob = image_statistics(_hard_step()), image_statistics(_soft_blob())
    # The step has edge pixels; the wide blob has none above the threshold.
    assert step["edge_pixels"] > 0 and blob["edge_pixels"] == 0
    assert total_variation(step["gradient_counts"], blob["gradient_counts"]) > 0.5


def test_clipped_pixels_count_the_ceiling_only():
    plane = np.full((H, W), 0.5)
    plane[:4, :4] = 1.0
    assert image_statistics(plane)["clipped_pixels"] == 16


def test_pooling_weights_by_pixel():
    total = empty_accumulator()
    for plane in (_hard_step(), np.zeros((H, W))):
        accumulate(total, image_statistics(plane))
    pooled = summarise_pool(total)
    assert pooled["images"] == 2
    single = summarise_pool(accumulate(empty_accumulator(), image_statistics(_hard_step())))
    assert np.isclose(pooled["edge_density"], single["edge_density"] / 2)
    assert np.isclose(pooled["clipped_fraction"], 0.25)  # half of one image of two


def test_total_variation_bounds():
    p = np.array([1.0, 0.0, 0.0])
    assert total_variation(p, p) == 0.0
    assert total_variation(p, np.array([0.0, 0.0, 5.0])) == 1.0


def test_threshold_is_a_fraction_of_a_full_step():
    assert 0 < EDGE_THRESHOLD < 1
