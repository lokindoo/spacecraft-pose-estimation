#!/usr/bin/env python
"""
Check by eye that the labels mean what you think they mean.

    python scripts/verify_labels.py --split synthetic --count 4

Projects each pose onto its image: a dot at the target's predicted image
position, and the three body axes drawn from it. If the dot sits on the
spacecraft and the axes look attached to it, the label convention, the camera
matrix and the quaternion-to-matrix conversion all agree.

Worth running before any training run and after any change to label loading.
A quaternion convention error (scalar-first versus scalar-last, or a transposed
rotation) produces a model that trains perfectly happily and is wrong, and this
is the cheapest place to catch it.
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from speedplus_pose.cli import add_path_arguments, paths_from_args
from speedplus_pose.experiment import SPLITS, load_split_labels
from speedplus_pose.paths import RAW_DOMAINS

AXIS_COLOURS = ("red", "green", "blue")  # body x, y, z


def project_point(point_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Pinhole projection: K @ p, divided through by depth. Returns (u, v)."""
    return (camera_matrix @ point_3d) / point_3d[2]


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    SPEED+ quaternions are scalar-FIRST (w, x, y, z).

    scipy is scalar-last by default, so the conversion is explicit. Getting this
    wrong yields a valid rotation matrix for a different rotation - which no
    assertion catches, but the overlay does.
    """
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def draw_pose_overlay(ax, image, r: np.ndarray, q: np.ndarray, camera_matrix: np.ndarray) -> None:
    ax.imshow(image, cmap="gray")

    origin = project_point(r, camera_matrix)
    ax.scatter(origin[0], origin[1], s=18, color="yellow", zorder=3)

    # Body axes expressed in camera coordinates, planted at the target.
    rotation = quaternion_to_matrix(q)
    axes_in_camera = np.eye(3) @ rotation.T

    for axis, colour in zip(r + axes_in_camera, AXIS_COLOURS):
        tip = project_point(axis, camera_matrix)
        ax.plot([origin[0], tip[0]], [origin[1], tip[1]], color=colour, linewidth=1.5)
    ax.set_axis_off()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--split", default="synthetic", choices=sorted(SPLITS), help="Split to check")
    parser.add_argument("--count", type=int, default=4, help="How many images to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Overlay on the full-resolution originals instead of the downscaled "
        "subset. The camera matrix belongs to the ORIGINAL resolution, so this is "
        "the honest check; the downscaled version is scaled to match.",
    )
    parser.add_argument("--output", default=None, help="Save instead of showing")
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    camera_matrix = np.array(json.loads(paths.camera().read_text(encoding="utf-8"))["cameraMatrix"])

    labels = load_split_labels(args.split, paths)
    rng = np.random.default_rng(args.seed)
    picks = labels.iloc[rng.choice(len(labels), min(args.count, len(labels)), replace=False)]

    domain = next((d for d in RAW_DOMAINS if d in SPLITS[args.split].subset), "synthetic")
    if args.raw:
        images_path = paths.raw_images(domain)
        scaled_matrix = camera_matrix
    else:
        images_path = paths.subset(SPLITS[args.split].subset) / "images"
        with Image.open(images_path / picks.iloc[0, 0]) as probe:
            width, height = probe.size
        with Image.open(paths.raw_images(domain) / picks.iloc[0, 0]) as original:
            original_width, original_height = original.size
        # Downscaling an image rescales its intrinsics by the same factor.
        scaled_matrix = camera_matrix.copy()
        scaled_matrix[0] *= width / original_width
        scaled_matrix[1] *= height / original_height

    figure, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 3.2))
    for ax, (_, row) in zip(np.atleast_1d(axes), picks.iterrows()):
        with Image.open(images_path / row.iloc[0]) as image:
            draw_pose_overlay(
                ax,
                np.array(image),
                row[["r1", "r2", "r3"]].to_numpy(dtype=float),
                row[["q1", "q2", "q3", "q4"]].to_numpy(dtype=float),
                scaled_matrix,
            )
        ax.set_title(row.iloc[0], fontsize=8)

    figure.suptitle(f"{args.split}: dot on the target, body axes attached to it")
    figure.tight_layout()

    if args.output:
        figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"saved {args.output}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
