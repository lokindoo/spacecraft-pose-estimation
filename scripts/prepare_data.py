#!/usr/bin/env python
"""
Build the downscaled subsets from a raw SPEED+ download.

    python scripts/prepare_data.py --all

SPEED+ itself is not redistributed by this project. Download it from Zenodo
record 5588480, accept its licence, and point --data-root at the extracted
directory (the one containing synthetic/, lightbox/, sunlamp/ and camera.json).

What this produces, under <data-root>/subsets/:

  synthetic_28k_train_2k_val_2026-07-15/   28k train + 2k validation, 320x512
  synthetic_split_conformal_coverage/      1k calibration + 1k in-domain test
  lightbox_downscaled/, sunlamp_downscaled/   the real domains, in full

The two synthetic subsets are disjoint by construction, and the conformal one is
additionally drawn from images the training run never saw, which is what makes
it a "fresh synthetic" control. Both properties are asserted rather than
assumed.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from speedplus_pose.cli import add_path_arguments
from speedplus_pose.config import ImageConfig
from speedplus_pose.data import archive_images, downscale_directory
from speedplus_pose.experiment import SPLITS
from speedplus_pose.labels import assert_disjoint
from speedplus_pose.paths import (
    SUBSET_CONFORMAL,
    SUBSET_LIGHTBOX,
    SUBSET_SUNLAMP,
    SUBSET_SYNTHETIC,
    Paths,
)

TRAIN_SIZE = 28_000
VALIDATION_SIZE = 2_000
CALIBRATION_SIZE = 1_000
IN_DOMAIN_TEST_SIZE = 1_000


def _write(labels: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_json(path, orient="records")
    print(f"  wrote {len(labels):6d} labels -> {path.name}")


def _canonical_path(paths: Paths, split_name: str):
    split = SPLITS[split_name]
    return paths.subset(split.subset) / split.labels


def prepare_synthetic(paths: Paths, seed: int, image: ImageConfig) -> None:
    """
    Sample the training and validation splits.

    SPEED+ ships its own train/validation division of the synthetic set, and we
    sample within each rather than re-splitting across both, so the validation
    images were never candidates for training.
    """
    print(f"\n{SUBSET_SYNTHETIC}")
    train = pd.read_json(paths.raw_labels("synthetic", "train")).sample(
        n=TRAIN_SIZE, random_state=seed
    )
    validation = pd.read_json(paths.raw_labels("synthetic", "validation")).sample(
        n=VALIDATION_SIZE, random_state=seed
    )

    assert_disjoint(train=set(train["filename"]), validation=set(validation["filename"]))
    _write(train, _canonical_path(paths, "train"))
    _write(validation, _canonical_path(paths, "synthetic"))

    destination = paths.subset(SUBSET_SYNTHETIC) / "images"
    filenames = set(train["filename"]) | set(validation["filename"])
    downscale_directory(paths.raw_images("synthetic"), destination, filenames, image)


def prepare_conformal(paths: Paths, seed: int, image: ImageConfig) -> None:
    """
    Carve the conformal calibration and in-domain test splits out of the
    synthetic validation images the 2k validation split did not take.

    The coverage result depends on this step. A guarantee calibrated on images
    the model was fitted on, or selected on, is not a guarantee about anything,
    so the exclusion is explicit and then asserted.
    """
    print(f"\n{SUBSET_CONFORMAL}")
    available = pd.read_json(paths.raw_labels("synthetic", "validation"))
    already_used = set(pd.read_json(_canonical_path(paths, "synthetic"))["filename"])
    available = available[~available["filename"].isin(already_used)].copy()
    print(f"  {len(available)} synthetic validation images unused by model selection")

    sampled = available.sample(
        n=CALIBRATION_SIZE + IN_DOMAIN_TEST_SIZE, random_state=seed
    )
    calibration = sampled.iloc[:CALIBRATION_SIZE]
    in_domain_test = sampled.iloc[CALIBRATION_SIZE:]

    assert_disjoint(
        training=set(pd.read_json(_canonical_path(paths, "train"))["filename"]),
        selection=already_used,
        calibration=set(calibration["filename"]),
        in_domain_test=set(in_domain_test["filename"]),
    )
    _write(calibration, _canonical_path(paths, "calibration"))
    _write(in_domain_test, _canonical_path(paths, "in_domain_test"))

    destination = paths.subset(SUBSET_CONFORMAL) / "images"
    filenames = set(calibration["filename"]) | set(in_domain_test["filename"])
    downscale_directory(paths.raw_images("synthetic"), destination, filenames, image)


def prepare_real(paths: Paths, image: ImageConfig) -> None:
    """
    Downscale both real domains in full. No sampling and no split: these are
    test sets, used once each, and never for training, tuning or selection.
    """
    for domain, subset in (("lightbox", SUBSET_LIGHTBOX), ("sunlamp", SUBSET_SUNLAMP)):
        print(f"\n{subset}")
        labels = pd.read_json(paths.raw_labels(domain, "test"))
        _write(labels, _canonical_path(paths, domain))
        downscale_directory(
            paths.raw_images(domain),
            paths.subset(subset) / "images",
            set(labels["filename"]),
            image,
        )


def archive(paths: Paths) -> None:
    """
    Zip each prepared subset. Only needed for Colab, where reading tens of
    thousands of small files across a mounted Drive is unusably slow; one
    archive copies over in seconds and unpacks to local disk.
    """
    for subset in (SUBSET_SYNTHETIC, SUBSET_CONFORMAL, SUBSET_LIGHTBOX, SUBSET_SUNLAMP):
        directory = paths.subset(subset)
        archive_path = directory / "images.zip"
        if archive_path.exists():
            print(f"{archive_path.name} already exists in {subset}, skipping")
            continue
        archive_images(directory / "images", archive_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--synthetic", action="store_true", help="Build the training subsets")
    parser.add_argument("--conformal", action="store_true", help="Build the conformal splits")
    parser.add_argument("--real", action="store_true", help="Downscale lightbox and sunlamp")
    parser.add_argument("--archive", action="store_true", help="Zip the prepared subsets")
    parser.add_argument("--all", action="store_true", help="Everything, in order")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed. Changing it produces a different dataset, and the "
        "published numbers no longer apply.",
    )
    args = parser.parse_args(argv)

    if not any((args.synthetic, args.conformal, args.real, args.archive, args.all)):
        parser.error("nothing to do; pass --all or one of the stage flags")

    paths = Paths(args.data_root, args.artifacts_root or ".")
    image = ImageConfig()
    print(f"raw SPEED+ at {paths.data}")

    if args.all or args.synthetic:
        prepare_synthetic(paths, args.seed, image)
    if args.all or args.conformal:
        prepare_conformal(paths, args.seed, image)
    if args.all or args.real:
        prepare_real(paths, image)
    if args.all or args.archive:
        archive(paths)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
