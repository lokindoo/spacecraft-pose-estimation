"""
Argument-parsing helpers shared by the scripts in scripts/.

Every script takes the same two roots so none of them has to know whether it is
running on Colab, a local machine, or a rented GPU box.
"""

from __future__ import annotations

import argparse

from .paths import Paths


def add_path_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root holding the SPEED+ download and subsets/ "
        "(default: $SPEEDPLUS_DATA_ROOT, then $DATASET_PATH)",
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Where checkpoints, predictions and figures go "
        "(default: $SPEEDPLUS_ARTIFACTS)",
    )
    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        required=True,
        help="Arm config: a name (unfrozen_aug), a filename, or a path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config's seed. The study used 42, 43 and 44.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    return parser


def paths_from_args(args: argparse.Namespace) -> Paths:
    return Paths(args.data_root, args.artifacts_root)


def add_local_images_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--local-images",
        default=None,
        help="Fast local directory to unpack image archives into. On Colab use "
        "something under /content, never the mounted Drive.",
    )
    return parser
