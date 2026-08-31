#!/usr/bin/env python
"""
Regenerate the paper's figures.

    python scripts/make_figures.py --all

  --rotation-error   Figure 1: rotation error per arm and domain, against the
                     chance level. Reads results/results_2x2.csv, so it needs
                     no data and no GPU.
  --crops            Figure 2, as first published: two bare grids, augmented
                     synthetic beside real. Needs the prepared image subsets.
  --crops-triptych   Figure 2, rebuilt: three labelled columns with an
                     intensity-histogram strip underneath. This is the version
                     the paper should carry; --crops is kept because the
                     published PNGs came from it.
  --reliability      Figure 3: conformal reliability curves. Needs the
                     prediction records from scripts/evaluate.py.

Figure 2 is the one that carries the argument: the augmented distribution looks
like lightbox and nothing like sunlamp, which is why augmentation moves one and
not the other. It is worth regenerating whenever the augmentation changes.

For the paper's two figure jobs:

    python scripts/make_figures.py --crops-triptych --seed 7
    python scripts/make_figures.py --reliability --arms unfrozen_no_aug_seed42 \
        --reliability-metrics rotation --font-scale 1.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import load_config
from speedplus_pose.coverage import scores_from_record, sweep_reliability
from speedplus_pose.evaluate import load_predictions
from speedplus_pose.experiment import build_dataset
from speedplus_pose.figures import (
    crops_with_histograms,
    image_grid,
    reliability,
    rotation_error_by_arm,
)
from speedplus_pose.transforms import build_train_transform

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ARM_LABELS = {
    "frozen_no_aug": "frozen",
    "frozen_aug": "frozen+aug",
    "unfrozen_no_aug": "layer 4",
    "unfrozen_aug": "layer 4+aug",
}


def figure_rotation_error(output_dir: Path) -> None:
    results = pd.read_csv(RESULTS_DIR / "results_2x2.csv")
    arms = list(ARM_LABELS)

    errors, stds = {}, {}
    for domain in ("synthetic", "lightbox", "sunlamp"):
        subset = results[results["domain"] == domain]
        grouped = subset.groupby("arm")["rotation"]
        errors[domain] = {arm: grouped.mean()[arm] for arm in arms}
        # ddof=1 over seeds; single-seed arms get 0, which draws no error bar.
        stds[domain] = {arm: (grouped.std(ddof=1).get(arm, 0.0) or 0.0) for arm in arms}

    rotation_error_by_arm(
        [ARM_LABELS[a] for a in arms],
        {d: {ARM_LABELS[a]: v[a] for a in arms} for d, v in errors.items()},
        {d: {ARM_LABELS[a]: v[a] for a in arms} for d, v in stds.items()},
        output_dir / "rotation_error_by_arm.png",
    )


def figure_crops(paths, output_dir: Path, samples: int, seed: int, local_images) -> None:
    """
    Two grids: augmented synthetic training samples, and real test samples.

    The augmented grid uses preview=True so it renders - the stack is otherwise
    identical to what the augmented arms trained on, normalisation aside.
    """
    config = load_config("unfrozen_aug")
    rng = np.random.default_rng(seed)

    augmented = build_dataset(
        "train",
        paths,
        build_train_transform(config.augmentation, config.image, preview=True),
        local_images=local_images,
    )
    indices = rng.choice(len(augmented), samples, replace=False)
    image_grid(
        [augmented[int(i)][0].numpy() for i in indices],
        rows=2,
        columns=samples // 2,
        path=output_dir / "train_transforms_2rows.png",
    )

    # Real samples go through the eval transform without normalisation, so the
    # comparison is like for like: same resize, same grayscale, no augmentation.
    real_images = []
    for domain in ("sunlamp", "lightbox"):
        dataset = build_dataset(
            domain,
            paths,
            build_train_transform(
                config.augmentation.__class__(enabled=False), config.image, preview=True
            ),
            local_images=local_images,
        )
        picks = rng.choice(len(dataset), samples // 2, replace=False)
        real_images.extend(dataset[int(i)][0].numpy() for i in picks)

    image_grid(
        real_images,
        rows=2,
        columns=samples // 2,
        path=output_dir / "real_2rows.png",
    )
    print("  real_2rows.png: sunlamp on the top row, lightbox on the bottom")


def _pick(dataset, filenames: list[str] | None, rng, count: int) -> list[int]:
    """
    Indices of the crops to draw: the named ones if given, otherwise a random
    sample.

    Naming them is the point for the paper's figure. Which sunlamp frames go in
    is an editorial choice - you want ones showing saturation and hard shadow
    boundaries, not merely bright ones - and a random draw will not reliably
    make that argument.
    """
    if not filenames:
        return [int(i) for i in rng.choice(len(dataset), count, replace=False)]

    available = list(dataset.filenames)
    lookup = {name: i for i, name in enumerate(available)}
    missing = [name for name in filenames if name not in lookup]
    if missing:
        raise SystemExit(f"not in this split: {missing} (e.g. {available[:3]})")
    return [lookup[name] for name in filenames]


def figure_crops_triptych(paths, output_dir: Path, chosen: dict, samples: int, seed: int, local_images) -> None:
    """
    Figure 2, rebuilt: three labelled columns with an intensity-histogram strip
    underneath, replacing the two bare grids the paper currently pastes side by
    side.

    Columns are (augmented synthetic | lightbox | sunlamp) so the two real
    domains sit next to each other and the eye compares them directly, and the
    augmented column's pixel distribution is overlaid on both histograms - which
    is what makes "the augmented distribution sits near lightbox, not sunlamp" a
    measurement rather than a request to trust the crops.
    """
    config = load_config("unfrozen_aug")
    rng = np.random.default_rng(seed)
    columns = {}

    augmented = build_dataset(
        "train",
        paths,
        build_train_transform(config.augmentation, config.image, preview=True),
        local_images=local_images,
    )
    columns["augmented synthetic"] = [
        augmented[i][0].numpy() for i in _pick(augmented, chosen.get("synthetic"), rng, samples)
    ]

    # Real crops go through the same resize and grayscale with no augmentation,
    # so the histograms compare like with like.
    plain = build_train_transform(
        config.augmentation.__class__(enabled=False), config.image, preview=True
    )
    for domain in ("lightbox", "sunlamp"):
        dataset = build_dataset(domain, paths, plain, local_images=local_images)
        columns[domain] = [
            dataset[i][0].numpy() for i in _pick(dataset, chosen.get(domain), rng, samples)
        ]

    crops_with_histograms(
        columns,
        path=output_dir / "crops_triptych.png",
        reference="augmented synthetic",
    )
    print(
        "  crops_triptych.png: pick the crops deliberately with --crop-sunlamp / "
        "--crop-lightbox /\n  --crop-synthetic once you have seen a draw you like "
        "- sunlamp wants saturation and hard\n  shadow boundaries, not merely "
        "bright frames."
    )


def figure_reliability(
    paths, output_dir: Path, arms: list[str], metrics: tuple[str, ...], font_scale: float
) -> None:
    alphas = np.round(np.arange(0.01, 0.51, 0.01), 2)
    nominal = 1 - alphas
    plot_domains = ("in_domain_test", "sunlamp", "lightbox")

    for arm in arms:
        records = load_predictions(paths.predictions / f"{arm}.pkl")
        scores = {split: scores_from_record(records[split]) for split in ("calibration", *plot_domains)}
        curves = {
            metric: sweep_reliability(
                scores["calibration"],
                {domain: scores[domain] for domain in plot_domains},
                metric,
                alphas,
            )
            for metric in metrics
        }
        suffix = "_rotation" if metrics == ("rotation",) else ""
        reliability(
            nominal, curves, arm, metrics=metrics, font_scale=font_scale,
            path=output_dir / f"reliability_{arm}{suffix}.png",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--rotation-error", action="store_true")
    parser.add_argument("--crops", action="store_true")
    parser.add_argument("--crops-triptych", action="store_true")
    parser.add_argument("--reliability", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default=None, help="Figure directory")
    parser.add_argument("--samples", type=int, default=8, help="Crops per grid (even)")
    parser.add_argument(
        "--triptych-rows", type=int, default=3, help="Crops per column of the triptych"
    )
    parser.add_argument("--seed", type=int, default=0, help="Which crops get sampled")
    for domain in ("synthetic", "lightbox", "sunlamp"):
        parser.add_argument(
            f"--crop-{domain}",
            nargs="+",
            default=None,
            metavar="FILENAME",
            help=f"Exact {domain} filenames for the triptych, in order "
            "(default: a random draw from --seed)",
        )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unfrozen_no_aug_seed42", "unfrozen_aug_seed42"],
        help="Arms to draw reliability curves for",
    )
    parser.add_argument(
        "--reliability-metrics",
        nargs="+",
        default=["translation_metres", "rotation"],
        help="Panels to draw. The paper's Figure 3 is 'rotation' alone - one "
        "panel at column width, legible; two are not.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier on reliability-plot label sizes; use ~1.4 for the "
        "single-panel version that goes in the paper",
    )
    args = parser.parse_args(argv)

    if not any(
        (args.rotation_error, args.crops, args.crops_triptych, args.reliability, args.all)
    ):
        parser.error("nothing to do - pass --all or one of the figure flags")

    # Figure 1 needs no dataset, so allow it to run without a data root.
    needs_data = args.all or args.crops or args.crops_triptych or args.reliability
    paths = paths_from_args(args) if needs_data else None
    output_dir = Path(args.output) if args.output else (paths.figures if paths else Path("figures"))

    if args.all or args.rotation_error:
        figure_rotation_error(output_dir)
    if args.all or args.crops:
        figure_crops(paths, output_dir, args.samples, args.seed, args.local_images)
    if args.all or args.crops_triptych:
        figure_crops_triptych(
            paths,
            output_dir,
            {
                "synthetic": args.crop_synthetic,
                "lightbox": args.crop_lightbox,
                "sunlamp": args.crop_sunlamp,
            },
            args.triptych_rows,
            args.seed,
            args.local_images,
        )
    if args.all or args.reliability:
        figure_reliability(
            paths, output_dir, args.arms, tuple(args.reliability_metrics), args.font_scale
        )

    print(
        "\nNote: the published Figure 3 had its title cropped out for the "
        "anonymised build (the title names the seed). figures.reliability keeps "
        "it off by default for the same reason."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
