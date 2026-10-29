#!/usr/bin/env python
"""
Pooled image statistics for the revised Figure 1: what each augmentation stack
actually produces, set against what each real domain looks like.

    python scripts/image_statistics.py

For the synthetic training set under each stack (the published full stack,
arm A, arm B, and no augmentation) and for both real domains un-augmented, it
pools over every image:

  - the pixel-intensity density (Figure 1's original statistic),
  - the Sobel gradient-magnitude density,
  - edge density and clipped-pixel fraction,

and reports the total variation of each synthetic pool from each real domain.
The intensity TV of the full stack is the published Figure 1 number (0.63
lightbox, 0.80 sunlamp), so that row is also a reproduction check.

Images go through the preview transform, exactly as Figure 1's crops did:
resize, grayscale and the stack, without Normalize. Augmentation draws come
from torch's RNG, seeded with --seed in the main process; workers derive their
seeds from it, so the result depends on --num-workers as well and both are
recorded.

Writes <artifacts>/image_stats/pools.npz and <artifacts>/results/image_statistics.csv.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import AugmentationConfig, ImageConfig, load_config
from speedplus_pose.experiment import build_dataset, load_split_labels
from speedplus_pose.image_stats import (
    EDGE_THRESHOLD,
    accumulate,
    empty_accumulator,
    image_statistics,
    summarise_pool,
    total_variation,
)
from speedplus_pose.transforms import build_train_transform

STACKS = {
    "synthetic, full stack": "unfrozen_aug",
    "synthetic, arm A (no blur/noise)": "unfrozen_aug_no_blur_noise",
    "synthetic, arm B (jitter + gamma + erasing)": "unfrozen_aug_jitter_gamma",
    "synthetic, no augmentation": None,
}


class StatisticsDataset(Dataset):
    """Wraps a split so workers compute the per-image statistics in parallel."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image_statistics(image[0].numpy())


def _keep(batch):
    return batch


def pool(dataset, num_workers: int, description: str) -> dict:
    loader = DataLoader(StatisticsDataset(dataset), batch_size=32, num_workers=num_workers, collate_fn=_keep)
    total = empty_accumulator()
    for batch in loader:
        for item in batch:
            accumulate(total, item)
    print(f"  {description}: {total['images']} images")
    return summarise_pool(total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--seed", type=int, default=7, help="Figure 1 was published with seed 7")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N images per split (smoke test)")
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    image = ImageConfig()

    def split(name: str, augmentation: AugmentationConfig):
        labels = load_split_labels(name, paths)
        if args.limit:
            labels = labels.head(args.limit)
        transform = build_train_transform(augmentation, image, preview=True)
        return build_dataset(name, paths, transform, labels=labels, local_images=args.local_images)

    pools = {}
    for domain in ("lightbox", "sunlamp"):
        pools[domain] = pool(split(domain, AugmentationConfig(enabled=False)), args.num_workers, domain)
    for label, config_name in STACKS.items():
        torch.manual_seed(args.seed)
        augmentation = load_config(config_name).augmentation if config_name else AugmentationConfig(enabled=False)
        pools[label] = pool(split("train", augmentation), args.num_workers, label)

    rows = []
    for label, stats in pools.items():
        row = {
            "pool": label,
            "images": stats["images"],
            "edge_density": stats["edge_density"],
            "clipped_fraction": stats["clipped_fraction"],
            "mean_gradient": stats["mean_gradient"],
        }
        for domain in ("lightbox", "sunlamp"):
            row[f"tv_intensity_{domain}"] = total_variation(stats["intensity_mass"], pools[domain]["intensity_mass"])
            row[f"tv_gradient_{domain}"] = total_variation(stats["gradient_mass"], pools[domain]["gradient_mass"])
        rows.append(row)

    table = pd.DataFrame(rows)
    output = paths.results / "image_statistics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    out_dir = paths.artifacts / "image_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "pools.npz",
        labels=np.array(list(pools)),
        intensity_mass=np.stack([p["intensity_mass"] for p in pools.values()]),
        gradient_mass=np.stack([p["gradient_mass"] for p in pools.values()]),
    )
    (out_dir / "settings.json").write_text(
        json.dumps({"seed": args.seed, "num_workers": args.num_workers, "edge_threshold": EDGE_THRESHOLD,
                    "limit": args.limit}, indent=2),
        encoding="utf-8",
    )

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n" + table.to_string(index=False))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
