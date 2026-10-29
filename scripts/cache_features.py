#!/usr/bin/env python
"""
Cache the frozen backbone's pooled features for every split a head experiment
needs.

    python scripts/cache_features.py --bn imagenet
    python scripts/cache_features.py --bn synthetic

--bn imagenet is the published frozen arm's backbone exactly: eval mode,
ImageNet BatchNorm statistics.

--bn synthetic first replaces every BatchNorm layer's statistics with those of
the un-augmented synthetic training set (weights untouched), then caches. A head
trained on those features asks whether the layer-4 arm's real-domain gain came
from adapting its weights or only its normalisation.

Writes <artifacts>/features/<bn>/<split>.npz, plus manifest.json and, for
--bn synthetic, the recomputed backbone state. Check the ImageNet cache with
scripts/check_cache.py before training anything on it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from speedplus_pose.batchnorm import recompute_bn_statistics
from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import ImageConfig, ModelConfig, TrainConfig
from speedplus_pose.data import build_dataloader
from speedplus_pose.experiment import build_dataset, load_split_labels
from speedplus_pose.features import BN_REGIMES, CACHE_SPLITS, extract_features, imagenet_fc, save_features
from speedplus_pose.model import build_model, resolve_device
from speedplus_pose.transforms import build_eval_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument("--bn", choices=BN_REGIMES, required=True)
    parser.add_argument("--splits", nargs="+", default=list(CACHE_SPLITS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="First N images per split (smoke test)")
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    device = resolve_device(args.device)
    output = paths.features(args.bn)
    loader_config = TrainConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    transform = build_eval_transform(ImageConfig())

    def dataloader(split: str):
        labels = load_split_labels(split, paths)
        if args.limit:
            labels = labels.head(args.limit)
        dataset = build_dataset(split, paths, transform, labels=labels, local_images=args.local_images)
        return build_dataloader(dataset, loader_config, shuffle=False)

    backbone = build_model(ModelConfig(), device).resnet
    backbone.eval()

    manifest = {
        "bn": args.bn,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "splits": {},
    }

    if args.bn == "synthetic":
        # The un-augmented training set: the frozen no-aug arm's own training
        # distribution, so the head below differs from the published frozen arm
        # in its normalisation statistics and nothing else.
        seen = recompute_bn_statistics(backbone, dataloader("train"), device, "BN <- synthetic train")
        output.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), output / "backbone_bn_synthetic.pt")
        manifest["bn_recomputed_on"] = {"split": "train", "images": seen}
        print(f"recomputed BatchNorm statistics on {seen} synthetic training images")

    for split in args.splits:
        record = extract_features(backbone, dataloader(split), device, split)
        save_features(output / f"{split}.npz", record)
        manifest["splits"][split] = int(len(record["filename"]))
        print(f"  {split:16} {record['features'].shape}  -> {output / f'{split}.npz'}")

    weight, bias = imagenet_fc()
    np.savez(output.parent / "imagenet_fc.npz", weight=weight, bias=bias)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
