#!/usr/bin/env python
"""
Train one arm of the 2x2 study.

    python scripts/train.py --config unfrozen_aug --seed 42

The four configs times three seeds are the whole experiment. Checkpoints land in
<artifacts>/model_checkpoints/<arm>_seed<n>/, and the run resumes from the
latest one automatically, which matters on Colab where a session can disappear.

Add --smoke to run the whole pipeline on a handful of images in a couple of
minutes on CPU. It checks the wiring, not the science.
"""

from __future__ import annotations

import argparse
import sys

from speedplus_pose.cli import (
    add_config_arguments,
    add_local_images_argument,
    add_path_arguments,
    paths_from_args,
)
from speedplus_pose.config import load_config, set_seed
from speedplus_pose.data import build_dataloader
from speedplus_pose.experiment import build_dataset, describe, load_split_labels
from speedplus_pose.labels import translation_normalisation
from speedplus_pose.model import build_model, resolve_device
from speedplus_pose.normalisation import TranslationNormaliser
from speedplus_pose.train import selected_checkpoint, train
from speedplus_pose.transforms import build_eval_transform, build_train_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    add_path_arguments(parser)
    add_local_images_argument(parser)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start from scratch",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny subset, 2 epochs, batch 4: a wiring check, not a run",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Keep only the latest and the best-so-far checkpoints on disk",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the config's epoch count (for a short smoke run on real data)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.seed)
    paths = paths_from_args(args)
    device = resolve_device(args.device)

    # Seeded before the model is built, so initialisation is covered too.
    set_seed(config.seed)

    train_labels = load_split_labels("train", paths)
    validation_labels = load_split_labels("synthetic", paths)

    from dataclasses import replace

    if args.epochs is not None:
        config = replace(config, train=replace(config.train, epochs=args.epochs))

    if args.smoke:
        config = replace(
            config,
            train=replace(config.train, epochs=2, batch_size=4, num_workers=0, checkpoint_every=2),
        )
        train_labels = train_labels.head(32)
        validation_labels = validation_labels.head(16)

    print(f"Training {describe(config)} on {device}")
    print(f"  {len(train_labels)} training / {len(validation_labels)} validation images")

    # Normalisation constants come from the training split alone; computing them
    # over train+validation would leak the validation distribution into the
    # model.
    means, stds = translation_normalisation(train_labels)
    normaliser = TranslationNormaliser(means, stds, device)

    train_dataset = build_dataset(
        "train",
        paths,
        build_train_transform(config.augmentation, config.image),
        labels=train_labels,
        local_images=args.local_images,
    )
    validation_dataset = build_dataset(
        "synthetic",
        paths,
        build_eval_transform(config.image),
        labels=validation_labels,
        local_images=args.local_images,
    )

    history = train(
        model=build_model(config.model, device),
        config=config,
        train_dataloader=build_dataloader(train_dataset, config.train, shuffle=True),
        validation_dataloader=build_dataloader(validation_dataset, config.train, shuffle=False),
        normaliser=normaliser,
        checkpoint_dir=paths.checkpoints(config.run_name),
        device=device,
        resume=not args.no_resume,
        prune=args.prune,
    )

    # Only saved epochs can be selected, so read the choice from the ledger
    # rather than from the full history, whose best epoch may be an odd one.
    selected = selected_checkpoint(paths.checkpoints(config.run_name))
    print(
        f"\nSelected checkpoint: epoch {selected['epoch'] + 1}, synthetic val_r "
        f"{selected['val_r']:.4f} rad (val_t {selected['val_t']:.4f}) -> {selected['file']}"
    )
    print("No real-domain number informed this choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
