#!/usr/bin/env python
"""
Train and evaluate one arm over several seeds, unattended and resumable.

    python scripts/run_seeds.py --config unfrozen_aug_no_blur_noise --seeds 42 43 44

Per seed, in order:

  1. train (scripts/train.py --prune), resuming if a previous session died
     part-way; skipped if the run already completed every epoch;
  2. select the checkpoint with the best synthetic validation rotation error,
     from the run's saved_checkpoints.json ledger;
  3. evaluate it on every split the tables and the conformal analysis need;
  4. append the per-seed summary to <artifacts>/results/<config>.csv.

Re-running the same command after an interruption picks up where it stopped, so
on Colab a reconnect costs at most the epochs since the last checkpoint. On a
Mac the chain holds a caffeinate assertion for as long as it runs, so the
machine does not sleep under it (it still needs mains power and an open lid or
an external display).

Each step runs as its own process, so memory is returned between seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from speedplus_pose.cli import add_local_images_argument, add_path_arguments, paths_from_args
from speedplus_pose.config import load_config
from speedplus_pose.evaluate import load_predictions, summarise
from speedplus_pose.train import read_ledger, selected_checkpoint

SCRIPTS = Path(__file__).resolve().parent
EVAL_DOMAINS = ("synthetic", "lightbox", "sunlamp", "calibration", "in_domain_test")


def _passthrough(args) -> list[str]:
    flags = []
    for name in ("data_root", "artifacts_root", "local_images", "device"):
        value = getattr(args, name)
        if value:
            flags += [f"--{name.replace('_', '-')}", str(value)]
    return flags


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _keep_awake() -> subprocess.Popen | None:
    if sys.platform != "darwin":
        return None
    # -i: no idle sleep; -s: no system sleep on mains power; -w: until we exit.
    return subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override, for a smoke run")
    add_path_arguments(parser)
    add_local_images_argument(parser)
    args = parser.parse_args(argv)

    paths = paths_from_args(args)
    awake = _keep_awake()
    python = sys.executable
    rows = []

    for seed in args.seeds:
        config = load_config(args.config, seed)
        epochs = args.epochs or config.train.epochs
        run_dir = paths.checkpoints(config.run_name)

        ledger = read_ledger(run_dir)
        if ledger and ledger[-1]["epoch"] + 1 >= epochs:
            print(f"\n{config.arm}: training already complete ({epochs} epochs)")
        else:
            command = [python, str(SCRIPTS / "train.py"), "--config", args.config, "--seed", str(seed), "--prune"]
            if args.epochs:
                command += ["--epochs", str(args.epochs)]
            _run(command + _passthrough(args))

        selected = selected_checkpoint(run_dir)
        predictions = paths.predictions / f"{config.arm}.pkl"
        previous = run_dir / "selected.json"
        if previous.exists() and json.loads(previous.read_text(encoding="utf-8"))["file"] != selected["file"]:
            # The run was extended and the selection moved: the old predictions
            # belong to a checkpoint that is no longer the model.
            predictions.unlink(missing_ok=True)
        previous.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"{config.arm}: selected epoch {selected['epoch'] + 1} (val_r {selected['val_r']:.4f}) -> {selected['file']}")

        if not predictions.exists():
            _run(
                [python, str(SCRIPTS / "evaluate.py"), "--config", args.config, "--seed", str(seed),
                 "--checkpoint", str(run_dir / selected["file"]), "--domains", *EVAL_DOMAINS,
                 "--output", str(predictions)]
                + _passthrough(args)
            )

        records = load_predictions(predictions)
        for domain in ("synthetic", "lightbox", "sunlamp"):
            rows.append(
                {"arm": args.config, "seed": seed, "domain": domain, "selected_epoch": selected["epoch"] + 1}
                | summarise(records[domain])
            )

    output = paths.results / f"{args.config}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output.exists():
        # A run over some seeds replaces those seeds' rows and keeps the rest.
        previous = pd.read_csv(output)
        frame = pd.concat([previous[~previous.seed.isin(args.seeds)], frame], ignore_index=True)
    frame.sort_values(["seed", "domain"]).to_csv(output, index=False)
    print(f"\nwrote {output}")
    if awake is not None:
        awake.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
