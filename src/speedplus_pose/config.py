"""
Experiment configuration.

The 2x2 study is four arms x three seeds, and in the original notebooks that was
eleven near-identical files differing by a seed literal and a checkpoint
directory name. Here it is four YAML files and a ``--seed`` flag.

A config file may set ``extends: base.yaml`` to inherit from another config in
the same directory; keys merge one level deep.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class ImageConfig:
    height: int = 320
    width: int = 512

    @property
    def size(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class ModelConfig:
    head_hidden: int = 1024
    # Backbone stages left trainable. Empty means a fully frozen backbone (only
    # the head trains); ["layer4"] is the "unfrozen" arm of the 2x2.
    trainable_stages: tuple[str, ...] = ()

    @property
    def backbone_frozen(self) -> bool:
        return not self.trainable_stages


@dataclass(frozen=True)
class AugmentationConfig:
    """
    The photometric stack. Defaults are the values the published arms trained
    with; see transforms.py for what each one does and article.tex section 3.2
    for what it turned out to model.
    """

    enabled: bool = False
    brightness: tuple[float, float] = (0.65, 1.5)
    contrast: float = 0.5
    gamma: tuple[float, float] = (0.5, 2.0)
    glare_sigma: tuple[float, float] = (10.0, 50.0)
    glare_amplitude: tuple[float, float] = (0.8, 2.0)
    blur_kernel: int = 7
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    noise_sigma: tuple[float, float] = (0.01, 0.08)
    stochastic_p: float = 0.5
    erasing_scale: tuple[float, float] = (0.02, 0.06)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    # Weight on the rotation term of the split loss (LAMBDA in the notebooks).
    lambda_rotation: float = 1.0
    # Checkpoints were saved on even epochs; checkpoint selection then picks the
    # best synthetic val rotation error among them.
    checkpoint_every: int = 2
    num_workers: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_name(self) -> str:
        """Directory name for this arm+seed's checkpoints."""
        return f"{self.name}_seed{self.seed}"

    @property
    def arm(self) -> str:
        """
        Key used in results tables and checkpoints.json, e.g.
        ``unfrozen_aug_seed42``. Same strings as project_context/results_table.md.
        """
        return f"{self.name}_seed{self.seed}"


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _build(section: type, values: dict[str, Any]) -> Any:
    return section(**{k: _as_tuple(v) for k, v in values.items()})


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return raw
    parent = _load_raw(path.parent / parent_name, _seen | {path})
    return _merge(parent, raw)


def load_config(path: str | Path, seed: int | None = None) -> ExperimentConfig:
    """
    Read a config file. ``path`` may be a bare name ("unfrozen_aug"), a filename,
    or a full path. ``seed`` overrides whatever the file says, which is how the
    three-seed repeats are run.
    """
    candidate = Path(path)
    if not candidate.exists():
        stem = candidate.name if candidate.suffix else f"{candidate.name}.yaml"
        candidate = CONFIG_DIR / stem
    if not candidate.exists():
        available = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"no config at {path!r}; available: {available}")

    raw = _load_raw(candidate)
    config = ExperimentConfig(
        name=raw.get("name", candidate.stem),
        seed=raw.get("seed", 42),
        image=_build(ImageConfig, raw.get("image", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        augmentation=_build(AugmentationConfig, raw.get("augmentation", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )
    return replace(config, seed=seed) if seed is not None else config


def set_seed(seed: int) -> None:
    """
    Seed torch, numpy and random.

    NOTE, and this is stated in the README too: the published runs called
    ``torch.manual_seed(seed)`` and nothing else. numpy, ``random``, and the
    DataLoader workers were left unseeded. Seeding all of them here is the right
    thing to do going forward, but it means re-training will NOT bit-reproduce
    the released checkpoints. Reproduce the tables from the checkpoints
    (scripts/reproduce_table.py), not by re-training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
