"""
Sim-to-real spacecraft pose estimation on SPEED+.

Direct pose regression (ResNet-50 + MLP head) trained on synthetic imagery
alone, then evaluated unchanged on the lightbox and sunlamp
hardware-in-the-loop domains. The study is a 2x2 over backbone adaptation and
photometric augmentation, with a split-conformal check on whether the model's
stated confidence survives the shift.

See README.md for the write-up and configs/ for the four arms.
"""

from .config import ExperimentConfig, load_config, set_seed
from .metrics import (
    CHANCE_LEVEL,
    rotation_error,
    skill,
    speed_score,
    translation_error_metres,
    translation_error_relative,
)
from .model import SpacecraftPoseEstimator, build_model, set_train_mode
from .paths import Paths

__all__ = [
    "CHANCE_LEVEL",
    "ExperimentConfig",
    "Paths",
    "SpacecraftPoseEstimator",
    "build_model",
    "load_config",
    "rotation_error",
    "set_seed",
    "set_train_mode",
    "skill",
    "speed_score",
    "translation_error_metres",
    "translation_error_relative",
]

__version__ = "1.0.0"
