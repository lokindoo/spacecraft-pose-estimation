"""
The estimator: ImageNet-pretrained ResNet-50, global-average-pooled to 2048-d,
into a two-layer MLP head regressing seven numbers, translation (3) and a
quaternion (4).

Direct pose regression, chosen because it is simple to reproduce and leans on
coarse global shape. Keypoint-plus-PnP pipelines are far sharper in-domain but
much more fragile across the sim-to-real gap, which is why the modest relative
degradation reported here is not evidence of a small gap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import ModelConfig

POSE_OUTPUTS = 7  # 3 translation + 4 quaternion
POOLED_FEATURES = 2048


class SpacecraftPoseEstimator(nn.Module):
    """
    ResNet-50 with ``fc`` replaced by identity, so the head sees the pooled
    2048-d feature vector rather than the 1000-d ImageNet logits.

    The project's first baseline read the logits instead. Moving to pooled
    features, together with scaling the training set from 4k to 28k, is the
    improvement that transferred almost nothing to sunlamp.
    """

    def __init__(self, head_hidden: int = 1024):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(POOLED_FEATURES, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, POSE_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.resnet(x))


def apply_freeze_policy(model: SpacecraftPoseEstimator, config: ModelConfig) -> None:
    """
    Freeze the backbone, then re-enable the stages this arm trains.

    ``trainable_stages = []``          -> only the head trains (frozen arms)
    ``trainable_stages = ["layer4"]``  -> head + the final ResNet stage
    """
    for param in model.resnet.parameters():
        param.requires_grad = False

    for stage_name in config.trainable_stages:
        stage = getattr(model.resnet, stage_name, None)
        if stage is None:
            raise ValueError(f"ResNet-50 has no stage named {stage_name!r}")
        for param in stage.parameters():
            param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def set_train_mode(model: SpacecraftPoseEstimator, config: ModelConfig) -> None:
    """
    Put the model into the training mode that defines this arm. It is half of
    what separates the two rows of the 2x2.

    A frozen arm keeps the whole backbone in eval mode, so its BatchNorm layers
    go on using the ImageNet running statistics. An unfrozen arm additionally
    puts layer 4 in train mode, so that stage's BatchNorm statistics track the
    training distribution, and with augmentation on, the augmented one, which
    Figure 1 shows fits neither real domain well and lightbox less badly. That
    is why augmentation moves lightbox and leaves sunlamp inside seed noise.

    Calling ``model.train()`` alone would quietly turn every frozen arm into a
    partially adapted one and erase the comparison.
    """
    model.train()
    model.resnet.eval()
    for stage_name in config.trainable_stages:
        getattr(model.resnet, stage_name).train()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total, trainable), printed at the start of a run as a sanity check."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(
    config: ModelConfig, device: torch.device | str | None = None
) -> SpacecraftPoseEstimator:
    """Construct, apply the freeze policy, and move to device."""
    model = SpacecraftPoseEstimator(head_hidden=config.head_hidden)
    apply_freeze_policy(model, config)
    if device is not None:
        model.to(device)
    return model


def resolve_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
