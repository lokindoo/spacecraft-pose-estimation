import pytest
import torch.nn as nn
from torchvision.models import resnet50

from speedplus_pose.model import POOLED_FEATURES, POSE_OUTPUTS, SpacecraftPoseEstimator


@pytest.fixture(scope="module")
def resnet_model():
    """
    A SpacecraftPoseEstimator with randomly initialised backbone weights.

    Built without downloading the ImageNet checkpoint: these tests are about
    wiring, which parameters train and which BatchNorm layers are in train mode,
    and none of that depends on the weight values.
    """
    model = SpacecraftPoseEstimator.__new__(SpacecraftPoseEstimator)
    nn.Module.__init__(model)
    model.resnet = resnet50(weights=None)
    model.resnet.fc = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(POOLED_FEATURES, 1024),
        nn.ReLU(),
        nn.Linear(1024, POSE_OUTPUTS),
    )
    return model
