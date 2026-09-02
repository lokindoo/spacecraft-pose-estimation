"""
Tests for what actually distinguishes the four arms of the 2x2.

Two things define an arm: which parameters carry gradients, and which BatchNorm
layers are in train mode. The second is easy to lose in a refactor and would not
fail loudly. It would quietly convert every frozen arm into a partially adapted
one and erase the comparison the study rests on.
"""

import torch
from PIL import Image

from speedplus_pose.config import AugmentationConfig, ImageConfig, load_config
from speedplus_pose.model import (
    apply_freeze_policy,
    count_parameters,
    set_train_mode,
)
from speedplus_pose.transforms import build_eval_transform, build_train_transform

FROZEN = load_config("frozen_no_aug").model
UNFROZEN = load_config("unfrozen_no_aug").model

ARM_NAMES = ("frozen_no_aug", "frozen_aug", "unfrozen_no_aug", "unfrozen_aug")


def _grey_image(height=600, width=960):
    return Image.fromarray((torch.rand(height, width) * 255).byte().numpy(), mode="L")


def test_configs_define_the_2x2():
    arms = {name: load_config(name) for name in ARM_NAMES}

    assert arms["frozen_no_aug"].model.backbone_frozen
    assert arms["frozen_aug"].model.backbone_frozen
    assert arms["unfrozen_no_aug"].model.trainable_stages == ("layer4",)
    assert arms["unfrozen_aug"].model.trainable_stages == ("layer4",)

    assert not arms["frozen_no_aug"].augmentation.enabled
    assert arms["frozen_aug"].augmentation.enabled
    assert not arms["unfrozen_no_aug"].augmentation.enabled
    assert arms["unfrozen_aug"].augmentation.enabled

    # Every arm shares one training configuration; only the two factors vary.
    assert len({arm.train for arm in arms.values()}) == 1


def test_seed_override_names_the_run():
    config = load_config("unfrozen_aug", seed=44)
    assert config.seed == 44
    assert config.arm == "unfrozen_aug_seed44"


def test_frozen_arm_trains_only_the_head(resnet_model):
    apply_freeze_policy(resnet_model, FROZEN)
    total, trainable = count_parameters(resnet_model)

    head = sum(p.numel() for p in resnet_model.fc.parameters())
    assert trainable == head
    assert trainable < total


def test_unfrozen_arm_adds_layer4(resnet_model):
    apply_freeze_policy(resnet_model, UNFROZEN)
    _, trainable = count_parameters(resnet_model)

    head = sum(p.numel() for p in resnet_model.fc.parameters())
    layer4 = sum(p.numel() for p in resnet_model.resnet.layer4.parameters())
    assert trainable == head + layer4

    # Nothing earlier than layer 4 is trainable.
    assert not any(p.requires_grad for p in resnet_model.resnet.layer1.parameters())


def test_frozen_arm_keeps_every_backbone_batchnorm_in_eval(resnet_model):
    """
    A frozen arm must go on using the ImageNet running statistics everywhere in
    the backbone. If any of them switch to train mode, the arm is adapting to
    the training distribution and is no longer the control.
    """
    set_train_mode(resnet_model, FROZEN)

    assert resnet_model.fc.training
    backbone_bn = [
        m for m in resnet_model.resnet.modules() if isinstance(m, torch.nn.BatchNorm2d)
    ]
    assert backbone_bn and not any(m.training for m in backbone_bn)


def test_unfrozen_arm_puts_exactly_layer4_batchnorm_in_train(resnet_model):
    """
    The mechanism behind the second finding: layer 4's BatchNorm statistics
    track the (augmented) training distribution, which is why augmentation moves
    lightbox and leaves sunlamp inside seed noise.
    """
    set_train_mode(resnet_model, UNFROZEN)

    layer4_bn = [
        m
        for m in resnet_model.resnet.layer4.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]
    earlier_bn = [
        m
        for name, module in resnet_model.resnet.named_children()
        if name in ("bn1", "layer1", "layer2", "layer3")
        for m in module.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]

    assert layer4_bn and all(m.training for m in layer4_bn)
    assert earlier_bn and not any(m.training for m in earlier_bn)


def test_no_augmentation_train_transform_matches_eval():
    """
    The no-augmentation arms must train on exactly what they are evaluated on,
    or the 2x2 has a third, uncontrolled factor in it.
    """
    train = build_train_transform(AugmentationConfig(enabled=False))
    assert repr(train) == repr(build_eval_transform())


def test_augmented_transform_order():
    """
    Order is the contract: photometric jitter on PIL, then tensor conversion,
    then the tensor-space augmentations, then normalise, then erasing last.
    """
    transform = build_train_transform(AugmentationConfig(enabled=True))

    assert [type(t).__name__ for t in transform.transforms] == [
        "Resize",
        "Grayscale",
        "ColorJitter",
        "GammaJitter",
        "ToImage",
        "ToDtype",
        "RandomApply",  # SunGlareBlob
        "RandomApply",  # GaussianBlur
        "RandomApply",  # AdditiveGaussianNoise
        "Normalize",
        "RandomErasing",
    ]


def test_preview_transform_drops_only_normalisation():
    """The Figure 1 path: the same stack, still displayable."""
    full = [
        type(t).__name__
        for t in build_train_transform(AugmentationConfig(enabled=True)).transforms
    ]
    preview = [
        type(t).__name__
        for t in build_train_transform(
            AugmentationConfig(enabled=True), preview=True
        ).transforms
    ]
    assert [name for name in full if name != "Normalize"] == preview


def test_augmented_output_stays_in_range_and_shape():
    config = ImageConfig()
    torch.manual_seed(0)
    output = build_train_transform(
        AugmentationConfig(enabled=True), config, preview=True
    )(_grey_image())

    assert output.shape == (3, config.height, config.width)
    # The glare blob clamps and ToDtype scales, so nothing may leave [0, 1]
    # before normalisation.
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_augmentation_is_reproducible_from_the_torch_seed():
    """
    Every random draw in the stack comes from the torch global RNG, so a single
    seed controls augmentation as well as initialisation and shuffling.
    """
    image = _grey_image()
    transform = build_train_transform(AugmentationConfig(enabled=True), preview=True)

    torch.manual_seed(7)
    first = transform(image)
    torch.manual_seed(7)
    second = transform(image)

    assert torch.equal(first, second)
