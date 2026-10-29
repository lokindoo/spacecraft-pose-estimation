"""
Tests for the augmentation ablation arms.

An ablation arm is only informative if it differs from the full stack in the
named components and nothing else: same survivors, same order, same
parameters, same training configuration.
"""

from dataclasses import replace

import pytest
from torchvision.transforms import v2

from speedplus_pose.config import AugmentationConfig, load_config
from speedplus_pose.transforms import build_train_transform

FULL_STACK = [
    "Resize",
    "Grayscale",
    "ColorJitter",
    "GammaJitter",
    "ToImage",
    "ToDtype",
    "SunGlareBlob",
    "GaussianBlur",
    "AdditiveGaussianNoise",
    "Normalize",
    "RandomErasing",
]

COMPONENT = {
    "use_jitter": "ColorJitter",
    "use_gamma": "GammaJitter",
    "use_glare": "SunGlareBlob",
    "use_blur": "GaussianBlur",
    "use_noise": "AdditiveGaussianNoise",
    "use_erasing": "RandomErasing",
}


def _names(transform: v2.Compose) -> list[str]:
    """Component names, looking inside RandomApply so the three p=0.5 ones are told apart."""
    names = []
    for t in transform.transforms:
        if isinstance(t, v2.RandomApply):
            names.extend(type(inner).__name__ for inner in t.transforms)
        else:
            names.append(type(t).__name__)
    return names


def test_all_switches_on_is_the_published_stack():
    assert _names(build_train_transform(AugmentationConfig(enabled=True))) == FULL_STACK


@pytest.mark.parametrize("switch", sorted(COMPONENT))
def test_each_switch_removes_exactly_its_component(switch):
    config = replace(AugmentationConfig(enabled=True), **{switch: False})
    assert _names(build_train_transform(config)) == [
        name for name in FULL_STACK if name != COMPONENT[switch]
    ]


def test_survivors_keep_their_parameters():
    full = build_train_transform(AugmentationConfig(enabled=True))
    ablated = build_train_transform(
        AugmentationConfig(enabled=True, use_blur=False, use_noise=False)
    )
    by_name = {type(t).__name__: repr(t) for t in full.transforms}
    for t in ablated.transforms:
        if type(t).__name__ in ("ColorJitter", "GammaJitter", "Normalize", "RandomErasing"):
            assert repr(t) == by_name[type(t).__name__]


def test_arm_a_drops_blur_and_noise_only():
    config = load_config("unfrozen_aug_no_blur_noise")
    assert _names(build_train_transform(config.augmentation)) == [
        n for n in FULL_STACK if n not in ("GaussianBlur", "AdditiveGaussianNoise")
    ]


def test_arm_b_is_jitter_gamma_and_erasing():
    # Erasing stays in, so arm B minus arm A isolates the glare blob.
    config = load_config("unfrozen_aug_jitter_gamma")
    assert _names(build_train_transform(config.augmentation)) == [
        "Resize", "Grayscale", "ColorJitter", "GammaJitter", "ToImage", "ToDtype", "Normalize",
        "RandomErasing",
    ]


@pytest.mark.parametrize("arm", ["unfrozen_aug_no_blur_noise", "unfrozen_aug_jitter_gamma"])
def test_ablation_arms_differ_from_the_full_stack_only_in_augmentation(arm):
    ablation, full = load_config(arm), load_config("unfrozen_aug")
    assert ablation.model == full.model
    assert ablation.train == full.train
    assert ablation.image == full.image
    # Every range and probability is shared; only the use_* switches differ.
    switches = {f: getattr(ablation.augmentation, f) for f in COMPONENT}
    assert replace(full.augmentation, **switches) == ablation.augmentation
