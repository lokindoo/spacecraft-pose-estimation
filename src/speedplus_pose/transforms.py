"""
The photometric augmentation stack, and the evaluation transform.

This stack was built in good faith to anticipate harsh real illumination. The
paper's second finding is that it does not: it models *diffuse* lighting
variation, which is the lightbox domain, not sunlamp's collimated beam. Blur and
noise actively teach the network to discard the high-frequency detail - specular
edges, shadow boundaries - that is exactly what survives harsh light. Nothing
here is "wrong"; the point of section 3.2 is that an augmentation is a model of a
domain, and this one brackets the wrong domain.

ORDER MATTERS and is not arbitrary:

    resize + grayscale          on PIL
    ColorJitter + GammaJitter   on PIL, so gamma acts on 8-bit levels
    ToImage + ToDtype           -> float32 tensor in [0, 1]
    glare / blur / noise        on tensors, each at p = 0.5
    Normalize                   ImageNet statistics, matching the pretrained backbone
    RandomErasing               a regulariser, applied last and deliberately after
                                normalisation (it writes zeros in normalised space)

``preview=True`` drops only Normalize, so the result is still displayable. That
is exactly how the augmented crops in Figure 1 were produced.
"""

from __future__ import annotations

import math

import torch
from torchvision.transforms import v2

from .config import AugmentationConfig, ImageConfig

# ImageNet normalisation, because the backbone is ImageNet-pretrained.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _uniform(low: float, high: float) -> float:
    """
    Single scalar draw from the torch global RNG.

    Deliberately torch and not numpy: torch.manual_seed then controls
    augmentation along with weight init and shuffling. The published arms used
    this; the notebook that rendered the augmented crops used numpy draws
    instead, which changes which samples you see but not the distribution.
    """
    return torch.empty(1).uniform_(low, high).item()


class GammaJitter(torch.nn.Module):
    """
    Random gamma correction, log-uniform over [lo, hi].

    Log-uniform rather than uniform so darkening and brightening are equally
    likely - uniform on [0.5, 2.0] would spend two thirds of its mass on gamma > 1.
    """

    def __init__(self, gamma: tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.log_low, self.log_high = math.log(gamma[0]), math.log(gamma[1])

    def forward(self, img):
        gamma = math.exp(_uniform(self.log_low, self.log_high))
        return v2.functional.adjust_gamma(inpt=img, gamma=gamma)


class AdditiveGaussianNoise(torch.nn.Module):
    """Sensor noise, with the sigma itself drawn per sample."""

    def __init__(self, sigma: tuple[float, float] = (0.01, 0.08)):
        super().__init__()
        self.low, self.high = sigma

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return v2.functional.gaussian_noise(inpt=img, sigma=_uniform(self.low, self.high))


class SunGlareBlob(torch.nn.Module):
    """
    A soft Gaussian bright blob, clamped at 1.0.

    Intended as stray light. The clamp is the interesting part: it saturates
    pixels and so destroys whatever detail was under the blob, teaching the
    network to keep working when part of the frame is blown out. Capped at one
    blob per image - more than that removed too much of the spacecraft.

    Section 3.2: this models a *soft* bright region, whereas sunlamp's bright
    regions are saturated metallic surfaces that still carry pose information.
    """

    def __init__(
        self,
        sigma: tuple[float, float] = (10.0, 50.0),
        amplitude: tuple[float, float] = (0.8, 2.0),
        num_blobs: int = 1,
    ):
        super().__init__()
        self.sigma = sigma
        self.amplitude = amplitude
        self.num_blobs = num_blobs

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, height, width = img.size()
        for _ in range(self.num_blobs):
            sigma = _uniform(*self.sigma)
            amplitude = _uniform(*self.amplitude)
            x0, y0 = _uniform(0, width), _uniform(0, height)

            # Column and row coordinates, shaped to broadcast into (H, W).
            xs = torch.arange(width)[None, :]
            ys = torch.arange(height)[:, None]

            blob = amplitude * torch.exp(
                -((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma**2)
            )
            img = (img + blob).clamp(0.0, 1.0)
        return img


def _base(image: ImageConfig) -> list:
    # Grayscale with three output channels: the imagery is monochrome, but the
    # ImageNet backbone expects three channels.
    return [v2.Resize(image.size), v2.Grayscale(num_output_channels=3)]


def _to_tensor() -> list:
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]


def _normalise() -> list:
    return [v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def build_eval_transform(image: ImageConfig | None = None) -> v2.Compose:
    """Deterministic: resize, grayscale, to tensor, normalise. Nothing random."""
    image = image or ImageConfig()
    return v2.Compose(_base(image) + _to_tensor() + _normalise())


def build_train_transform(
    augmentation: AugmentationConfig,
    image: ImageConfig | None = None,
    preview: bool = False,
) -> v2.Compose:
    """
    Training transform for one arm.

    With ``augmentation.enabled`` false this is identical to the eval transform -
    the no-augmentation arms of the 2x2 train on exactly what they are evaluated
    on. With ``preview`` true, Normalize is omitted so the tensor still renders
    as an image (this is the Figure 1 path).
    """
    image = image or ImageConfig()
    if not augmentation.enabled:
        return build_eval_transform(image) if not preview else v2.Compose(
            _base(image) + _to_tensor()
        )

    a = augmentation
    photometric = [
        v2.ColorJitter(brightness=a.brightness, contrast=a.contrast),
        GammaJitter(a.gamma),
    ]
    tensor_augmentations = [
        v2.RandomApply([SunGlareBlob(a.glare_sigma, a.glare_amplitude)], p=a.stochastic_p),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=a.blur_kernel, sigma=a.blur_sigma)],
            p=a.stochastic_p,
        ),
        v2.RandomApply([AdditiveGaussianNoise(a.noise_sigma)], p=a.stochastic_p),
    ]
    erasing = [v2.RandomErasing(scale=a.erasing_scale)]

    return v2.Compose(
        _base(image)
        + photometric
        + _to_tensor()
        + tensor_augmentations
        + ([] if preview else _normalise())
        + erasing
    )
