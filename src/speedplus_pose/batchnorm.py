"""
Re-estimating BatchNorm statistics with every weight frozen.

Two experiments are this one operation on different data:

  synthetic-BN head   the backbone's ImageNet statistics replaced by statistics
                      of the (un-augmented) synthetic training set, before
                      features are cached and a head is trained on them. It
                      separates "layer 4 adapted its weights" from "layer 4
                      adapted its normalisation", which a frozen arm with
                      ImageNet statistics cannot.

  AdaBN               a trained model's statistics replaced by statistics of
                      unlabelled real images (Li et al., 2018). This is
                      test-time adaptation, and is reported separately for
                      that reason: it uses the target domain, though not its
                      labels.

The estimate is a cumulative average (``momentum=None``) over one pass, so it
does not depend on the order of the batches, unlike an exponential moving
average whose last few batches dominate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm


def batchnorm_layers(module: nn.Module) -> list[nn.BatchNorm2d]:
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def recompute_bn_statistics(
    backbone: nn.Module,
    dataloader,
    device: torch.device,
    description: str = "",
    layers: list[nn.BatchNorm2d] | None = None,
) -> int:
    """
    Reset the BatchNorm layers in ``backbone`` and re-estimate their running
    mean and variance from ``dataloader``, without touching any weight.
    Returns the number of images seen.

    ``layers`` restricts the re-estimate to a subset (default: every one).
    The others keep their statistics and stay in eval mode, so they normalise
    the pass exactly as they will at inference.

    Only the chosen BatchNorm layers go into train mode, so nothing else in
    the backbone can change behaviour during the pass. Afterwards the backbone
    is left in eval mode with momentum restored, ready for inference.
    """
    layers = batchnorm_layers(backbone) if layers is None else layers
    if not layers:
        raise ValueError("no BatchNorm2d layers to recompute")

    momenta = [layer.momentum for layer in layers]
    backbone.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None  # cumulative average over the whole pass
        layer.train()

    seen = 0
    for inputs, _ in tqdm(dataloader, desc=description or "recomputing BN", leave=False):
        backbone(inputs.to(device))
        seen += len(inputs)

    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
    backbone.eval()
    return seen
