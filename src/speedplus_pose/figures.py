"""
The paper's figures, regenerable from artifacts.

A figure that only ever existed in a Colab output cell is a figure you will have
to regenerate at the worst possible moment, so each of these writes to disk at
publication size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import CHANCE_LEVEL

# matplotlib defaults, which the paper's pgfplots version matches by hand.
DOMAIN_COLOURS = {
    "synthetic": "#1F77B4",
    "in_domain_test": "#1F77B4",
    "lightbox": "#2CA02C",
    "sunlamp": "#FF7F0E",
}

# What each conformal score is, in the terms the paper uses for it.
METRIC_LABELS = {
    "rotation": "rotation (attitude cone)",
    "translation_metres": "translation (keep-out sphere)",
    "translation_relative": "translation (range-normalised)",
}


def _save(figure, path: Path, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return path


def rotation_error_by_arm(
    arms: list[str],
    errors: dict[str, dict[str, float]],
    stds: dict[str, dict[str, float]] | None = None,
    path: Path | None = None,
):
    """
    Figure 1: rotation error per arm and domain, against the chance level.

    The dashed line is the point of the figure. Read distances from it, not
    absolute heights - an error of 2.05 rad against a 2.207 chance level is a
    model that has learned almost nothing, which the raw number hides.
    """
    figure, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(1, len(arms) + 1)

    for domain in ("synthetic", "lightbox", "sunlamp"):
        y = [errors[domain][arm] for arm in arms]
        yerr = [stds[domain].get(arm, 0.0) for arm in arms] if stds else None
        ax.errorbar(
            x, y, yerr=yerr, marker="o", markersize=4, linewidth=1.2,
            capsize=0, color=DOMAIN_COLOURS[domain], label=domain,
        )

    ax.axhline(CHANCE_LEVEL, color="black", linestyle="--", linewidth=0.9)
    ax.annotate(
        # Raw string: these backslashes are mathtext markup, not Python escapes.
        rf"chance level $\pi/2 + 2/\pi \approx {CHANCE_LEVEL:.3f}$",
        xy=(np.mean(x), CHANCE_LEVEL),
        xytext=(0, -12), textcoords="offset points",
        ha="center", fontsize=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("rotation error $E_R$ [rad]")
    ax.set_ylim(0, 2.45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)

    figure.tight_layout()
    return _save(figure, path) if path else figure


def image_grid(images, rows: int, columns: int, path: Path | None = None, title: str = ""):
    """
    A bare grid of image crops - Figure 2 is two of these side by side.

    No axes, no ticks: the comparison is between what the augmented synthetic
    distribution looks like and what the real domains look like, and axis
    furniture only gets in the way of seeing it.
    """
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 1.5))
    for ax, image in zip(np.atleast_1d(axes).flat, images):
        image = np.asarray(image)
        if image.ndim == 3:
            # (C, H, W) tensors come out of the transform stack; take one channel
            # since the imagery is grayscale replicated across three.
            image = image[0] if image.shape[0] <= 4 else image[..., 0]
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()
    for ax in np.atleast_1d(axes).flat[len(images):]:
        ax.set_axis_off()
    if title:
        figure.suptitle(title)
    figure.tight_layout(pad=0.15)
    return _save(figure, path) if path else figure


def reliability(
    nominal: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],
    arm_name: str,
    metrics: tuple[str, ...] = ("translation_metres", "rotation"),
    path: Path | None = None,
    show_title: bool = False,
    font_scale: float = 1.0,
):
    """
    Figure 3: nominal versus measured coverage. The diagonal is a perfectly
    honest model; distance below it is how much the stated confidence overstates
    the real thing.

    ``show_title`` defaults off because the title names the seed, and the
    paper's build is anonymised - the published PNG had it cropped out.

    ``metrics`` may name one metric or two. In a 4-page paper there is no room
    for two panels that are both legible at column width, so the paper's version
    passes ("rotation",) alone - it is the stronger result - and ``font_scale``
    bumps every label so the single panel still reads at print size.
    """
    sizes = {
        "font.size": 11 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 11 * font_scale,
    }
    with plt.rc_context(sizes):
        figure, axes = plt.subplots(
            1, len(metrics), figsize=(5.5 * len(metrics), 4.6), sharey=True
        )
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            ax.plot([nominal.min(), 1.0], [nominal.min(), 1.0], "k--", lw=1, label="perfect")
            for domain, values in curves[metric].items():
                ax.plot(nominal, values, lw=2, label=domain, color=DOMAIN_COLOURS.get(domain))
            ax.set_xlabel("nominal coverage (calibrated on synthetic)")
            ax.set_title(METRIC_LABELS.get(metric, metric))
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("measured coverage")
        np.atleast_1d(axes)[0].legend(loc="lower right")
        if show_title:
            figure.suptitle(f"Conformal reliability - {arm_name}")
        figure.tight_layout()
    return _save(figure, path) if path else figure


def crops_with_histograms(
    columns: dict[str, list],
    path: Path | None = None,
    bins: int = 48,
    reference: str | None = None,
    log_counts: bool = True,
):
    """
    Figure 2: a triptych of image crops, one column per distribution, with an
    intensity-histogram strip underneath each column.

    ``columns`` maps a column heading ("augmented synthetic", "lightbox",
    "sunlamp") to a list of images - 2-D arrays, or (C, H, W) tensors as they
    come out of the transform stack. Every column needs the same number of them.

    THE HISTOGRAM STRIP IS THE ARGUMENT, not decoration. Without it the figure
    asks the reader to agree that the augmented crops "look like" lightbox; with
    it, the augmented column's pixel distribution is drawn over both real
    columns (dashed) and the reader can see it track one and not the other. That
    turns section 3.2's claim from an appeal to the eye into a measurement.
    ``reference`` names the column that gets overlaid, defaulting to the first.

    Counts are log-scaled by default: these frames are mostly empty space, so on
    a linear axis the black background is the only thing visible.
    """
    labels = list(columns)
    reference = reference or labels[0]
    rows = len(columns[labels[0]])
    if any(len(v) != rows for v in columns.values()):
        raise ValueError("every column needs the same number of crops")

    figure = plt.figure(figsize=(2.6 * len(labels), 1.55 * rows + 1.7))
    grid = figure.add_gridspec(
        rows + 1, len(labels),
        height_ratios=[1] * rows + [1.15],
        hspace=0.06, wspace=0.04,
    )

    edges = np.linspace(0.0, 1.0, bins + 1)
    densities = {label: _intensity_density(columns[label], edges) for label in labels}

    for column, label in enumerate(labels):
        for row in range(rows):
            ax = figure.add_subplot(grid[row, column])
            ax.imshow(_as_image(columns[label][row]), cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            if row == 0:
                # Column headings drawn on the figure itself - no axis furniture,
                # and nothing the caption has to explain.
                ax.set_title(label, fontsize=10, pad=4)

        ax = figure.add_subplot(grid[rows, column])
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(centres, densities[label], step="mid", alpha=0.65, color="#444444")
        if label != reference:
            ax.step(
                centres, densities[reference], where="mid",
                color="#C0392B", lw=1.2, ls="--", label=reference,
            )
            ax.legend(fontsize=7, frameon=False, loc="best")
        if log_counts:
            ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        # The neighbouring panel's "0" sits directly under this one's "1", so
        # only the rightmost column labels its right-hand end.
        ax.set_xticklabels(["0", "0.5", "1" if column == len(labels) - 1 else ""])
        ax.tick_params(labelsize=7)
        ax.set_yticklabels([])
        ax.set_xlabel("pixel intensity", fontsize=8)
        if column == 0:
            ax.set_ylabel("density (log)" if log_counts else "density", fontsize=8)

    return _save(figure, path) if path else figure


def _as_image(image) -> np.ndarray:
    """(C, H, W) tensors and 2-D arrays alike, as one grayscale plane in [0, 1]."""
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = image[0] if image.shape[0] <= 4 else image[..., 0]
    # Preview transforms leave images in [0, 1]; anything else (an 8-bit crop
    # read straight off disk) is brought onto the same scale so the histograms
    # of different columns are comparable at all.
    return image / 255.0 if image.max() > 1.5 else image


def _intensity_density(images: list, edges: np.ndarray) -> np.ndarray:
    """Pixel-intensity density pooled over one column's crops."""
    pixels = np.concatenate([_as_image(image).ravel() for image in images])
    counts, _ = np.histogram(pixels, bins=edges, density=True)
    return counts
