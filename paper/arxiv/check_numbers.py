#!/usr/bin/env python
"""
The number ledger: every derived number in the paper, recomputed from the
results CSVs and checked against what the text says.

    python check_numbers.py            # from this directory
    python check_numbers.py --verbose  # also list the claims that pass

Each claim names the exact phrase it checks, with {} where a number sits, and
the formula that should produce those numbers. A claim fails if a number is off
by more than the rounding its stated precision allows (plus the CSVs' own
3-decimal rounding), or if the phrase is no longer in the text, which means the
text changed and the ledger needs updating alongside it.

Run it on the current draft to find inconsistencies, and again after the
results freeze, before anything is submitted. Exit status is non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
# The arXiv source: body, appendices, and the tables make_tables.py writes.
TEX_FILES = ("article.tex", "appendix.tex", "appendix_constant.tex", "appendix_revision_text.tex",
             *sorted(f"tables/{p.name}" for p in (Path(__file__).resolve().parent / "tables").glob("*.tex")))

CHANCE = math.pi / 2 + 2 / math.pi

# Numbers that live outside the results CSVs, with where they come from.
LEGACY = {
    # The first baseline (4k images, 1000-d logits, canonicalised labels),
    # re-scored on the 2,000-image validation split by the code repository's
    # scripts/legacy_baseline_eval.py from checkpoint_20260705_1620.pt.
    "synthetic": 1.666,
    "lightbox": 2.002,
    "sunlamp": 2.073,
}


# --- data -------------------------------------------------------------------

table = pd.read_csv(RESULTS / "results_2x2.csv")
coverage = pd.read_csv(RESULTS / "conformal_coverage.csv")
constant = pd.read_csv(RESULTS / "constant_baseline.csv").set_index(["constant", "domain"])

# Revision experiments (App. D), from results/revision/.
REVISION = RESULTS / "revision"
heads = pd.read_csv(REVISION / "frozen_heads.csv")
adabn = pd.read_csv(REVISION / "adabn.csv")
stats = pd.read_csv(REVISION / "image_statistics.csv").set_index("pool")
arm_a = pd.read_csv(REVISION / "arm_a_no_blur_noise.csv")
arm_b = pd.read_csv(REVISION / "arm_b_jitter_gamma_erasing.csv")
control = pd.read_csv(REVISION / "control_unfrozen_aug_seed42.csv").set_index("domain")

# Frozen augmented arm, seeds 43 and 44 (trained on the M5): Table 1 now averages
# them with the published seed 42, and App. B lists their coverage.
table = pd.concat([table, pd.read_csv(REVISION / "frozen_aug_seeds43_44.csv")], ignore_index=True)
coverage = pd.concat([coverage, pd.read_csv(REVISION / "conformal_frozen_aug.csv")
                      .replace({"domain": {"synthetic": "synthetic_old"}})], ignore_index=True)


# arXiv-only analyses, from results/arxiv/ and the ablation's coverage.
ARXIV = RESULTS / "arxiv"
transfer_points = pd.read_csv(ARXIV / "transfer_points.csv")
shape = pd.read_csv(ARXIV / "rotation_error_shape.csv")
by_range = pd.read_csv(ARXIV / "coverage_by_range.csv")
ablation_cov = pd.read_csv(REVISION / "conformal_ablation.csv")

# Sec. 4's numbers from before the package existed, with where each was recorded.
NOTEBOOK = {
    "naive_train_loss": 0.8915531449317932,   # scripts/04-0-resnet50_baseline_test.ipynb, cell 25, epoch 50
    "canon_train_loss": 0.2853,               # checkpoint_20260705_1620.pt, train_losses[-1]
    "scaled_train_fit": 0.6539115686385194,   # 12-resnet50_..._scaled_data.ipynb, cell 32
}


def runs(arm: str, domain: str, metric: str = "rotation") -> pd.Series:
    rows = table[(table.arm == arm) & (table.domain == domain)].sort_values("seed")
    return rows[metric].reset_index(drop=True)


def mean(arm, domain, metric="rotation") -> float:
    return float(runs(arm, domain, metric).mean())


def sd(arm, domain, metric="rotation") -> float:
    return float(runs(arm, domain, metric).std(ddof=1))


def paired(domain: str, metric: str = "rotation") -> list[float]:
    """unfrozen_aug minus unfrozen_no_aug, seed by seed."""
    return list(runs("unfrozen_aug", domain, metric) - runs("unfrozen_no_aug", domain, metric))


heads_cov = pd.read_csv(REVISION / "conformal_heads.csv")


def frozen_cones() -> pd.Series:
    """In-domain rotation q_hat of every frozen run behind Table 1 under D1."""
    pub = coverage[coverage.arm.str.startswith("frozen") & (coverage.domain == "in_domain_test") & (coverage.metric == "rotation")]
    ret = heads_cov[heads_cov.arm.isin([f"{DEFAULT}_seed43", f"{DEFAULT}_seed44"])
                    & (heads_cov.domain == "in_domain_test") & (heads_cov.metric == "rotation")]
    values = pd.concat([pub.q_hat, ret.q_hat], ignore_index=True)
    assert len(values) == 6
    return values


def cov(arms_prefix: str, domain: str, metric: str, column: str = "coverage") -> pd.Series:
    rows = coverage[coverage.arm.str.startswith(arms_prefix) & (coverage.domain == domain) & (coverage.metric == metric)]
    return rows[column]


def pct(series: pd.Series) -> tuple[float, float]:
    return 100 * float(series.min()), 100 * float(series.max())


def skill(error: float) -> float:
    return CHANCE - error


def so3_fraction(angle: float) -> float:
    return (angle - math.sin(angle)) / math.pi


def transfer(domain: str) -> float:
    """Share of layer 4's extra synthetic gain that reaches a real domain, against the frozen
    three-seed cell (Table 3), the one frozen figure the text quotes."""
    gain = head(DEFAULT, "synthetic") - mean("unfrozen_no_aug", "synthetic")
    return (head(DEFAULT, domain) - mean("unfrozen_no_aug", domain)) / gain


def scaling(domain: str) -> float:
    """Finding 2 (D1 option B): 4k/logits minus 28k/pooled, three-seed head cells."""
    return head(CELLS["logits_4k"], domain) - head(CELLS["pooled_28k"], domain)


def speed(arm, domain) -> float:
    return mean(arm, domain, "translation_relative") + mean(arm, domain)


# The arm whose skill retention and SPEED+ degradation 3.1 quotes.
NOAUG = "unfrozen_no_aug"


def head(prefix: str, domain: str, stat: str = "mean") -> float:
    """Frozen heads on cached features, over seeds 42-44."""
    rows = heads[heads.name.str.startswith(prefix) & (heads.domain == domain)]
    assert len(rows) == 3, prefix
    return float(getattr(rows.rotation, stat)())


DEFAULT = "head_imagenetbn_pooled_28k_lr0.001_ep20"
CELLS = {c: f"head_imagenetbn_{c}_lr0.001_ep20" for c in ("logits_4k", "pooled_4k", "logits_28k", "pooled_28k")}


def frozen_paired(domain: str) -> list[float]:
    """Frozen backbone: augmented minus the head-only retrain without augmentation, seed by seed."""
    rows = heads[heads.name.str.startswith(DEFAULT) & (heads.domain == domain)].sort_values("seed")
    return list(runs("frozen_aug", domain) - rows.rotation.reset_index(drop=True))


def d1(domain: str, metric: str = "rotation") -> pd.Series:
    """Decision D1: published frozen no-aug seed 42, plus head-only retrains for seeds 43 and 44."""
    pub42 = table[(table.arm == "frozen_no_aug") & (table.domain == domain)][metric]
    retrains = heads[heads.name.str.startswith(DEFAULT) & heads.seed.isin([43, 44]) & (heads.domain == domain)][metric]
    values = pd.concat([pub42, retrains], ignore_index=True)
    assert len(values) == 3
    return values


def paired_vs(arm_rows: pd.DataFrame, base: str, domain: str) -> list[float]:
    """An arm minus the published `base` arm, seed by seed (42, 43, 44)."""
    x = arm_rows[arm_rows.domain == domain].sort_values("seed").rotation.reset_index(drop=True)
    return list(x - runs(base, domain))


def transfer_line(domain: str) -> float:
    """Least-squares slope of real on synthetic E_R over the frozen configurations with ImageNet BN."""
    g = transfer_points[(transfer_points.family == "frozen backbone") & (transfer_points.config != "synthetic BN")
                        & (transfer_points.domain == domain)]
    assert len(g) == 6
    return float(np.polyfit(g.synthetic, g.real, 1)[0])


def transfer_gap(domain: str) -> float:
    """How far the worst layer-4 arm beats the best frozen configuration on a real domain."""
    t = transfer_points[transfer_points.domain == domain]
    return t[t.family == "frozen backbone"].real.min() - t[t.family == "layer 4 trained"].real.max()


def l4_shape(domain: str, column: str, frame: str = "body") -> pd.Series:
    rows = shape[shape.record.str.match(r"unfrozen_(no_)?aug_seed4\d$") & (shape.domain == domain)
                 & (shape.frame == frame) & (shape.subset == "near_pi")]
    assert len(rows) == 6
    return rows[column]


def range_cov(domain: str, metric: str, b: int) -> float:
    """Layer 4 no aug: coverage in range bin b (0 nearest, 4 farthest), mean over seeds."""
    g = by_range[by_range.record.isin([f"unfrozen_no_aug_seed{s}" for s in (42, 43, 44)])
                 & (by_range.domain == domain) & (by_range.metric == metric) & (by_range.bin == b)]
    assert len(g) == 3
    return 100 * float(g.coverage.mean())


def adabn_et() -> pd.DataFrame:
    w = adabn[adabn.domain == "sunlamp"].pivot_table(index="arm", columns="condition", values="translation_relative")
    return w.assign(change=w.adapted - w.unadapted)


PREDICTIONS = RESULTS / "predictions"
labels = pd.read_csv(PREDICTIONS / "labels.csv.gz")


def range_slopes(domain: str) -> pd.Series:
    """Least-squares slope of predicted on true range, per layer-4 checkpoint, unadapted."""
    slopes = {}
    for record in (f"unfrozen_{arm}_seed{s}" for arm in ("no_aug", "aug") for s in (42, 43, 44)):
        p = pd.read_csv(PREDICTIONS / f"{record}.csv.gz").merge(labels, on=["domain", "filename"])
        p = p[p.domain == domain]
        predicted = np.linalg.norm(p[["x_pred", "y_pred", "z_pred"]].to_numpy(), axis=1)
        slopes[record] = np.polyfit(p["range"], predicted, 1)[0]
    return pd.Series(slopes)


def best_fixed_position(domain: str) -> float:
    """
    Lowest range-normalised translation error any single position scores on a domain,
    fitted to that domain's own labels: a geometric median weighted by 1/range (Weiszfeld).
    """
    t = labels[labels.domain == domain][["x_true", "y_true", "z_true"]].to_numpy()
    w = 1 / np.linalg.norm(t, axis=1)
    c = t.mean(axis=0)
    for _ in range(500):
        d = np.maximum(np.linalg.norm(t - c, axis=1), 1e-12)
        c = (t * (w / d)[:, None]).sum(axis=0) / (w / d).sum()
    return float(np.mean(np.linalg.norm(t - c, axis=1) * w))


def by_size(values: list[float]) -> tuple[float, float]:
    """The smallest and the largest effect, as a range is written in the text."""
    ordered = sorted(values, key=abs)
    return ordered[0], ordered[-1]


def factor(domain: str, which: str) -> float:
    """Average gain from 4k->28k images ('data') or logits->pooled ('features')."""
    m = {c: head(p, domain) for c, p in CELLS.items()}
    if which == "data":
        return ((m["logits_4k"] - m["logits_28k"]) + (m["pooled_4k"] - m["pooled_28k"])) / 2
    return ((m["logits_4k"] - m["pooled_4k"]) + (m["logits_28k"] - m["pooled_28k"])) / 2


def adabn_delta(domain: str) -> float:
    w = adabn[adabn.domain == domain].pivot(index="arm", columns="condition", values="rotation")
    return float((w.adapted - w.unadapted).mean())


def adabn_coverage(domain: str) -> tuple[float, float]:
    return pct(adabn[(adabn.domain == domain) & (adabn.condition == "adapted")].coverage_rotation)


def in_domain_body() -> pd.DataFrame:
    """In-domain coverage for the metrics the body scores: metres and rotation."""
    rows = coverage[coverage.domain == "in_domain_test"]
    return rows[rows.metric != "translation_relative"]


# --- claims -------------------------------------------------------------------


@dataclass
class Claim:
    where: str
    phrase: str  # literal text with {} for each number
    compute: Callable[[], tuple]
    eps: float = 0.0005  # rounding already present in the published 3-decimal CSVs; 0 for full-precision sources
    slack: float = 0.0  # extra tolerance for explicitly approximate claims ("~", "roughly")
    note: str = ""


CLAIMS = [
    # ---- abstract
    Claim("abstract", r"cover only {}--{}\% of rotation errors on sunlamp and",
          lambda: pct(cov("unfrozen", "sunlamp", "rotation")), eps=0.05),
    Claim("abstract", r"{}--{}\% on lightbox, so a wrong pose",
          lambda: pct(cov("unfrozen", "lightbox", "rotation")), eps=0.05),
    Claim("abstract", r"rotation error from {} to {}\,rad",
          lambda: (head(CELLS["logits_4k"], "synthetic"), head(CELLS["pooled_28k"], "synthetic"))),
    Claim("abstract", r"domain by only {}\,rad, under twice its seed s.d.", lambda: (scaling("sunlamp"),)),
    Claim("abstract", r"carried {}--{}\% of its extra synthetic gain",
          lambda: (100 * transfer("sunlamp"), 100 * transfer("lightbox"))),
    Claim("abstract", r"\emph{lightbox} domain from {} to {}\,rad",
          lambda: (mean("unfrozen_no_aug", "lightbox"), mean("unfrozen_aug", "lightbox"))),
    # ---- introduction
    Claim("intro", r"On sunlamp they cover only {}--{}\% of rotation errors",
          lambda: pct(cov("unfrozen", "sunlamp", "rotation")), eps=0.05),
    Claim("intro", r"improved synthetic rotation error from {} to {}\,rad, but sunlamp only from {} to   {}\,rad, less than twice its seed s.d.",
          lambda: (head(CELLS["logits_4k"], "synthetic"), head(CELLS["pooled_28k"], "synthetic"),
                   head(CELLS["logits_4k"], "sunlamp"), head(CELLS["pooled_28k"], "sunlamp"))),
    Claim("intro", r"from {} to {}\,rad, and {}--{}\% of that   extra gain carried across: lightbox went from {} to {}\,rad and sunlamp from {} to   {}.",
          lambda: (head(DEFAULT, "synthetic"), mean(NOAUG, "synthetic"),
                   100 * transfer("sunlamp"), 100 * transfer("lightbox"),
                   head(DEFAULT, "lightbox"), mean(NOAUG, "lightbox"),
                   head(DEFAULT, "sunlamp"), mean(NOAUG, "sunlamp"))),
    Claim("intro", r"test-time BatchNorm adaptation   lifts that only to {}--{}\%",
          lambda: adabn_coverage("sunlamp"), eps=0.0),
    Claim("intro", r"it improved lightbox by ${\sim}{}$\,rad on every seed",
          lambda: (-sum(paired("lightbox")) / 3,), slack=0.01),
    Claim("setup", r"\nicefrac{2}{\pi} \approx {}$\,rad, which we call the",
          lambda: (CHANCE,)),
    # ---- setup
    Claim("setup", r"lightbox ({} images) and sunlamp ({})",
          lambda: (constant.loc[("closed_form", "lightbox"), "n"], constant.loc[("closed_form", "sunlamp"), "n"])),
    Claim("setup", r"scores $\Erot = {}$ on synthetic val, ${}$ on lightbox and ${}$ on sunlamp",
          lambda: tuple(constant.loc[("closed_form", d), "rotation"] for d in ("synthetic", "lightbox", "sunlamp")),
          eps=0.001, note="the text quotes the Karcher-refined constant; the CSV holds the chordal one (agree to 0.001)"),
    Claim("setup", r"($\Etrans = {}$, ${}$, ${}$)",
          lambda: tuple(constant.loc[("closed_form", d), "translation_relative"] for d in ("synthetic", "lightbox", "sunlamp")),
          eps=0.001),
    Claim("setup", r"at most ${}$\,rad from the uniform value",
          lambda: (CHANCE - constant.xs("synthetic", level="domain").rotation.min(),), eps=0.001),
    Claim("setup", r"As raw errors, {} and {}\,rad are neighbours,   but as skill they are {} and {}\,rad",
          lambda: (LEGACY["synthetic"], LEGACY["sunlamp"], skill(LEGACY["synthetic"]), skill(LEGACY["sunlamp"])),
          note="skill of the legacy sunlamp error 2.073 is 0.134"),
    # ---- 3.1 asymmetric transfer
    Claim("3.1", r"$\Erot = {}$\,rad on the   2{,}000-image synthetic validation split, {} on lightbox and {} on sunlamp",
          lambda: tuple(head(CELLS["logits_4k"], d) for d in ("synthetic", "lightbox", "sunlamp"))),
    Claim("3.1", r"improves   synthetic rotation error by {}\,rad ({}\,$\to$\,{}), but sunlamp by only {}\,rad   ({}\,$\to$\,{})",
          lambda: (scaling("synthetic"), head(CELLS["logits_4k"], "synthetic"), head(CELLS["pooled_28k"], "synthetic"),
                   scaling("sunlamp"), head(CELLS["logits_4k"], "sunlamp"), head(CELLS["pooled_28k"], "sunlamp")), eps=0.001),
    Claim("3.1", r"less than twice the scaled head's seed s.d.\ on sunlamp   ({}\,rad)",
          lambda: (head(DEFAULT, "sunlamp", "std"),), eps=0.0),
    Claim("table 1", r"frozen$^{\dagger}$ & -- & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$",
          lambda: tuple(v for d in ("synthetic", "lightbox", "sunlamp") for m in ("translation_relative", "rotation")
                        for v in (d1(d, m).mean(), d1(d, m).std())), eps=0.001),
    Claim("table 1", r"frozen$^{\ddagger}$ & \checkmark & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$",
          lambda: tuple(v for d in ("synthetic", "lightbox", "sunlamp") for m in ("translation_relative", "rotation")
                        for v in (mean("frozen_aug", d, m), sd("frozen_aug", d, m))), eps=0.001),
    Claim("3.1", r"split the synthetic gain roughly evenly ({} and {}\,rad)",
          lambda: (factor("synthetic", "data"), factor("synthetic", "features")), eps=0.0),
    Claim("3.1", r"tuned head gains another {}\,rad on synthetic",
          lambda: (head(DEFAULT, "synthetic") - head("head_imagenetbn_pooled_28k_lr0.0003_ep60", "synthetic"),), eps=0.0),
    Claim("3.1", r"absorbs roughly a third of the gain ({}\,$\to$\,{})",
          lambda: (head(CELLS["logits_4k"], "lightbox"), head(CELLS["pooled_28k"], "lightbox"))),
    Claim("3.1", r"the scaled head   is {}\,rad better than chance on synthetic images but keeps only {}\% of that on sunlamp   ({}\% on lightbox)",
          lambda: (skill(head(CELLS["pooled_28k"], "synthetic")),
                   100 * skill(head(CELLS["pooled_28k"], "sunlamp")) / skill(head(CELLS["pooled_28k"], "synthetic")),
                   100 * skill(head(CELLS["pooled_28k"], "lightbox")) / skill(head(CELLS["pooled_28k"], "synthetic")))),
    Claim("3.1", r"against the same frozen head, synthetic $\Erot$ drops   {}\,$\to$\,{}",
          lambda: (head(DEFAULT, "synthetic"), mean("unfrozen_no_aug", "synthetic"))),
    Claim("3.1", r"sunlamp {}\,$\to$\,{} and lightbox {}\,$\to$\,{}. Between {}\%   (sunlamp) and {}\% (lightbox)",
          lambda: (head(DEFAULT, "sunlamp"), mean("unfrozen_no_aug", "sunlamp"),
                   head(DEFAULT, "lightbox"), mean("unfrozen_no_aug", "lightbox"),
                   100 * transfer("sunlamp"), 100 * transfer("lightbox"))),
    Claim("3.1", r"the layer-4 arm without   augmentation keeps only {}\% of its skill on sunlamp ({}\% on lightbox)",
          lambda: (100 * skill(mean(NOAUG, "sunlamp")) / skill(mean(NOAUG, "synthetic")),
                   100 * skill(mean(NOAUG, "lightbox")) / skill(mean(NOAUG, "synthetic")))),
    Claim("3.1", r"degrades {}--{}$\times$ ($S{=}{}$ synthetic, {} lightbox, {} sunlamp)",
          lambda: (speed(NOAUG, "lightbox") / speed(NOAUG, "synthetic"), speed(NOAUG, "sunlamp") / speed(NOAUG, "synthetic"),
                   speed(NOAUG, "synthetic"), speed(NOAUG, "lightbox"), speed(NOAUG, "sunlamp")), eps=0.002),
    # ---- 3.2 directional augmentation
    Claim("3.2", r"lowers the error on every seed:   ${}$, ${}$ and ${}$\,rad", lambda: tuple(paired("lightbox")), eps=0.001),
    Claim("3.2", r"about {}\% each time",
          lambda: (-100 * sum(d / b for d, b in zip(paired("lightbox"), runs("unfrozen_no_aug", "lightbox"))) / 3,), slack=1.0),
    Claim("3.2", r"the mean moves from   ${} \pm {}$ to ${} \pm {}$\,rad",
          lambda: (mean("unfrozen_no_aug", "lightbox"), sd("unfrozen_no_aug", "lightbox"),
                   mean("unfrozen_aug", "lightbox"), sd("unfrozen_aug", "lightbox")), eps=0.001),
    Claim("3.2", r"differences change sign:   ${}$, ${}$ and ${}$\,rad", lambda: tuple(paired("sunlamp")), eps=0.001),
    Claim("3.2", r"(${} \pm {} \to {} \pm {}$)",
          lambda: (mean("unfrozen_no_aug", "sunlamp"), sd("unfrozen_no_aug", "sunlamp"),
                   mean("unfrozen_aug", "sunlamp"), sd("unfrozen_aug", "sunlamp")), eps=0.001),
    Claim("3.2", r"augmented seeds (${}$, ${}$) fail to beat the fitted constant's ${}$",
          lambda: (*sorted(runs("unfrozen_aug", "sunlamp", "translation_relative"), reverse=True)[:2],
                   constant.loc[("closed_form", "sunlamp"), "translation_relative"])),
    Claim("3.2", r"essentially unchanged (${} \to {}$\,rad)",
          lambda: (mean("unfrozen_no_aug", "synthetic"), mean("unfrozen_aug", "synthetic"))),
    Claim("3.2", r"augmentation worsens synthetic error on every seed (by ${}$--${}$\,rad",
          lambda: by_size(frozen_paired("synthetic")), eps=0.001),
    Claim("3.2", r"sunlamp improves on every seed   (${}$ to ${}$\,rad), and by more than lightbox (${}$ to ${}$)",
          lambda: (*by_size(frozen_paired("sunlamp")), *by_size(frozen_paired("lightbox"))), eps=0.001),
    Claim("3.2", r"lowers sunlamp $\Erot$ on two of three seeds (${}$, ${}$, ${}$\,rad)",
          lambda: tuple(arm_a[arm_a.domain == "sunlamp"].sort_values("seed").rotation.to_numpy()
                        - runs("unfrozen_aug", "sunlamp").to_numpy()), eps=0.001),
    Claim("3.2", r"shifts seed 42's sunlamp error by ${}$\,rad",
          lambda: (control.loc["sunlamp", "rotation"] - runs("unfrozen_aug", "sunlamp")[0],), eps=0.001),
    Claim("3.2", r"clip less often than sunlamp ({}\% of pixels against {}\%)",
          lambda: (100 * stats.loc["synthetic, full stack", "clipped_fraction"], 100 * stats.loc["sunlamp", "clipped_fraction"]), eps=0.0),
    Claim("3.2", r"(edge density {}\%, against {}\% for lightbox and {}\% for the augmented stack)",
          lambda: tuple(100 * stats.loc[p, "edge_density"] for p in ("sunlamp", "lightbox", "synthetic, full stack")), eps=0.0),
    Claim("3.2", r"(gradient total variation {} against {})",
          lambda: tuple(stats.loc["synthetic, no augmentation", f"tv_gradient_{d}"] for d in ("lightbox", "sunlamp")), eps=0.0),
    Claim("3.2", r"pushes them away from both ({} and {})",
          lambda: tuple(stats.loc["synthetic, full stack", f"tv_gradient_{d}"] for d in ("lightbox", "sunlamp")), eps=0.0),
    # ---- 3.3 conformal
    Claim("3.3", r"arm and metric ({}--{}\%)", lambda: pct(in_domain_body().coverage), eps=0.05),
    Claim("3.3", r"translation ({}\%, {}--{}\%), which is slightly anti-conservative",
          lambda: tuple(100 * in_domain_body().query("coverage_high < 0.9").iloc[0][c]
                        for c in ("coverage", "coverage_low", "coverage_high")), eps=0.05),
    Claim("3.3", r"rotation coverage is {}--{}\% on sunlamp and {}--{}\% on lightbox (translation   {}--{}\% and {}--{}\%)",
          lambda: (*pct(cov("unfrozen_no_aug", "sunlamp", "rotation")), *pct(cov("unfrozen_no_aug", "lightbox", "rotation")),
                   *pct(cov("unfrozen_no_aug", "sunlamp", "translation_metres")),
                   *pct(cov("unfrozen_no_aug", "lightbox", "translation_metres"))), eps=0.05),
    Claim("3.3", r"attitude cone would have to grow ${\sim}{}$--${}\times$, from ${\sim}{}$ to ${\sim}{}$\,rad",
          lambda: (cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").min(),
                   cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").max(),
                   cov("unfrozen_no_aug", "in_domain_test", "rotation", "q_hat").mean(),
                   float((cov("unfrozen_no_aug", "in_domain_test", "rotation", "q_hat").to_numpy()
                          * cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").to_numpy()).mean())),
          slack=0.1),
    Claim("3.3", r"lifts rotation coverage only to {}--{}\% on   lightbox and {}--{}\% on sunlamp",
          lambda: (*adabn_coverage("lightbox"), *adabn_coverage("sunlamp")), eps=0.0),
    Claim("3.3", r"roughly {}\% of all orientations", lambda: (100 * so3_fraction(2.8),), slack=1.0),
    Claim("3.3", r"$F(2.79) \approx {}$, and $F({})$ to $F({})$ span ${}$ to ${}$",
          lambda: (so3_fraction(2.79), frozen_cones().min(), frozen_cones().max(),
                   so3_fraction(frozen_cones().min()), so3_fraction(frozen_cones().max())), eps=0.002),
    Claim("3.3", r"raises lightbox rotation   coverage to {}--{}\%, but sunlamp only to {}--{}\%",
          lambda: (*pct(cov("unfrozen_aug", "lightbox", "rotation")), *pct(cov("unfrozen_aug", "sunlamp", "rotation"))), eps=0.05),
    Claim("3.3", r"attitude cone already spans {}--{}\,rad, or {}--{}\%",
          lambda: (frozen_cones().min(), frozen_cones().max(),
                   100 * so3_fraction(frozen_cones().min()), 100 * so3_fraction(frozen_cones().max())), eps=0.002),
    # ---- appendices
    # ---- Sec. 4: the baseline in sequence
    Claim("4", r"the   training loss was still {}, barely below the 1.0",
          lambda: (NOTEBOOK["naive_train_loss"],)),
    Claim("4", r"brought the training loss to {} and the validation rotation error to {}\,rad",
          lambda: (NOTEBOOK["canon_train_loss"], LEGACY["synthetic"]),
          note="the 2,000-image validation split, as every other number"),
    Claim("4", r"scores {}\,rad on its own training images, {}\,rad better than on   validation. Those {}\,rad",
          lambda: (NOTEBOOK["scaled_train_fit"], mean("frozen_no_aug", "synthetic") - NOTEBOOK["scaled_train_fit"],
                   mean("frozen_no_aug", "synthetic") - NOTEBOOK["scaled_train_fit"]),
          note="seed 42, end to end: the only run with a recorded training-set fit"),
    # ---- Sec. 5: the transfer plot
    Claim("5", r"synthetic error spans   {}--{}\,rad. Yet a least-squares line through them moves sunlamp error by only {}\,rad   per radian of synthetic error (lightbox {})",
          lambda: (transfer_points[(transfer_points.family == "frozen backbone") & (transfer_points.config != "synthetic BN")].synthetic.min(),
                   transfer_points[(transfer_points.family == "frozen backbone") & (transfer_points.config != "synthetic BN")].synthetic.max(),
                   transfer_line("sunlamp"), transfer_line("lightbox"))),
    Claim("5", r"at a   synthetic error of {}--{}\,rad, every one of them beats every frozen configuration, by at   least {}\,rad on sunlamp and {} on lightbox",
          lambda: (transfer_points[transfer_points.family == "layer 4 trained"].synthetic.min(),
                   transfer_points[transfer_points.family == "layer 4 trained"].synthetic.max(),
                   transfer_gap("sunlamp"), transfer_gap("lightbox"))),
    # ---- Sec. 7: error shape
    Claim("7", r"is {}--{} on lightbox and {}--{} on   sunlamp, \emph{below} the {} that uniform guessing",
          lambda: (l4_shape("lightbox", "near_pi_share").min(), l4_shape("lightbox", "near_pi_share").max(),
                   l4_shape("sunlamp", "near_pi_share").min(), l4_shape("sunlamp", "near_pi_share").max(),
                   shape.near_pi_uniform.iloc[0]), eps=0.001),
    Claim("7", r"(medians {}--{}\,rad). Sunlamp errors spread from there towards $\pi$   (medians {}--{})",
          lambda: (l4_shape("lightbox", "median").min(), l4_shape("lightbox", "median").max(),
                   l4_shape("sunlamp", "median").min(), l4_shape("sunlamp", "median").max())),
    # ---- Sec. 8: normalisation statistics
    Claim("8", r"worse on every domain: {}\,rad on synthetic, {} on lightbox   and {} on sunlamp, against {}, {} and {} with ImageNet statistics",
          lambda: (*(head("head_syntheticbn_pooled_28k_lr0.001_ep20", d) for d in ("synthetic", "lightbox", "sunlamp")),
                   *(head(DEFAULT, d) for d in ("synthetic", "lightbox", "sunlamp")))),
    Claim("8", r"lowers rotation error on every layer-4 checkpoint, by {}\,rad on   lightbox and {} on sunlamp on average, yet rotation coverage reaches only {}--{}\% and   {}--{}\%",
          lambda: (adabn_delta("lightbox"), adabn_delta("sunlamp"), *adabn_coverage("lightbox"), *adabn_coverage("sunlamp")), eps=0.0),
    Claim("8", r"translation error on every layer-4   checkpoint, by {}--{}, to {}--{}, against the fitted constant's {}",
          lambda: (adabn_et().change.abs().min(), adabn_et().change.abs().max(), adabn_et().adapted.min(), adabn_et().adapted.max(),
                   constant.loc[("closed_form", "sunlamp"), "translation_relative"])),
    Claim("8", r"barely follows the true range (slope {}--{}, against {}--{} on synthetic)",
          lambda: (range_slopes("sunlamp").min(), range_slopes("sunlamp").max(),
                   range_slopes("synthetic").min(), range_slopes("synthetic").max()), eps=0.0),
    Claim("8", r"No fixed position scores below {} there", lambda: (best_fixed_position("sunlamp"),), eps=0.0),
    Claim("limitations", r"seed 42 moves by {}\,rad on synthetic and {} on   sunlamp",
          lambda: tuple(runs("frozen_no_aug", d)[0] - heads[heads.name == f"{DEFAULT}_seed42"].set_index("domain").rotation[d]
                        for d in ("synthetic", "sunlamp")),
          note="published end-to-end seed 42 against the head-only retrain of the same seed"),
    # ---- Sec. 9: coverage by range
    Claim("9", r"covers   {}\% of the nearest fifth of in-domain targets but only {}\% of the farthest. The   range-normalised score reverses the slope: {}\% near and {}\% far",
          lambda: (range_cov("in_domain_test", "translation_metres", 0), range_cov("in_domain_test", "translation_metres", 4),
                   range_cov("in_domain_test", "translation_relative", 0), range_cov("in_domain_test", "translation_relative", 4)), eps=0.05),
    Claim("9", r"under-covers at both ends ({}\% and {}\%)",
          lambda: (range_cov("in_domain_test", "rotation", 0), range_cov("in_domain_test", "rotation", 4)), eps=0.05),
    Claim("9", r"from {}\% of the nearest fifth to   {}\% of the farthest for the metre sphere, and from {}\% to {}\% when range-normalised,   while rotation coverage stays at {}--{}\%",
          lambda: (range_cov("sunlamp", "translation_metres", 0), range_cov("sunlamp", "translation_metres", 4),
                   range_cov("sunlamp", "translation_relative", 0), range_cov("sunlamp", "translation_relative", 4),
                   min(range_cov("sunlamp", "rotation", b) for b in range(5)), max(range_cov("sunlamp", "rotation", b) for b in range(5))), eps=0.05),
    # ---- App. H: the outcome column, typed by hand
    Claim("App. H", r"2/3 (${}$, ${}$, ${}$): softened to a hypothesis",
          lambda: tuple(paired_vs(arm_a, "unfrozen_aug", "sunlamp")), eps=0.001),
    Claim("App. H", r"1/3 (${}$, ${}$, ${}$): softened",
          lambda: tuple(b - a for a, b in zip(paired_vs(arm_a, "unfrozen_aug", "sunlamp"),
                                               paired_vs(arm_b, "unfrozen_aug", "sunlamp"))), eps=0.001),
    Claim("App. H", r"Lightbox ${}$ ($\pm{}$), sunlamp ${}$ ($\pm{}$): inside",
          lambda: (control.loc["lightbox", "rotation"] - runs("unfrozen_aug", "lightbox")[0], 2 * sd("unfrozen_aug", "lightbox"),
                   control.loc["sunlamp", "rotation"] - runs("unfrozen_aug", "sunlamp")[0], 2 * sd("unfrozen_aug", "sunlamp")), eps=0.001),
    Claim("App. H", r"s.d.\ {}: reworded", lambda: (head(DEFAULT, "sunlamp", "std"),), eps=0.0),
    Claim("App. H", r"Data {}, features {} ({}\%); largest sunlamp move {} against {}",
          lambda: (factor("synthetic", "data"), factor("synthetic", "features"),
                   100 * factor("synthetic", "data") / (factor("synthetic", "data") + factor("synthetic", "features")),
                   head(CELLS["logits_4k"], "sunlamp") - head(CELLS["logits_28k"], "sunlamp"), 2 * head(DEFAULT, "sunlamp", "std")), eps=0.001),
    Claim("App. H", r"{} against {}: tuned head reported",
          lambda: (float(heads[(heads.name == "head_imagenetbn_pooled_28k_lr0.0003_ep60_seed42") & (heads.domain == "synthetic")].rotation.iloc[0]),
                   mean("frozen_no_aug", "synthetic"))),
    Claim("App. H", r"${}$ (lightbox), ${}$ (sunlamp): ``not",
          lambda: tuple((head(DEFAULT, d) - head("head_syntheticbn_pooled_28k_lr0.001_ep20", d))
                        / (head(DEFAULT, d) - mean("unfrozen_no_aug", d)) for d in ("lightbox", "sunlamp"))),
    Claim("App. H", r"coverage at most {}\%",
          lambda: (100 * adabn[adabn.condition == "adapted"].coverage_rotation.max(),), eps=0.05),
    Claim("App. H", r"Clipping holds ({}\% vs {}\%); edge density fails ({}\% vs {}\%, {}\%)",
          lambda: (100 * stats.loc["sunlamp", "clipped_fraction"], 100 * stats.loc["synthetic, full stack", "clipped_fraction"],
                   100 * stats.loc["sunlamp", "edge_density"], 100 * stats.loc["lightbox", "edge_density"],
                   100 * stats.loc["synthetic, full stack", "edge_density"])),
    Claim("App. H", r"${}$, ${}$, ${}$: kept; sunlamp ${}$, ${}$, ${}$",
          lambda: (*frozen_paired("synthetic"), *frozen_paired("sunlamp")), eps=0.001),
    Claim("App. C", r"scores $2.048$\,rad and keeps ${}$\,rad",
          lambda: (constant.loc[("closed_form", "sunlamp"), "rotation"] - mean("frozen_no_aug", "sunlamp"),),
          note="read as skill over the fitted constant (2.199); over the uniform 2.207 it would be 0.16"),
    Claim("App. C", r"scoring   ${}$ on sunlamp, and matches this one on rotation to within ${}$",
          lambda: (constant.loc[("best", "sunlamp"), "translation_relative"],
                   float((constant.xs("best", level="constant").rotation
                          - constant.xs("closed_form", level="constant").rotation).abs().max())),
          note="the `best` constant minimises the reported metric directly (results/README.md)"),
    Claim("10", r"scores only ${}\times$   our synthetic $S$",
          lambda: (constant.loc[("closed_form", "sunlamp"), "speed_score"] / speed(NOAUG, "synthetic"),), eps=0.002),
]


# --- structural checks (claims about a whole table rather than one number) ---


def structural_checks(tex: str) -> list[tuple[str, bool, str]]:
    results = []
    body = in_domain_body()
    excluded = body[(body.coverage_low > 0.9) | (body.coverage_high < 0.9)]
    results.append(("3.3 'include 90% in every case but one, the frozen no-aug arm's translation'",
                    len(excluded) == 1 and excluded.iloc[0].arm == "frozen_no_aug_seed42"
                    and excluded.iloc[0].metric == "translation_metres" and excluded.iloc[0].coverage_high < 0.9,
                    "; ".join(f"{r.arm} {r.metric}" for r in excluded.itertuples())))


    t1 = pd.concat([coverage, heads_cov[heads_cov.arm.isin([f"{DEFAULT}_seed43", f"{DEFAULT}_seed44"])]], ignore_index=True)
    t1 = t1[t1.metric.isin(["translation_metres", "rotation"])]
    real, ind = t1[t1.domain.isin(["lightbox", "sunlamp"])], t1[t1.domain == "in_domain_test"]
    worst = real.loc[real.inflation.idxmax(), "arm"]
    results.append(("App. B 'near 1 in-domain and reaches 1.2--13x on the real domains, worst on the unfrozen arms'",
                    ind.inflation.between(0.9, 1.15).all() and round(real.inflation.min(), 1) == 1.2
                    and round(real.inflation.max()) == 13 and worst.startswith("unfrozen")
                    and "reaches $1.2$--$13\\times$ on the real domains" in " ".join(tex.split()),
                    f"in-domain {ind.inflation.min():.2f}-{ind.inflation.max():.2f}, real {real.inflation.min():.2f}-{real.inflation.max():.2f}, worst {worst}"))

    sun = pd.concat([table[(table.domain == "sunlamp")].assign(run=lambda d: d.arm + "_" + d.seed.astype(str))[["run", "rotation"]],
                     heads[heads.name.isin([f"{DEFAULT}_seed43", f"{DEFAULT}_seed44"]) & (heads.domain == "sunlamp")]
                     .assign(run=lambda d: d.name)[["run", "rotation"]]], ignore_index=True)
    weakest = sun.loc[sun.rotation.idxmax()]
    results.append(("App. C 'the weakest run, seed 42 of the frozen no-aug arm on sunlamp'",
                    weakest.run == "frozen_no_aug_42" and len(sun) == 12, f"{weakest.run} {weakest.rotation:.3f} of {len(sun)} runs"))

    old = coverage[coverage.domain == "synthetic_old"]
    implied_n = (old.coverage * (1 - old.coverage) * (1.96 / ((old.coverage_high - old.coverage_low) / 2)) ** 2).median()
    flat = " ".join(tex.split())
    results.append(("App. B 'synthetic (val)' rows = the 2,000-image selection split",
                    abs(implied_n - 2000) < 300 and "(val)} rows are the 2{,}000-image validation split used for checkpoint selection" in flat,
                    f"interval widths imply n ~ {implied_n:.0f}"))

    sd_sunlamp = head(DEFAULT, "sunlamp", "std")
    moved = scaling("sunlamp")
    results.append(("'moved sunlamp by less than twice its seed s.d.': the cells' move",
                    abs(moved) < 2 * sd_sunlamp, f"moved {moved:.3f}, 2 s.d. {2 * sd_sunlamp:.3f}"))
    share = scaling("lightbox") / scaling("synthetic")
    results.append(("3.1 'lightbox absorbs roughly a third of the gain'", 0.28 <= share <= 0.38, f"{share:.3f}"))
    results.append(("'neither moves sunlamp beyond seed variation': both factors within 2 s.d.",
                    all(abs(factor("sunlamp", w)) < 2 * sd_sunlamp for w in ("data", "features")),
                    f"data {factor('sunlamp', 'data'):.3f}, features {factor('sunlamp', 'features'):.3f}"))
    tuned = "head_imagenetbn_pooled_28k_lr0.0003_ep60"
    results.append(("'a tuned head gains ... nothing on the real domains': within one s.d.",
                    all(abs(head(DEFAULT, d) - head(tuned, d)) < head(DEFAULT, d, "std") for d in ("lightbox", "sunlamp")),
                    ", ".join(f"{d} {head(DEFAULT, d) - head(tuned, d):+.3f}" for d in ("lightbox", "sunlamp"))))
    syn_bn = "head_syntheticbn_pooled_28k_lr0.001_ep20"
    results.append(("'synthetic-BN makes the frozen head worse everywhere'",
                    all(head(syn_bn, d) > head(DEFAULT, d) for d in ("synthetic", "lightbox", "sunlamp")), ""))
    w = adabn.pivot_table(index=["arm", "domain"], columns="condition", values="rotation")
    results.append(("'AdaBN lowers rotation error on every layer-4 checkpoint'",
                    bool((w.adapted < w.unadapted).all()), f"{len(w)} checkpoint x domain pairs"))
    synth, light, sun = (frozen_paired(d) for d in ("synthetic", "lightbox", "sunlamp"))
    results.append(("3.2 frozen: 'worsens synthetic error on every seed'", all(x > 0 for x in synth),
                    ", ".join(f"{x:+.3f}" for x in synth)))
    results.append(("3.2 frozen: 'sunlamp improves on every seed, and by more than lightbox'",
                    all(s < 0 and s < l for s, l in zip(sun, light)),
                    "; ".join(f"sunlamp {s:+.3f} vs lightbox {l:+.3f}" for s, l in zip(sun, light))))
    results.append(("'sunlamp ... edge density is the lowest of any split'",
                    stats.edge_density.idxmin() == "sunlamp", ""))
    # Sec. 7: the pre-registered axis rules, as the text states them.
    for d in ("lightbox", "sunlamp"):
        body, cam = (l4_shape(d, "share_in_cone", f) for f in ("body", "camera"))
        results.append((f"7 '0.18 on only two of six checkpoints per domain' (body) and on none (camera), {d}",
                        int((body >= 0.18).sum()) == 2 and int((cam >= 0.18).sum()) == 0,
                        f"body {int((body >= 0.18).sum())}/6, camera {int((cam >= 0.18).sum())}/6"))
    # Sec. 9: in-domain rotation over-covers in the middle bins.
    mid = [range_cov("in_domain_test", "rotation", b) for b in (1, 2, 3)]
    results.append(("9 'over-covers between them'", all(m > 90 for m in mid), ", ".join(f"{m:.1f}" for m in mid)))
    # Sec. 9: the ablation rule, coverage moving opposite to error on 4 of 6 checkpoints per domain.
    counts = {}
    for d in ("lightbox", "sunlamp"):
        n = 0
        for prefix, frame in (("unfrozen_aug_no_blur_noise", arm_a), ("unfrozen_aug_jitter_gamma", arm_b)):
            for seed in (42, 43, 44):
                c_arm = ablation_cov[(ablation_cov.arm == f"{prefix}_seed{seed}") & (ablation_cov.domain == d) & (ablation_cov.metric == "rotation")].coverage.iloc[0]
                c_pub = coverage[(coverage.arm == f"unfrozen_aug_seed{seed}") & (coverage.domain == d) & (coverage.metric == "rotation")].coverage.iloc[0]
                e_arm = frame[(frame.seed == seed) & (frame.domain == d)].rotation.iloc[0]
                n += (c_arm - c_pub) * (e_arm - runs("unfrozen_aug", d)[[42, 43, 44].index(seed)]) < 0
        counts[d] = n
    results.append(("9 'four held on each' (ablation coverage opposite to error)", counts == {"lightbox": 4, "sunlamp": 4}, str(counts)))
    # Intro: 'five of those rules softened or qualified a claim'.
    pre = tex[tex.index("label{tab:prereg}"):tex.index("Deviations and corrections")]
    changed = len(re.findall(r"softened|reworded|qualified", pre))
    results.append(("intro 'five of those rules softened or qualified a claim'", changed == 5, f"{changed} in App. H"))
    return results


# --- runner -------------------------------------------------------------------

NUMBER = r"([+-]?\d[\d{},]*(?:\.\d+)?)"


def phrase_regex(phrase: str) -> re.Pattern:
    parts = phrase.split("{}")
    escaped = [re.sub(r"(\\\s)+|\s+", r"\\s+", re.escape(p)) for p in parts]
    return re.compile(NUMBER.join(escaped))


def parse(number: str) -> tuple[float, int]:
    clean = number.replace("{,}", "").replace(",", "")
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return float(clean), decimals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    tex = "\n".join((HERE / name).read_text(encoding="utf-8") for name in TEX_FILES)
    failures = 0

    for claim in CLAIMS:
        match = phrase_regex(claim.phrase).search(tex)
        label = f"[{claim.where}] {claim.phrase.replace(chr(10), ' ')}"
        if match is None:
            failures += 1
            print(f"MISSING  {label}\n         phrase not found: the text changed, update the ledger")
            continue
        expected = claim.compute()
        stated = [parse(g) for g in match.groups()]
        if len(stated) != len(expected):
            if expected:
                failures += 1
                print(f"ARITY    {label}: {len(stated)} numbers in text, {len(expected)} computed")
            elif args.verbose:
                print(f"ok       {label} (checked elsewhere)")
            continue
        bad = []
        for (value, decimals), computed in zip(stated, expected):
            computed = float(computed)
            tolerance = 0.5 * 10 ** (-decimals) + claim.eps + claim.slack
            # Signs are written explicitly in the text when they matter.
            if abs(value - computed) > tolerance and abs(abs(value) - abs(computed)) > tolerance:
                bad.append(f"text {value:g} vs computed {computed:.4f}")
        if bad:
            failures += 1
            print(f"FAIL     {label}\n         " + "; ".join(bad) + (f"\n         note: {claim.note}" if claim.note else ""))
        elif args.verbose:
            print(f"ok       {label}")

    for name, ok, detail in structural_checks(tex):
        if not ok:
            failures += 1
            print(f"FAIL     [structure] {name}\n         {detail}")
        elif args.verbose:
            print(f"ok       [structure] {name}: {detail}")

    print(f"\n{len(CLAIMS)} claims + structural checks, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
