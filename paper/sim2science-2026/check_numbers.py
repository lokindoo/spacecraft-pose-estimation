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

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
TEX_FILES = ("article.tex", "appendix.tex")  # appendix_revision.tex is generated from the CSVs

CHANCE = math.pi / 2 + 2 / math.pi

# Numbers that live outside the results CSVs, with where they come from.
LEGACY = {
    # The first baseline (4k images, 1000-d logits), re-scored on the current
    # 2,000-image validation split by spacecraft-pose-estimation/scripts/
    # legacy_baseline_eval.py from checkpoint_20260705_1620.pt (article
    # README, TODO 1). Re-derive before the freeze.
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
control = pd.read_csv(REVISION / "control_unfrozen_aug_seed42.csv").set_index("domain")

# Frozen augmented arm, seeds 43 and 44 (trained on the M5): Table 1 now averages
# them with the published seed 42, and App. B lists their coverage.
table = pd.concat([table, pd.read_csv(REVISION / "frozen_aug_seeds43_44.csv")], ignore_index=True)
coverage = pd.concat([coverage, pd.read_csv(REVISION / "conformal_frozen_aug.csv")
                      .replace({"domain": {"synthetic": "synthetic_old"}})], ignore_index=True)


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
    """Share of layer 4's marginal synthetic gain that reaches a real domain."""
    gain = mean("frozen_no_aug", "synthetic") - mean("unfrozen_no_aug", "synthetic")
    return (mean("frozen_no_aug", domain) - mean("unfrozen_no_aug", domain)) / gain


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
    Claim("abstract", r"cover {}--{}\% of rotation errors on sunlamp",
          lambda: pct(cov("unfrozen", "sunlamp", "rotation")), eps=0.05),
    Claim("abstract", r"{}--{}\% on lightbox, so a wrong pose",
          lambda: pct(cov("unfrozen", "lightbox", "rotation")), eps=0.05),
    Claim("abstract", r"rotation error from {} to {}\,rad",
          lambda: (LEGACY["synthetic"], mean("frozen_no_aug", "synthetic"))),
    Claim("abstract", r"domain ({}\,rad, within seed variation)",
          lambda: (LEGACY["sunlamp"] - mean("frozen_no_aug", "sunlamp"),)),
    Claim("abstract", r"carried {}--{}\% of its marginal",
          lambda: (100 * transfer("sunlamp"), 100 * transfer("lightbox"))),
    Claim("abstract", r"\emph{lightbox} domain from {} to {}\,rad",
          lambda: (mean("unfrozen_no_aug", "lightbox"), mean("unfrozen_aug", "lightbox"))),
    # ---- introduction
    Claim("intro", r"cover   only {}--{}\% of rotation errors on sunlamp",
          lambda: pct(cov("unfrozen", "sunlamp", "rotation")), eps=0.05),
    Claim("intro", r"{}\,rad of synthetic improvement left sunlamp within seed variation",
          lambda: (LEGACY["synthetic"] - mean("frozen_no_aug", "synthetic"),)),
    Claim("intro", r"test-time BatchNorm adaptation   lifts that only to {}--{}\%",
          lambda: adabn_coverage("sunlamp"), eps=0.0),
    Claim("intro", r"it improved lightbox by ${\sim}{}$\,rad on every seed",
          lambda: (-sum(paired("lightbox")) / 3,), slack=0.01),
    Claim("setup", r"\nicefrac{2}{\pi} \approx {}$\,rad; we call this",
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
    Claim("setup", r"As raw errors, {} and {}\,rad are neighbours; as skill they are {} and {}\,rad",
          lambda: (LEGACY["synthetic"], LEGACY["sunlamp"], skill(LEGACY["synthetic"]), skill(LEGACY["sunlamp"])),
          note="skill of the legacy sunlamp error 2.073 is 0.134"),
    # ---- 3.1 asymmetric transfer
    Claim("3.1", r"reached $\Erot = {}$\,rad", lambda: (LEGACY["synthetic"],)),
    Claim("3.1", r"rotation by {}\,rad ({}\,$\to$\,{}), but only {}\,rad on sunlamp ({}\,$\to$\,{})",
          lambda: (LEGACY["synthetic"] - mean("frozen_no_aug", "synthetic"), LEGACY["synthetic"],
                   mean("frozen_no_aug", "synthetic"), LEGACY["sunlamp"] - mean("frozen_no_aug", "sunlamp"),
                   LEGACY["sunlamp"], mean("frozen_no_aug", "sunlamp"))),
    Claim("3.1", r"less than the frozen head's seed s.d.\ there ({}\,rad)",
          lambda: (head(DEFAULT, "sunlamp", "std"),), eps=0.0),
    Claim("table 1", r"$\Erot$ s.d.\ {}, {}, {}",
          lambda: tuple(head(DEFAULT, d, "std") for d in ("synthetic", "lightbox", "sunlamp")), eps=0.0),
    Claim("table 1", r"frozen$^{\ddagger}$ & \checkmark & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$ & ${}{\pm}{}$",
          lambda: tuple(v for d in ("synthetic", "lightbox", "sunlamp") for m in ("translation_relative", "rotation")
                        for v in (mean("frozen_aug", d, m), sd("frozen_aug", d, m))), eps=0.001),
    Claim("3.1", r"between data and features ({} and {}\,rad)",
          lambda: (factor("synthetic", "data"), factor("synthetic", "features")), eps=0.0),
    Claim("3.1", r"A tuned head gains another {}\,rad on synthetic",
          lambda: (head(DEFAULT, "synthetic") - head("head_imagenetbn_pooled_28k_lr0.0003_ep60", "synthetic"),), eps=0.0),
    Claim("3.1", r"Lightbox absorbed roughly {}\% of the gain ({}\,$\to$\,{})",
          lambda: (100 * (LEGACY["lightbox"] - mean("frozen_no_aug", "lightbox"))
                   / (LEGACY["synthetic"] - mean("frozen_no_aug", "synthetic")),
                   LEGACY["lightbox"], mean("frozen_no_aug", "lightbox")), slack=1.0),
    Claim("3.1", r"holds {}\,rad of synthetic skill but keeps only {}\% of it on sunlamp ({}\% on lightbox)",
          lambda: (skill(mean("frozen_no_aug", "synthetic")),
                   100 * skill(mean("frozen_no_aug", "sunlamp")) / skill(mean("frozen_no_aug", "synthetic")),
                   100 * skill(mean("frozen_no_aug", "lightbox")) / skill(mean("frozen_no_aug", "synthetic")))),
    Claim("3.1", r"Synthetic $\Erot$ drops {}\,$\to$\,{}",
          lambda: (mean("frozen_no_aug", "synthetic"), mean("unfrozen_no_aug", "synthetic"))),
    Claim("3.1", r"sunlamp {}\,$\to$\,{} and lightbox {}\,$\to$\,{}, so roughly {}--{}\%",
          lambda: (mean("frozen_no_aug", "sunlamp"), mean("unfrozen_no_aug", "sunlamp"),
                   mean("frozen_no_aug", "lightbox"), mean("unfrozen_no_aug", "lightbox"),
                   100 * transfer("sunlamp"), 100 * transfer("lightbox"))),
    Claim("3.1", r"the layer-4 no-augmentation arm keeps only {}\% of its skill on sunlamp ({}\% on lightbox)",
          lambda: (100 * skill(mean(NOAUG, "sunlamp")) / skill(mean(NOAUG, "synthetic")),
                   100 * skill(mean(NOAUG, "lightbox")) / skill(mean(NOAUG, "synthetic")))),
    Claim("3.1", r"degrades {}--{}$\times$ ($S{=}{}$ synthetic, {} lightbox, {} sunlamp)",
          lambda: (speed(NOAUG, "lightbox") / speed(NOAUG, "synthetic"), speed(NOAUG, "sunlamp") / speed(NOAUG, "synthetic"),
                   speed(NOAUG, "synthetic"), speed(NOAUG, "lightbox"), speed(NOAUG, "sunlamp")), eps=0.002),
    # ---- 3.2 directional augmentation
    Claim("3.2", r"every seed: ${}$, ${}$, ${}$\,rad", lambda: tuple(paired("lightbox")), eps=0.001),
    Claim("3.2", r"3/3 at ${\sim}{}\%$",
          lambda: (-100 * sum(d / b for d, b in zip(paired("lightbox"), runs("unfrozen_no_aug", "lightbox"))) / 3,), slack=1.0),
    Claim("3.2", r"the mean from ${} \pm {}$ to ${} \pm {}$\,rad",
          lambda: (mean("unfrozen_no_aug", "lightbox"), sd("unfrozen_no_aug", "lightbox"),
                   mean("unfrozen_aug", "lightbox"), sd("unfrozen_aug", "lightbox")), eps=0.001),
    Claim("3.2", r"differences change sign: ${}$, ${}$, ${}$\,rad", lambda: tuple(paired("sunlamp")), eps=0.001),
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
    Claim("3.2", r"sunlamp improves on every seed (${}$ to ${}$\,rad), and by more than lightbox (${}$ to ${}$)",
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
    Claim("3.3", r"rotation coverage is {}--{}\% on sunlamp and {}--{}\% on lightbox (translation {}--{}\% / {}--{}\%)",
          lambda: (*pct(cov("unfrozen_no_aug", "sunlamp", "rotation")), *pct(cov("unfrozen_no_aug", "lightbox", "rotation")),
                   *pct(cov("unfrozen_no_aug", "sunlamp", "translation_metres")),
                   *pct(cov("unfrozen_no_aug", "lightbox", "translation_metres"))), eps=0.05),
    Claim("3.3", r"inflating these arms' attitude cone ${\sim}{}$--${}\times$, from ${\sim}{}$ to ${\sim}{}$\,rad",
          lambda: (cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").min(),
                   cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").max(),
                   cov("unfrozen_no_aug", "in_domain_test", "rotation", "q_hat").mean(),
                   float((cov("unfrozen_no_aug", "in_domain_test", "rotation", "q_hat").to_numpy()
                          * cov("unfrozen_no_aug", "sunlamp", "rotation", "inflation").to_numpy()).mean())),
          slack=0.1),
    Claim("3.3", r"(on average by {}\,rad on lightbox, {} on sunlamp)",
          lambda: (adabn_delta("lightbox"), adabn_delta("sunlamp")), eps=0.0),
    Claim("3.3", r"rotation coverage reaches only {}--{}\% and {}--{}\%",
          lambda: (*adabn_coverage("lightbox"), *adabn_coverage("sunlamp")), eps=0.0),
    Claim("3.3", r"roughly {}\% of all orientations", lambda: (100 * so3_fraction(2.8),), slack=1.0),
    Claim("3.3", r"$F(2.79) \approx {}$, and $F(2.30)$ to $F(2.42)$ span ${}$ to ${}$",
          lambda: (so3_fraction(2.79), so3_fraction(2.30), so3_fraction(2.42))),
    Claim("3.3", r"lightbox rotation coverage to {}--{}\% but sunlamp only to {}--{}\%",
          lambda: (*pct(cov("unfrozen_aug", "lightbox", "rotation")), *pct(cov("unfrozen_aug", "sunlamp", "rotation"))), eps=0.05),
    Claim("3.3", r"attitude cone already spans {}--{}\,rad---{}--{}\%",
          lambda: (cov("frozen", "in_domain_test", "rotation", "q_hat").min(), cov("frozen", "in_domain_test", "rotation", "q_hat").max(),
                   100 * so3_fraction(cov("frozen", "in_domain_test", "rotation", "q_hat").min()),
                   100 * so3_fraction(cov("frozen", "in_domain_test", "rotation", "q_hat").max())), eps=0.002),
    # ---- appendices
    Claim("App. B", r"reaches $1.2$--$13\times$ on the real domains",
          lambda: (), note="range checked structurally below"),
    Claim("App. C", r"at $2.048$\,rad, keeps ${}$\,rad",
          lambda: (constant.loc[("closed_form", "sunlamp"), "rotation"] - mean("frozen_no_aug", "sunlamp"),),
          note="read as skill over the fitted constant (2.199); over the uniform 2.207 it would be 0.16"),
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

    real = coverage[coverage.domain.isin(["lightbox", "sunlamp"])]
    results.append(("App. B inflation '1.2--13x' on real domains",
                    round(real.inflation.min(), 1) == 1.2 and round(real.inflation.max()) == 13,
                    f"{real.inflation.min():.2f}-{real.inflation.max():.2f}"))

    old = coverage[coverage.domain == "synthetic_old"]
    implied_n = (old.coverage * (1 - old.coverage) * (1.96 / ((old.coverage_high - old.coverage_low) / 2)) ** 2).median()
    results.append(("App. B 'synthetic (selection split)' = the 2,000-image selection split",
                    abs(implied_n - 2000) < 300 and "2{,}000-image validation split used\nfor checkpoint selection" in tex,
                    f"interval widths imply n ~ {implied_n:.0f}"))

    sd_sunlamp = head(DEFAULT, "sunlamp", "std")
    moved = LEGACY["sunlamp"] - mean("frozen_no_aug", "sunlamp")
    results.append(("'did not detectably move sunlamp': the move is below the frozen head's seed s.d.",
                    moved < sd_sunlamp, f"moved {moved:.3f}, s.d. {sd_sunlamp:.3f}"))
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
