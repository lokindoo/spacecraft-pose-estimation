"""Generate every table of the arXiv version into tables/, one file each, from the
code repository's results/ (no hand-typed numbers except the two external sources
marked below: Kisantal et al. 2020 Table II, and the notebook-era baseline).

    python make_tables.py        # from this directory

The body and appendices \\input these files, so a table can move between them
without being regenerated."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parents[1] / "results"
V, A = R / "revision", R / "arxiv"
OUT = HERE / "tables"
D = ["synthetic", "lightbox", "sunlamp"]
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s: f"${m:.3f}{{\\pm}}{s:.3f}$"
pct = lambda x: f"{100 * x:.1f}"
CHECK = r"\checkmark"
D1_HEAD = "head_imagenetbn_pooled_28k_lr0.001_ep20"  # the frozen un-augmented arm's head retrains


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(label, caption, spec, header, rows, size=r"\small", sep=None, foot=None):
    lines = [r"\begin{table}[ht]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"  \centering", f"  {size}"]
    if sep:
        lines.append(f"  \\setlength{{\\tabcolsep}}{{{sep}}}")
    lines += [f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule", *[f"    {h}" for h in header], r"    \midrule"]
    lines += [f"    {r}" if r.startswith(("\\midrule", "\\cmidrule")) else f"    {r} \\\\" for r in rows]
    lines.append(r"    \bottomrule")
    if foot:
        lines.append(f"    {foot}")
    lines += [r"  \end{tabular}", r"\end{table}"]
    return lines


# --- data -------------------------------------------------------------------
pub = pd.read_csv(R / "results_2x2.csv")
fa_new = pd.read_csv(V / "frozen_aug_seeds43_44.csv")
heads = pd.read_csv(V / "frozen_heads.csv")
heads["config"] = heads["name"].str.replace(r"_seed\d+$", "", regex=True)
d1 = heads[heads.config == D1_HEAD].assign(arm="frozen_no_aug")
runs = pd.concat([pub, fa_new], ignore_index=True)
# Table 1 under decision D1: the published frozen un-augmented seed 42, plus head-only
# retrains for seeds 43 and 44.
table1 = pd.concat([runs, d1[d1.seed.isin([43, 44])]], ignore_index=True)


def arm_stats(df, arm, metric):
    x = df[df.arm == arm].pivot(index="seed", columns="domain", values=metric)[D]
    assert len(x) == 3, (arm, metric, len(x))
    return x.mean(), x.std()


# --- Table 1 ----------------------------------------------------------------
rows = []
for backbone, aug, arm, mark in [("frozen", "--", "frozen_no_aug", r"$^{\dagger}$"), ("frozen", CHECK, "frozen_aug", r"$^{\ddagger}$"),
                                 ("layer 4", "--", "unfrozen_no_aug", ""), ("layer 4", CHECK, "unfrozen_aug", "")]:
    (tm, ts), (rm, rs) = arm_stats(table1, arm, "translation_relative"), arm_stats(table1, arm, "rotation")
    cells = []
    for d in D:
        r = pm(rm[d], rs[d])
        if arm == "unfrozen_aug" and d == "lightbox":
            r = f"$\\mathbf{{{rm[d]:.3f}{{\\pm}}{rs[d]:.3f}}}$"
        cells += [pm(tm[d], ts[d]), r]
    rows.append(f"{backbone}{mark} & {aug} & " + " & ".join(cells))
write("tab_results", table(
    "tab:results",
    r"Main results: mean $\pm$ s.d.\ over seeds \{42, 43, 44\}. $\Etrans$ range-normalised, $\Erot$ in rad (chance level 2.207). Checkpoints are picked by best synthetic validation $\Erot$.",
    "ll cc cc cc",
    [r"& & \multicolumn{2}{c}{Synthetic (val)} & \multicolumn{2}{c}{Lightbox} & \multicolumn{2}{c}{Sunlamp} \\",
     r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
     r"Backbone & Aug. & $\Etrans$ & $\Erot$ & $\Etrans$ & $\Erot$ & $\Etrans$ & $\Erot$ \\"],
    rows, size=r"\footnotesize", sep="2.4pt",
    foot=r"\multicolumn{8}{@{}p{0.98\linewidth}@{}}{\scriptsize $^{\dagger}$Seed 42 trained end to end, and seeds 43, 44 retrain the head on cached features (App.~\ref{app:revision}). Table~\ref{tab:frozen} retrains all three, and the text quotes it. $^{\ddagger}$Seeds 43, 44 on an Apple M5. Per-seed rows: App.~\ref{app:perseed}.}\\"))

# --- SPEC2019 (external: Kisantal et al. 2020, Table II, arXiv 1911.02050v2 p. 9) ---
spec = [("UniAdelaide", 0.0094, 0.3752, True), ("EPFL\\_cvlab", 0.0215, 0.1140, True), ("pedro\\_fairspace", 0.0571, 0.1555, False),
        ("Team Platypus", 0.0703, 1.7201, False), ("motokimura1", 0.0758, 0.6011, False), ("Magpies", 0.1393, 1.2659, False),
        ("GabrielA", 0.2423, 2.6209, False), ("stainsby", 0.3711, 5.0004, False), ("VSI\\_Feeney", 0.4658, 1.5993, False),
        ("jblumenkamp", 0.8859, 4.2418, True)]
ratios = [r / s for _, s, r, _ in spec]
write("tab_spec2019", table(
    "tab:spec2019",
    r"SPEC2019 top ten on SPEED~\citep{kisantal2020speed}: SPEED score on the synthetic and real test sets, and their ratio. "
    + f"Median ratio ${np.median(ratios):.1f}\\times$.",
    "rl cc c c",
    [r"\# & Team & $S_\mathrm{syn}$ & $S_\mathrm{real}$ & Ratio & PnP \\"],
    [f"{i} & {t} & {s:.4f} & {r:.4f} & ${r / s:.1f}\\times$ & {'yes' if p else 'no'}" for i, (t, s, r, p) in enumerate(spec, 1)]))

# --- the baseline in sequence (external: notebooks and the legacy checkpoint;
# check_numbers.py's LEGACY and NOTEBOOK give every source) ---
legacy_new = {"synthetic": 1.666, "lightbox": 2.002, "sunlamp": 2.073}
p42 = pub[(pub.arm == "frozen_no_aug") & (pub.seed == 42)].set_index("domain")
write("tab_baseline", table(
    "tab:baseline",
    r"The baseline in sequence: $\Erot$ (rad), on the 2{,}000-image synthetic validation split used everywhere else.",
    "l c cc",
    [r"Step & Synthetic (val) & Lightbox & Sunlamp \\"],
    [r"0. 4k images, logits head, MSE on raw quaternions & -- & -- & --",
     f"1. + labels canonicalised to one hemisphere & {legacy_new['synthetic']:.3f} & {legacy_new['lightbox']:.3f} & {legacy_new['sunlamp']:.3f}",
     r"2. + sign-invariant rotation loss & -- & -- & --",
     r"3. + pooled 2048-d features & -- & -- & --",
     f"4. + 28k images (published frozen arm, seed 42) & {p42.loc['synthetic','rotation']:.3f} & {p42.loc['lightbox','rotation']:.3f} & {p42.loc['sunlamp','rotation']:.3f}"],
    foot=r"\multicolumn{4}{@{}l}{\scriptsize Step 0 recorded only its loss, and steps 2 and 3 have no saved score on this split.}\\"))

# --- frozen heads (decomposition) --------------------------------------------
def cell(config, metric="rotation"):
    x = heads[heads.config == config].pivot(index="seed", columns="domain", values=metric)[D]
    assert len(x) == 3, config
    return x.mean(), x.std()
rows = []
for label, config in [("4k images, 1000-d logits", "head_imagenetbn_logits_4k_lr0.001_ep20"),
                      ("4k images, 2048-d pooled", "head_imagenetbn_pooled_4k_lr0.001_ep20"),
                      ("28k images, 1000-d logits", "head_imagenetbn_logits_28k_lr0.001_ep20"),
                      ("28k images, 2048-d pooled (published config.)", D1_HEAD),
                      ("28k, pooled, tuned (lr $3{\\times}10^{-4}$, 60 ep.)", "head_imagenetbn_pooled_28k_lr0.0003_ep60"),
                      ("28k, pooled, synthetic-BN backbone", "head_syntheticbn_pooled_28k_lr0.001_ep20")]:
    m, s = cell(config)
    rows.append(f"{label} & " + " & ".join(pm(m[d], s[d]) for d in D))
write("tab_frozen", table("tab:frozen", r"Frozen-backbone heads retrained on cached features: $\Erot$ (rad), mean $\pm$ s.d.\ over three seeds, every one a head retrain. The published configuration is Table~\ref{tab:results}'s frozen arm, whose seed 42 was trained end to end instead.",
                          "l ccc", [r"Configuration & Synthetic (val) & Lightbox & Sunlamp \\"], rows))

# --- frozen + aug, paired --------------------------------------------------------
fa = runs[runs.arm == "frozen_aug"].pivot(index="seed", columns="domain", values="rotation")[D]
ret = heads[heads.config == D1_HEAD].pivot(index="seed", columns="domain", values="rotation")[D]
rows = [f"{s} & " + " & ".join(f3(fa.loc[s, k]) for k in D) + " & " + " & ".join(f"${(fa.loc[s, k] - ret.loc[s, k]):+.3f}$" for k in D)
        for s in (42, 43, 44)]
write("tab_frozenaug", table(
    "tab:frozenaug", r"Frozen backbone with augmentation, $\Erot$ (rad) per seed and paired difference from the head-only retrain without augmentation (same seed).",
    "c ccc ccc",
    [r"& \multicolumn{3}{c}{$\Erot$} & \multicolumn{3}{c}{$\Delta\Erot$ vs.\ no augmentation} \\", r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
     r"Seed & Synth. & Lightbox & Sunlamp & Synth. & Lightbox & Sunlamp \\"], rows, sep="3pt"))

# --- ablation --------------------------------------------------------------------
def piv(df, arm, col="rotation"):
    return df[df.arm == arm].pivot(index="seed", columns="domain", values=col)[D]
arm_a, arm_b = pd.read_csv(V / "arm_a_no_blur_noise.csv"), pd.read_csv(V / "arm_b_jitter_gamma_erasing.csv")
full = piv(pub, "unfrozen_aug")
rows = []
for name, x, xt in [("full stack", full, piv(pub, "unfrozen_aug", "translation_relative")),
                    ("A: no blur/noise", piv(arm_a, "unfrozen_aug_no_blur_noise"), piv(arm_a, "unfrozen_aug_no_blur_noise", "translation_relative")),
                    ("B: jitter, gamma, erasing", piv(arm_b, "unfrozen_aug_jitter_gamma"), piv(arm_b, "unfrozen_aug_jitter_gamma", "translation_relative"))]:
    for seed in (42, 43, 44):
        dd = x.loc[seed] - full.loc[seed]
        diffs = " & ".join("--" if name == "full stack" else f"${dd[k]:+.3f}$" for k in D)
        rows.append(f"{name if seed == 42 else ''} & {seed} & " + " & ".join(f3(x.loc[seed, k]) for k in D) + f" & {diffs} & {f3(xt.loc[seed, 'sunlamp'])}")
    if name != "B: jitter, gamma, erasing":
        rows.append(r"\midrule")
write("tab_ablation", table(
    "tab:ablation", r"Augmentation ablation, $\Erot$ (rad) per seed and paired difference from the published full stack (layer 4, same seed). Last column: sunlamp $\Etrans$.",
    "l c ccc ccc c",
    [r"& & \multicolumn{3}{c}{$\Erot$} & \multicolumn{3}{c}{$\Delta\Erot$ vs.\ full stack} & Sunlamp \\", r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
     r"Arm & Seed & Synth. & Lightbox & Sunlamp & Synth. & Lightbox & Sunlamp & $\Etrans$ \\"], rows, sep="3pt"))

# --- AdaBN -----------------------------------------------------------------------
ad = pd.read_csv(V / "adabn.csv")
ad["aug"], ad["seed"] = ad.arm.str.startswith("unfrozen_aug"), ad.arm.str[-2:].astype(int)
rows = []
for aug in (False, True):
    for seed in (42, 43, 44):
        for dom in ("lightbox", "sunlamp"):
            g = ad[(ad.aug == aug) & (ad.seed == seed) & (ad.domain == dom)].set_index("condition")
            u, v = g.loc["unadapted"], g.loc["adapted"]
            first = (CHECK if aug else "--") if (seed == 42 and dom == "lightbox") else ""
            rows.append(f"{first} & {seed if dom == 'lightbox' else ''} & {dom} & {f3(u.rotation)} $\\to$ {f3(v.rotation)} & "
                        f"{f3(u.translation_relative)} $\\to$ {f3(v.translation_relative)} & {pct(u.coverage_rotation)} $\\to$ {pct(v.coverage_rotation)}")
    if not aug:
        rows.append(r"\midrule")
write("tab_adabn", table("tab:adabn", r"AdaBN on the held-out half of each real domain: unadapted $\to$ adapted. Coverage is of the 90\% synthetic-calibrated rotation cone.",
                         "l c l ccc", [r"Aug. & Seed & Domain & $\Erot$ & $\Etrans$ & Rotation coverage (\%) \\"], rows))

# --- image statistics ------------------------------------------------------------
st = pd.read_csv(V / "image_statistics.csv").set_index("pool")
labels = {"lightbox": "lightbox", "sunlamp": "sunlamp", "synthetic, no augmentation": "synthetic, no aug.",
          "synthetic, full stack": "synthetic, full stack", "synthetic, arm A (no blur/noise)": "synthetic, arm A",
          "synthetic, arm B (jitter + gamma + erasing)": "synthetic, arm B"}
rows = [f"{lab} & {100 * st.loc[p].edge_density:.1f} & {100 * st.loc[p].clipped_fraction:.2f} & {st.loc[p].tv_intensity_lightbox:.2f} & "
        f"{st.loc[p].tv_intensity_sunlamp:.2f} & {st.loc[p].tv_gradient_lightbox:.2f} & {st.loc[p].tv_gradient_sunlamp:.2f}" for p, lab in labels.items()]
write("tab_imagestats", table(
    "tab:imagestats", r"Pooled image statistics and total-variation distance to each real domain.", "l cc cc cc",
    [r"& Edge & Clipped & \multicolumn{2}{c}{TV intensity} & \multicolumn{2}{c}{TV gradient} \\", r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
     r"Pool & density (\%) & (\%) & lightbox & sunlamp & lightbox & sunlamp \\"], rows))

# --- augmentation parameters (speedplus_pose.config.AugmentationConfig defaults) ----
write("tab_augparams", table(
    "tab:augparams", r"The photometric stack, in order. Stochastic components apply with $p{=}0.5$. Arm A drops blur and noise, and arm B also drops the glare blob.",
    "l l c c", [r"Component & Parameters & Arm A & Arm B \\"],
    [r"Brightness / contrast jitter & factor $[0.65, 1.5]$ / $\pm 0.5$ & \checkmark & \checkmark",
     r"Gamma & log-uniform in $[0.5, 2]$ & \checkmark & \checkmark",
     r"Sun-glare blob ($p{=}0.5$) & clamped Gaussian, $\sigma \in [10, 50]$\,px, peak $[0.8, 2]$ & \checkmark & --",
     r"Gaussian blur ($p{=}0.5$) & kernel 7, $\sigma \in [0.1, 2]$ & -- & --",
     r"Additive Gaussian noise ($p{=}0.5$) & $\sigma \in [0.01, 0.08]$ & -- & --",
     r"Random erasing ($p{=}0.5$) & area $[0.02, 0.06]$ & \checkmark & \checkmark"], size=r"\footnotesize"))

# --- App. A: per-seed, with hardware --------------------------------------------
hw = {("frozen_no_aug", 42): "Colab", ("frozen_no_aug", 43): "head only", ("frozen_no_aug", 44): "head only",
      ("frozen_aug", 42): "Colab", ("frozen_aug", 43): "M5", ("frozen_aug", 44): "M5"}
rows = []
for backbone, aug, arm in [("frozen", "--", "frozen_no_aug"), ("frozen", CHECK, "frozen_aug"), ("layer 4", "--", "unfrozen_no_aug"), ("layer 4", CHECK, "unfrozen_aug")]:
    for seed in (42, 43, 44):
        g = table1[(table1.arm == arm) & (table1.seed == seed)].set_index("domain")
        for dom, lab in [("synthetic", "synthetic (val)"), ("lightbox", "lightbox"), ("sunlamp", "sunlamp")]:
            rows.append(f"{backbone} & {aug} & {seed} & {hw.get((arm, seed), 'Colab')} & {lab} & {f3(g.loc[dom, 'translation_relative'])} & {f3(g.loc[dom, 'rotation'])}")
    if arm != "unfrozen_aug":
        rows.append(r"\midrule")
write("tab_perseed", table("tab:perseed", r"Per-seed results behind Table~\ref{tab:results}. $\Etrans$ range-normalised, $\Erot$ in rad (chance level 2.207). \emph{Head only}: the head retrained on cached frozen features.",
                           "ll c l l cc", [r"Backbone & Aug. & Seed & Trained on & Domain & $\Etrans$ & $\Erot$ \\"], rows, size=r"\footnotesize"))

# --- App. B: conformal coverage ------------------------------------------------
cov = pd.concat([pd.read_csv(R / "conformal_coverage.csv"),
                 pd.read_csv(V / "conformal_frozen_aug.csv").replace({"domain": {"synthetic": "synthetic_old"}}),
                 pd.read_csv(V / "conformal_heads.csv").replace({"domain": {"synthetic": "synthetic_old"}}),
                 # the ablation file holds the M5 control under the published run's name
                 pd.read_csv(V / "conformal_ablation.csv").replace({"domain": {"synthetic": "synthetic_old"},
                                                                     "arm": {"unfrozen_aug_seed42": "control_unfrozen_aug_seed42"}})],
                ignore_index=True)
assert not cov.duplicated(["arm", "domain", "metric"]).any()
cov = cov.set_index(["arm", "domain", "metric"])
DOMS = [("in_domain_test", "in-domain test"), ("synthetic_old", "synthetic (val)"), ("lightbox", "lightbox"), ("sunlamp", "sunlamp")]


def cov_cell(arm, dom, metric, qfmt="{:.3f}"):
    r = cov.loc[(arm, dom, metric)]
    return f"{pct(r.coverage)} ({pct(r.coverage_low)}--{pct(r.coverage_high)}) & {qfmt.format(r.q_hat)} & {r.inflation:.2f}"


TABLE1_RUNS = [("frozen", "--", "frozen_no_aug_seed42", 42), ("frozen", "--", f"{D1_HEAD}_seed43", 43), ("frozen", "--", f"{D1_HEAD}_seed44", 44)] + \
              [("frozen", CHECK, f"frozen_aug_seed{s}", s) for s in (42, 43, 44)] + \
              [("layer 4", "--", f"unfrozen_no_aug_seed{s}", s) for s in (42, 43, 44)] + \
              [("layer 4", CHECK, f"unfrozen_aug_seed{s}", s) for s in (42, 43, 44)]
abs_rows, rel_rows = [], []
for i, (bb, aug, arm, seed) in enumerate(TABLE1_RUNS):
    for dom, lab in DOMS:
        abs_rows.append(f"{bb} & {aug} & {seed} & {lab} & {cov_cell(arm, dom, 'translation_metres')} & {cov_cell(arm, dom, 'rotation')}")
        rel_rows.append(f"{bb} & {aug} & {seed} & {lab} & {cov_cell(arm, dom, 'translation_relative', '{:.4f}')}")
    if i % 3 == 2 and i < len(TABLE1_RUNS) - 1:
        abs_rows.append(r"\midrule"); rel_rows.append(r"\midrule")
write("tab_conformal_abs", table(
    "tab:conformal-abs", r"Conformal coverage at nominal 90\%, absolute scores, for every run in Table~\ref{tab:results}. Coverage in \% with Clopper--Pearson 95\% intervals, $\hat{q}_T$ in metres and $\hat{q}_R$ in radians.",
    "ll c l ccc ccc",
    [r"& & & & \multicolumn{3}{c}{Translation (sphere)} & \multicolumn{3}{c}{Rotation (cone)} \\", r"\cmidrule(lr){5-7} \cmidrule(lr){8-10}",
     r"Backbone & Aug. & Seed & Domain & Cov. & $\hat{q}_T$ & Infl. & Cov. & $\hat{q}_R$ & Infl. \\"], abs_rows, size=r"\scriptsize", sep="3pt"))
write("tab_conformal_rel", table(
    "tab:conformal-rel", r"Conformal coverage at nominal 90\%, translation scored with the range-normalised error $\Etrans$.",
    "ll c l ccc", [r"Backbone & Aug. & Seed & Domain & Coverage & $\hat{q}$ (relative) & Infl. \\"], rel_rows, size=r"\scriptsize", sep="4pt"))

# Arms A and B, the control, and every frozen-head configuration: rotation coverage.
rows = []
for name, prefix in [("A: no blur/noise", "unfrozen_aug_no_blur_noise"), ("B: jitter, gamma, erasing", "unfrozen_aug_jitter_gamma")]:
    for seed in (42, 43, 44):
        arm = f"{prefix}_seed{seed}"
        rows.append(f"{name if seed == 42 else ''} & {seed} & " + " & ".join(pct(cov.loc[(arm, d, 'rotation')].coverage) for d, _ in DOMS)
                    + f" & {cov.loc[(arm, 'in_domain_test', 'rotation')].q_hat:.3f}")
    rows.append(r"\midrule")
for name, config in [("heads: 4k, logits", "head_imagenetbn_logits_4k_lr0.001_ep20"), ("heads: 4k, pooled", "head_imagenetbn_pooled_4k_lr0.001_ep20"),
                     ("heads: 28k, logits", "head_imagenetbn_logits_28k_lr0.001_ep20"), ("heads: 28k, pooled", D1_HEAD),
                     ("heads: tuned", "head_imagenetbn_pooled_28k_lr0.0003_ep60"), ("heads: synthetic BN", "head_syntheticbn_pooled_28k_lr0.001_ep20")]:
    vals = {d: [cov.loc[(f"{config}_seed{s}", d, "rotation")].coverage for s in (42, 43, 44)] for d, _ in DOMS}
    q = [cov.loc[(f"{config}_seed{s}", "in_domain_test", "rotation")].q_hat for s in (42, 43, 44)]
    rows.append(f"{name} & 42--44 & " + " & ".join(f"{pct(min(v))}--{pct(max(v))}" for v in vals.values()) + f" & {min(q):.2f}--{max(q):.2f}")
write("tab_conformal_revision", table(
    "tab:conformal-revision", r"Rotation coverage (\%) at nominal 90\% for the ablation arms (per seed) and the frozen-backbone heads (range over seeds 42--44), each calibrated on its own predictions. $\hat{q}_R$ in rad.",
    "l c cccc c", [r"Run & Seed & In-domain test & Synthetic (val) & Lightbox & Sunlamp & $\hat{q}_R$ \\"], rows, size=r"\scriptsize", sep="4pt"))

# --- §9 summary: rotation coverage by arm, with AdaBN ------------------------------
rows = []
for name, arms in [("frozen, no aug.", ["frozen_no_aug_seed42", f"{D1_HEAD}_seed43", f"{D1_HEAD}_seed44"]),
                   ("frozen, full stack", [f"frozen_aug_seed{s}" for s in (42, 43, 44)]),
                   ("layer 4, no aug.", [f"unfrozen_no_aug_seed{s}" for s in (42, 43, 44)]),
                   ("layer 4, full stack", [f"unfrozen_aug_seed{s}" for s in (42, 43, 44)]),
                   ("layer 4, arm A", [f"unfrozen_aug_no_blur_noise_seed{s}" for s in (42, 43, 44)]),
                   ("layer 4, arm B", [f"unfrozen_aug_jitter_gamma_seed{s}" for s in (42, 43, 44)])]:
    rng = lambda d: f"{pct(min(cov.loc[(a, d, 'rotation')].coverage for a in arms))}--{pct(max(cov.loc[(a, d, 'rotation')].coverage for a in arms))}"
    q = [cov.loc[(a, "in_domain_test", "rotation")].q_hat for a in arms]
    adapted = ""
    if name in ("layer 4, no aug.", "layer 4, full stack"):
        g = ad[(ad.arm.isin(arms)) & (ad.condition == "adapted")]
        adapted = " & ".join(f"{pct(g[g.domain == d].coverage_rotation.min())}--{pct(g[g.domain == d].coverage_rotation.max())}" for d in ("lightbox", "sunlamp"))
    else:
        adapted = "-- & --"
    rows.append(f"{name} & {min(q):.2f}--{max(q):.2f} & {rng('in_domain_test')} & {rng('lightbox')} & {rng('sunlamp')} & {adapted}")
write("tab_coverage_summary", table(
    "tab:coverage-summary", r"Rotation coverage (\%) of the 90\% synthetic-calibrated cone, range over three seeds. AdaBN columns: after re-estimating layer 4's BatchNorm statistics on the other half of each real domain.",
    "l c ccc cc",
    [r"& & & & & \multicolumn{2}{c}{With AdaBN} \\", r"\cmidrule(lr){6-7}",
     r"Arm & $\hat{q}_R$ (rad) & In-domain & Lightbox & Sunlamp & Lightbox & Sunlamp \\"], rows, size=r"\footnotesize", sep="3pt"))

# --- App. E: rotation-error shape (the pre-registered measures only) ----------------
sh = pd.read_csv(A / "rotation_error_shape.csv")
LABEL = {**{f"unfrozen_no_aug_seed{s}": f"layer 4, no aug., {s}" for s in (42, 43, 44)},
         **{f"unfrozen_aug_seed{s}": f"layer 4, full stack, {s}" for s in (42, 43, 44)}}
rows = []
for rec, lab in LABEL.items():
    for dom in ("lightbox", "sunlamp"):
        g = sh[(sh.record == rec) & (sh.domain == dom) & (sh.subset == "near_pi")].set_index("frame")
        b, c = g.loc["body"], g.loc["camera"]
        rows.append(f"{lab if dom == 'lightbox' else ''} & {dom} & {b['median']:.2f} & {b.near_pi_share:.3f} & {int(b.n_axes)} & {b.share_in_cone:.3f} & {c.share_in_cone:.3f}")
near_pi_uniform = sh.near_pi_uniform.iloc[0]
write("tab_rotshape", table(
    "tab:rotshape", f"Shape of the rotation errors on the real domains, layer-4 checkpoints. Near-$\\pi$: share of errors within 0.3\\,rad of $\\pi$ (uniform guessing: {near_pi_uniform:.2f}). "
    r"Cone: share of the near-$\pi$ errors' axes within $20^{\circ}$ of their principal axis, in the body and camera frames. A symmetry flip would concentrate the first and an in-plane confusion the second.",
    "l l c c c cc", [r"Run & Domain & Median $\Erot$ & Near-$\pi$ & $n_{\pi}$ & Cone, body & Cone, camera \\"], rows, size=r"\scriptsize", sep="4pt"))

# --- App. F: coverage by range bin ----------------------------------------------------
rb = pd.read_csv(A / "coverage_by_range.csv")
GROUPS = [("frozen, no aug.", ["frozen_no_aug_seed42"]), ("frozen, full stack", [f"frozen_aug_seed{s}" for s in (42, 43, 44)]),
          ("layer 4, no aug.", [f"unfrozen_no_aug_seed{s}" for s in (42, 43, 44)]), ("layer 4, full stack", [f"unfrozen_aug_seed{s}" for s in (42, 43, 44)])]
edges = rb[rb.record == "frozen_no_aug_seed42"].drop_duplicates("bin").sort_values("bin")
bin_labels = []
for _, e in edges.iterrows():
    lo = "" if np.isinf(e.range_low) else f"{e.range_low:.1f}"
    hi = "" if np.isinf(e.range_high) else f"{e.range_high:.1f}"
    bin_labels.append(f"$<{hi}$" if not lo else (f"$\\geq{lo}$" if not hi else f"{lo}--{hi}"))
for metric, mlabel in [("translation_metres", "translation, metres"), ("translation_relative", "translation, range-normalised"), ("rotation", "rotation")]:
    rows = []
    for gname, recs in GROUPS:
        for dom, dlab in [("in_domain_test", "in-domain"), ("lightbox", "lightbox"), ("sunlamp", "sunlamp")]:
            g = rb[rb.record.isin(recs) & (rb.domain == dom) & (rb.metric == metric)]
            m = g.groupby("bin").coverage.mean()
            rows.append(f"{gname if dom == 'in_domain_test' else ''} & {dlab} & " + " & ".join(pct(m[b]) for b in range(5)))
        if gname != GROUPS[-1][0]:
            rows.append(r"\midrule")
    write(f"tab_rangebins_{metric}", table(
        f"tab:rangebins-{metric.replace('_', '-')}",
        f"Coverage (\\%) of the marginal 90\\% region by range bin (m), {mlabel}, as the mean over the seeds of each arm. Bins are quintiles of the calibration split's range.",
        "l l ccccc", [r"Arm & Domain & " + " & ".join(bin_labels) + r" \\"], rows, size=r"\scriptsize", sep="4pt"))

# App. C (constant predictor) is ported by hand from the camera-ready: it reports the
# Karcher-refined constant, which scripts/constant_baseline.py does not implement.

print("wrote", ", ".join(sorted(p.stem for p in OUT.glob("*.tex"))))
