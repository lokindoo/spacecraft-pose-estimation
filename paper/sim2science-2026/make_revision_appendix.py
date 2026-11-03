"""Generate appendix_revision.tex from the revision result files in the code
repository's results/revision/ (no hand-typed numbers)."""
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"
V = R / "revision"
D = ["synthetic", "lightbox", "sunlamp"]
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s: f"${m:.3f}{{\\pm}}{s:.3f}$"

heads = pd.read_csv(V / "frozen_heads.csv")
def cell(prefix, metric="rotation"):
    x = heads[heads.name.str.startswith(prefix)].pivot(index="seed", columns="domain", values=metric)[D]
    assert len(x) == 3, prefix
    return x.mean(), x.std()

out = [r"""\section{Revision experiments}
\label{app:revision}

All runs in this appendix were added after review and pre-registered: predictions and
decision rules were written down and committed before any result was opened. Selection
uses synthetic validation only, as in the body.

\paragraph{Frozen-backbone block.} Because the frozen backbone runs in inference mode
without augmentation, its pooled features (and the ImageNet logits computed from them)
can be cached once and heads retrained on them in minutes. Applying the published
seed-42 head to the cache reproduces its published errors to within $0.0005$.
Table~\ref{tab:frozen} gives three seeds (42, 43, 44) for each configuration, all with
the published head, loss and selection rule. The 4{,}000-image cells use one fixed
draw from the 28{,}000 (not the original 4{,}000-image set). The tuned head is the best
of learning rate $\{3{\times}10^{-4}, 10^{-3}, 3{\times}10^{-3}\}\times\{20, 60\}$
epochs on seed 42, chosen on synthetic validation. The synthetic-BN head re-estimates
every BatchNorm layer on the un-augmented synthetic training images (weights
untouched) before caching.

\begin{table}[ht]
  \caption{Frozen-backbone heads retrained on cached features: $\Erot$ (rad), mean
  $\pm$ s.d.\ over three seeds.}
  \label{tab:frozen}
  \centering
  \small
  \begin{tabular}{l ccc}
    \toprule
    Configuration & Synthetic (val) & Lightbox & Sunlamp \\
    \midrule"""]
rows = [
    ("4k images, 1000-d logits", "head_imagenetbn_logits_4k_lr0.001_ep20"),
    ("4k images, 2048-d pooled", "head_imagenetbn_pooled_4k_lr0.001_ep20"),
    ("28k images, 1000-d logits", "head_imagenetbn_logits_28k_lr0.001_ep20"),
    ("28k images, 2048-d pooled (published config.)", "head_imagenetbn_pooled_28k_lr0.001_ep20"),
    ("28k, pooled, tuned (lr $3{\\times}10^{-4}$, 60 ep.)", "head_imagenetbn_pooled_28k_lr0.0003_ep60"),
    ("28k, pooled, synthetic-BN backbone", "head_syntheticbn_pooled_28k_lr0.001_ep20"),
]
for label, prefix in rows:
    m, s = cell(prefix)
    out.append(f"    {label} & " + " & ".join(pm(m[d], s[d]) for d in D) + r" \\")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

# Frozen backbone with augmentation: seed 42 published, seeds 43 and 44 trained on the
# M5, each paired with the head-only retrain of the same seed.
pub = pd.read_csv(R / "results_2x2.csv")
fa = pd.concat([pub[pub.arm == "frozen_aug"], pd.read_csv(V / "frozen_aug_seeds43_44.csv")])
fa = fa.pivot(index="seed", columns="domain", values="rotation")[D]
ret = heads[heads.name.str.startswith("head_imagenetbn_pooled_28k_lr0.001_ep20")].pivot(index="seed", columns="domain", values="rotation")[D]
out.append(r"""\paragraph{Frozen backbone with augmentation.} Augmented inputs cannot use the feature
cache, so seeds 43 and 44 of the frozen augmented arm were trained end to end with the
released code on an Apple M5 GPU; seed 42 is the published Colab run.
Table~\ref{tab:frozenaug} pairs each seed with the head-only retrain of the same seed in
the published configuration (Table~\ref{tab:frozen}). Their conformal coverage is in
App.~\ref{app:conformal}.

\begin{table}[ht]
  \setlength{\tabcolsep}{3pt}
  \caption{Frozen backbone with augmentation, $\Erot$ (rad) per seed and paired difference
  from the head-only retrain without augmentation (same seed).}
  \label{tab:frozenaug}
  \centering
  \small
  \begin{tabular}{c ccc ccc}
    \toprule
    & \multicolumn{3}{c}{$\Erot$} & \multicolumn{3}{c}{$\Delta\Erot$ vs.\ no augmentation} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    Seed & Synth. & Lightbox & Sunlamp & Synth. & Lightbox & Sunlamp \\
    \midrule""")
for seed in (42, 43, 44):
    d = fa.loc[seed] - ret.loc[seed]
    out.append(f"    {seed} & " + " & ".join(f3(fa.loc[seed, k]) for k in D) + " & " + " & ".join(f"${d[k]:+.3f}$" for k in D) + r" \\")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

# Ablation arms, per seed, against the published full stack.
def piv(df, arm, col="rotation"):
    return df[df.arm == arm].pivot(index="seed", columns="domain", values=col)[D]
full = piv(pub, "unfrozen_aug")
a = piv(pd.read_csv(V / "arm_a_no_blur_noise.csv"), "unfrozen_aug_no_blur_noise")
b = piv(pd.read_csv(V / "arm_b_jitter_gamma_erasing.csv"), "unfrozen_aug_jitter_gamma")
aT = piv(pd.read_csv(V / "arm_a_no_blur_noise.csv"), "unfrozen_aug_no_blur_noise", "translation_relative")
bT = piv(pd.read_csv(V / "arm_b_jitter_gamma_erasing.csv"), "unfrozen_aug_jitter_gamma", "translation_relative")
ctl = pd.read_csv(V / "control_unfrozen_aug_seed42.csv").set_index("domain")
out.append(r"""\paragraph{Augmentation ablation.} Arm A is the full stack without blur and noise;
arm B keeps only brightness/contrast jitter, gamma and random erasing (no glare blob,
blur or noise). Both train layer 4 on seeds 42, 43 and 44 with the published
configuration, and are read as per-seed differences. They were trained with the
released code on an Apple M5 GPU, whereas the published arms were trained on Colab.
A control rerun of the full stack (seed 42) on the M5 differs from its published
counterpart by """ + f"{ctl.loc['synthetic','rotation']-full.loc[42,'synthetic']:+.3f}, {ctl.loc['lightbox','rotation']-full.loc[42,'lightbox']:+.3f} and {ctl.loc['sunlamp','rotation']-full.loc[42,'sunlamp']:+.3f}" + r"""\,rad in $\Erot$
(synthetic, lightbox, sunlamp): within seed variation, but on sunlamp as large as the
ablation effects themselves.

\begin{table}[ht]
  \setlength{\tabcolsep}{3pt}
  \caption{Augmentation ablation, $\Erot$ (rad) per seed and paired difference from
  the published full stack (layer 4, same seed). Last column: sunlamp $\Etrans$.}
  \label{tab:ablation}
  \centering
  \small
  \begin{tabular}{l c ccc ccc c}
    \toprule
    & & \multicolumn{3}{c}{$\Erot$} & \multicolumn{3}{c}{$\Delta\Erot$ vs.\ full stack} & Sunlamp \\
    \cmidrule(lr){3-5} \cmidrule(lr){6-8}
    Arm & Seed & Synth. & Lightbox & Sunlamp & Synth. & Lightbox & Sunlamp & $\Etrans$ \\
    \midrule""")
for name, x, xt in [("full stack", full, piv(pub, "unfrozen_aug", "translation_relative")), ("A: no blur/noise", a, aT), ("B: jitter, gamma, erasing", b, bT)]:
    for seed in (42, 43, 44):
        d = x.loc[seed] - full.loc[seed]
        diffs = " & ".join("--" if name == "full stack" else f"${d[k]:+.3f}$" for k in D)
        out.append(f"    {name if seed == 42 else ''} & {seed} & " + " & ".join(f3(x.loc[seed, k]) for k in D) + f" & {diffs} & {f3(xt.loc[seed, 'sunlamp'])} \\\\")
    out.append(r"    \midrule" if name != "B: jitter, gamma, erasing" else r"    \bottomrule")
out.append(r"""  \end{tabular}
\end{table}
""")

# AdaBN
ad = pd.read_csv(V / "adabn.csv")
n = ad.groupby("domain").n.first()
out.append(r"""\paragraph{Test-time BatchNorm adaptation (AdaBN).} For each layer-4 checkpoint we
re-estimate only layer 4's BatchNorm statistics, the ones fitted on synthetic data, from
the unlabelled images of one half of each real domain (a fixed random split by
filename), leave every weight untouched, and evaluate on the other half (""" + f"{n['lightbox']:,}".replace(",", "{,}") + " lightbox, " + f"{n['sunlamp']:,}".replace(",", "{,}") + r""" sunlamp
images). Coverage uses each arm's published synthetic-calibrated $\hat{q}$. This is
test-time adaptation: it uses target images, though not their labels. The frozen arms
are excluded, as all their statistics are ImageNet's.

\begin{table}[ht]
  \caption{AdaBN on the held-out half of each real domain: unadapted $\to$ adapted.
  Coverage is of the 90\% synthetic-calibrated rotation cone.}
  \label{tab:adabn}
  \centering
  \small
  \begin{tabular}{l c l ccc}
    \toprule
    Aug. & Seed & Domain & $\Erot$ & $\Etrans$ & Rotation coverage (\%) \\
    \midrule""")
ad["aug"] = ad.arm.str.startswith("unfrozen_aug")
ad["seed"] = ad.arm.str[-2:].astype(int)
for aug in (False, True):
    for seed in (42, 43, 44):
        for dom in ("lightbox", "sunlamp"):
            g = ad[(ad.aug == aug) & (ad.seed == seed) & (ad.domain == dom)].set_index("condition")
            u, v = g.loc["unadapted"], g.loc["adapted"]
            out.append(f"    {(r'\checkmark' if aug else '--') if (seed == 42 and dom == 'lightbox') else ''} & {seed if dom == 'lightbox' else ''} & {dom} & "
                       f"{f3(u.rotation)} $\\to$ {f3(v.rotation)} & {f3(u.translation_relative)} $\\to$ {f3(v.translation_relative)} & "
                       f"{100*u.coverage_rotation:.1f} $\\to$ {100*v.coverage_rotation:.1f} \\\\")
    if not aug:
        out.append(r"    \midrule")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

# Image statistics
st = pd.read_csv(V / "image_statistics.csv").set_index("pool")
labels = {"lightbox": "lightbox", "sunlamp": "sunlamp", "synthetic, no augmentation": "synthetic, no aug.",
          "synthetic, full stack": "synthetic, full stack", "synthetic, arm A (no blur/noise)": "synthetic, arm A",
          "synthetic, arm B (jitter + gamma + erasing)": "synthetic, arm B"}
out.append(r"""\paragraph{Image statistics.} Figure~\ref{fig:crops} compares pooled intensity
densities, which cannot separate a soft bright blob from a hard edge.
Table~\ref{tab:imagestats} adds spatial statistics, pooled over every image of each
split after the same resize and grayscale conversion (augmentation seed 7, as in
Figure~\ref{fig:crops}): the Sobel gradient magnitude, scaled so a full black-to-white
step reads 1; edge density, the share of pixels whose gradient exceeds 0.1; the
clipped fraction, the share of pixels at or above $254/255$; and the total variation
(TV) between pooled densities (48 intensity bins, 64 gradient bins). The intensity TV of
the full stack reproduces Figure~\ref{fig:crops} ($0.63$, $0.80$).

\begin{table}[ht]
  \caption{Pooled image statistics and total-variation distance to each real domain.}
  \label{tab:imagestats}
  \centering
  \small
  \begin{tabular}{l cc cc cc}
    \toprule
    & Edge & Clipped & \multicolumn{2}{c}{TV intensity} & \multicolumn{2}{c}{TV gradient} \\
    \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    Pool & density (\%) & (\%) & lightbox & sunlamp & lightbox & sunlamp \\
    \midrule""")
for pool, label in labels.items():
    r = st.loc[pool]
    out.append(f"    {label} & {100*r.edge_density:.1f} & {100*r.clipped_fraction:.2f} & {r.tv_intensity_lightbox:.2f} & {r.tv_intensity_sunlamp:.2f} & {r.tv_gradient_lightbox:.2f} & {r.tv_gradient_sunlamp:.2f} \\\\")
out.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
Path(__file__).with_name("appendix_revision.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote appendix_revision.tex")
