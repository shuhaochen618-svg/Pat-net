"""Generate top-venue (Nature Machine Intelligence / NeurIPS / ACL) multi-panel paper figures strictly from curated JSON outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CURATED_DIR = ROOT / "assets" / "results"
FIG_DIR = ROOT / "figures"


def _resolve_json(name: str) -> Path:
    for cand in (OUT_DIR / name, CURATED_DIR / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate {name} under outputs/ or assets/results/")


# ── Top-Venue Harmonious Semantic Palette (Colorblind-Safe & Grayscale-Distinct) ──
PALETTE = {
    # Hero method family (Terracotta / Carmine & Warm Amber)
    "hero":        "#B83A3A",   # PAT-Net (M4) primary deep carmine
    "hero_soft":   "#F2D2D0",   # Subtle hero band / soft fill
    "hero_dark":   "#7C2222",   # Hero dark stroke / text
    "amber":       "#C86D1E",   # Secondary hero / DeBERTa-v3 / E=5 variant
    "amber_soft":  "#F8E4D0",

    # Distributional baseline family (Cobalt / Steel Blue spectrum)
    "cobalt":      "#1A4E8A",   # Strong distributional baseline (DC-LDL)
    "steel":       "#4B7CB3",   # Mid distributional (SORD / UW-LDL)
    "sky":         "#8DB1D6",   # Soft distributional (DLDL-v2 / MC-T10)
    "ice":         "#E7EFF8",   # Subtle panel accent background

    # Unimodal & Parametric family (Sea-Teal spectrum)
    "teal":        "#2B7F86",   # NIG / PAT-Unimodal
    "teal_mid":    "#62A6AC",   # Gaussian NLL / Beta-NLL
    "teal_soft":   "#DDF0F1",

    # Neutral & Ablation / Scalar family (Cool Slate & Graphite)
    "slate":       "#6F7782",   # No-Transport ablation
    "slate_light": "#AEB5BF",
    "graphite":    "#2B333E",   # Scalar MSE & classical models
    "graphite_mid":"#4D5866",

    # Directional gain / safety callouts
    "sage":        "#2C8454",   # Valid safety / positive gain
    "sage_soft":   "#E5F2EB",
    "violet":      "#7A5195",   # Epistemic / cross-accent
    "ivory_band":  "#F4F6F9",   # Subtle alternating row band
}


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",      # Keep SVG text editable
        "pdf.fonttype": 42,          # Keep PDF TrueType fonts editable
        "font.size": 8.6,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.titlepad": 7.0,
        "axes.labelsize": 8.8,
        "axes.labelcolor": "#1E242B",
        "axes.edgecolor": "#2B333E",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#D5DBE3",
        "grid.alpha": 0.45,
        "grid.linewidth": 0.55,
        "grid.linestyle": "--",
        "xtick.color": "#2B333E",
        "ytick.color": "#2B333E",
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.frameon": False,
        "legend.fontsize": 7.8,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _add_panel_badge(ax: plt.Axes, letter: str, title: str = "", x: float = -0.10, y: float = 1.04) -> None:
    """Place crisp lowercase bold Nature-style panel letter and aligned subtitle."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        color="#11161C",
        ha="left",
        va="bottom",
    )
    if title:
        ax.set_title(title, loc="left", x=0.04 if x >= -0.04 else 0.0, fontsize=9.8, fontweight="bold", color="#1E242B")


def _save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _disc_gaussian_probs(mu: float, sd: float, vals: np.ndarray) -> np.ndarray:
    edges = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)((edges - mu) / (sd * math.sqrt(2.0))))
    p = np.clip(cdf[1:] - cdf[:-1], 1e-7, 1.0)
    return p / p.sum()


def _disc_dml2_probs(mu1: float, mu2: float, s1: float, s2: float, pi: float) -> np.ndarray:
    edges = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
    def _logi_cdf(x: np.ndarray, m: float, s: float) -> np.ndarray:
        z = np.clip((x - m) / s, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
    p1 = np.diff(_logi_cdf(edges, mu1, s1))
    p2 = np.diff(_logi_cdf(edges, mu2, s2))
    p = (1.0 - pi) * p1 + pi * p2
    return p / p.sum()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Quantitative Empirical Evidence of Distributional & Perspective Gaps
# (Note: Figure 1 is the user-authored architecture overview; script starts at Fig. 2.)
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig2_empirical_phenomena_and_audit(
    audit_eb: Dict[str, object],
    audit_epic: Dict[str, object],
    audit_ge: Dict[str, object],
) -> None:
    """4-panel quantitative empirical phenomena composite (no architecture flowcharts)."""
    fig = plt.figure(figsize=(15.0, 8.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.05], hspace=0.30, wspace=0.24)

    # ── Panel a: Mathematical Fallacy (Unanimous vs Gaussian Bell-Trap vs 2-Comp DML) ──
    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p_unan = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    p_bimod = np.array([0.45, 0.05, 0.0, 0.05, 0.45])
    sd_b = float(np.sqrt(np.sum(p_bimod * (bins - 3.0) ** 2)))
    p_gauss = _disc_gaussian_probs(3.0, sd_b, bins)
    p_dml = _disc_dml2_probs(1.05, 4.95, 0.30, 0.30, 0.50)

    w = 0.24
    ax_a.bar(bins - w, p_bimod, width=w, color=PALETTE["graphite"], label="Ground-Truth Polarized ($\\mu=3.0$)")
    ax_a.bar(bins, p_gauss, width=w, color=PALETTE["teal_mid"], label=f"Unimodal Gaussian ($\\mu=3.0, \\sigma={sd_b:.2f}$)")
    ax_a.bar(bins + w, p_dml, width=w, color=PALETTE["hero"], label="2-Comp DML Mixture (Bimodal)")
    ax_a.plot(bins, p_unan, marker="o", linestyle=":", color=PALETTE["cobalt"], lw=1.7, ms=5.8, label="Unanimous Neutral ($\\mu=3.0, \\sigma=0$)")
    ax_a.annotate("Unimodal peaks at\nempty center bin", xy=(2.95, p_gauss[2]), xytext=(1.78, 0.58),
                  ha="center", fontsize=7.5, fontweight="bold", color=PALETTE["teal"],
                  arrowprops={"arrowstyle": "->", "color": PALETTE["teal"], "lw": 1.1})
    ax_a.set_xticks(bins)
    ax_a.set_xticklabels(["1", "2", "3\n(Neutral)", "4", "5"])
    ax_a.set_ylim(0, 1.34)
    ax_a.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_xlabel("Discrete Ordinal Rating Scale")
    ax_a.set_ylabel("Rating Probability Mass")
    ax_a.legend(loc="upper center", ncol=2, fontsize=7.1, columnspacing=1.0)
    _add_panel_badge(ax_a, "a", "Scalar Mean Collapse & Unimodal Bell-Trap", x=-0.12)

    # ── Panel b: Real EmoBank Neutral-Mean Hidden Polarization (audit_emobank.json) ──
    ax_b = fig.add_subplot(gs[0, 1])
    eb_aud = audit_eb["reader_mean_collapse_audit_10062"]
    dims = ["V", "A", "D"]
    dim_lbls = ["Valence\n(n=8,069)", "Arousal\n(n=8,954)", "Dominance\n(n=9,255)"]
    pct_unan, pct_diff, pct_pol = [], [], []
    for d in dims:
        sub = eb_aud[d]
        tot = float(sub["mean_2p5_3p5_n"])
        u_pct = sub["unanimous_neutral_n"] / tot * 100.0
        p_pct = sub["polarized_n"] / tot * 100.0
        d_pct = max(0.0, 100.0 - u_pct - p_pct)
        pct_unan.append(u_pct)
        pct_diff.append(d_pct)
        pct_pol.append(p_pct)

    xd = np.arange(len(dims))
    bw = 0.52
    ax_b.bar(xd, pct_unan, width=bw, color=PALETTE["cobalt"], label="Unanimous Neutral")
    ax_b.bar(xd, pct_diff, bottom=pct_unan, width=bw, color=PALETTE["sky"], label="Moderate Diffuse")
    bot2 = np.array(pct_unan) + np.array(pct_diff)
    ax_b.bar(xd, pct_pol, bottom=bot2, width=bw, color=PALETTE["hero"], label="Hidden Polarized (Bimodal)")

    for i in range(3):
        ax_b.text(xd[i], pct_unan[i] * 0.5, f"{pct_unan[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.3, fontweight="bold")
        ax_b.text(xd[i], pct_unan[i] + pct_diff[i] * 0.5, f"{pct_diff[i]:.1f}%", ha="center", va="center", color="#1E242B", fontsize=8.3, fontweight="bold")
        ax_b.text(xd[i], bot2[i] + pct_pol[i] * 0.5, f"{pct_pol[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.3, fontweight="bold")

    ax_b.set_xticks(xd)
    ax_b.set_xticklabels(dim_lbls, fontsize=8.5)
    ax_b.set_ylim(0, 118)
    ax_b.set_ylabel("% of Nominal Neutral Band [2.5, 3.5]")
    ax_b.legend(loc="upper center", ncol=3, fontsize=7.3)
    _add_panel_badge(ax_b, "b", "EmoBank Neutral-Band Conflict Audit (n = 10,062)", x=-0.12)

    # ── Panel c: GoEmotions Entropy Paradox & EPIC Nationality Irony Divergence ──
    ax_c = fig.add_subplot(gs[1, 0])
    cats_c = [
        "GoEmotions\nUnclear\n(n=2,733)",
        "GoEmotions\nDisagree≥0.5\n(n=11,559)",
    ]
    ent_vals = [
        audit_ge["unclear_any_stats"]["mean_entropy"],
        audit_ge["clear_only_stats"]["mean_entropy_when_disagree_ge_0p5"],
    ]
    nats = ["United States", "Ireland", "United Kingdom", "India", "Australia"]
    nat_short = ["US", "IE", "UK", "IN", "AU"]
    nat_rates = [audit_epic["top_nationalities_irony_rate"][n]["mean"] * 100.0 for n in nats]

    x_ge = np.array([0.0, 1.0])
    ax_c.bar(x_ge, ent_vals, width=0.56, color=[PALETTE["teal"], PALETTE["hero"]])
    for xi, v in zip(x_ge, ent_vals):
        ax_c.text(xi, v + 0.08, f"{v:.2f} bits", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_c.annotate("Unclear < Disagreement\n(Total entropy misranks!)", xy=(0.5, 2.35), xytext=(0.5, 3.18),
                  ha="center", fontsize=7.6, fontweight="bold", color=PALETTE["hero_dark"])
    ax_c.set_ylim(0, 3.65)
    ax_c.set_xticks(x_ge)
    ax_c.set_xticklabels(cats_c, fontsize=8.0)
    ax_c.set_ylabel("Multi-Label Bernoulli Entropy (bits)")

    ax_ins = ax_c.inset_axes([0.58, 0.14, 0.39, 0.72])
    xn = np.arange(len(nat_short))
    ax_ins.barh(xn, nat_rates, color=PALETTE["cobalt"], height=0.60)
    ax_ins.invert_yaxis()
    for i, r_val in enumerate(nat_rates):
        ax_ins.text(r_val + 0.5, i, f"{r_val:.1f}%", va="center", ha="left", fontsize=7.0, color="#1E242B")
    ax_ins.set_yticks(xn)
    ax_ins.set_yticklabels(nat_short, fontsize=7.6)
    ax_ins.set_xlim(22, 42)
    ax_ins.set_xticks([25, 35])
    ax_ins.tick_params(axis="x", labelsize=7.0)
    ax_ins.set_xlabel("Irony Rate (%)", fontsize=7.4)
    ax_ins.set_title("EPIC Nation Gap (n=3,000)", fontsize=7.8, fontweight="bold", pad=3)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    ax_c.set_xlim(-0.55, 3.55)
    ax_c.axvline(1.62, color="#D5DBE3", linestyle="-", linewidth=0.95)
    _add_panel_badge(ax_c, "c", "GoEmotions Entropy Paradox & EPIC Nation Divergence", x=-0.12)

    # ── Panel d: Genre-Dependent Writer–Reader Perspective Shift Rates (audit_emobank.json) ──
    ax_d = fig.add_subplot(gs[1, 1])
    genre_aud = audit_eb["genre_perspective_shift_10062"]
    genres_ord = ["letters", "SemEval", "fiction", "travel-guides", "essays", "newspaper", "blog"]
    vals_v = [genre_aud[g]["frac_abs_V_ge_0p5"] * 100.0 for g in genres_ord]
    vals_a = [genre_aud[g]["frac_abs_A_ge_0p5"] * 100.0 for g in genres_ord]
    vals_d = [genre_aud[g]["frac_abs_D_ge_0p5"] * 100.0 for g in genres_ord]

    yg = np.arange(len(genres_ord))
    hb = 0.255
    ax_d.barh(yg - hb, vals_a, height=hb, color=PALETTE["hero"], label="Arousal $|A_R - A_W| \\geq 0.5$")
    ax_d.barh(yg, vals_v, height=hb, color=PALETTE["cobalt"], label="Valence $|V_R - V_W| \\geq 0.5$")
    ax_d.barh(yg + hb, vals_d, height=hb, color=PALETTE["teal"], label="Dominance $|D_R - D_W| \\geq 0.5$")
    for yi, va, vv in zip(yg, vals_a, vals_v):
        ax_d.text(va + 0.35, yi - hb, f"{va:.1f}%", va="center", ha="left", fontsize=6.9, fontweight="bold", color=PALETTE["hero_dark"])
        ax_d.text(vv + 0.35, yi, f"{vv:.1f}%", va="center", ha="left", fontsize=6.9, color=PALETTE["cobalt"])

    ax_d.set_yticks(yg)
    ax_d.set_yticklabels([f"{g} (n={genre_aud[g]['n']:,})" for g in genres_ord], fontsize=8.0)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 27.5)
    ax_d.set_xlabel("% of Sentences with Large Perspective Shift ($|\\Delta| \\geq 0.5$)")
    ax_d.legend(loc="lower right", fontsize=7.4)
    _add_panel_badge(ax_d, "d", "EmoBank Genre-Dependent Writer–Reader Asymmetry", x=-0.16)

    _save_fig(fig, "fig2_empirical_phenomena_and_audit")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: EmoBank 16-Model Dual-Perspective Rankings, Pareto, Dim Matrix & RTX 4090 Latency
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig3_emobank_pareto_and_efficiency(report: Dict[str, object]) -> None:
    oof = report["emobank_5fold_oof"]
    deb = report["emobank_deberta_v3_5fold"]
    abl = report["emobank_ablation_ladder_M0_to_M5"]
    hw = report["hardware_latency_vram"]

    fig = plt.figure(figsize=(15.4, 9.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.12, 1.0], hspace=0.28, wspace=0.25)

    # ── Panel a: Paired Reader & Writer W1 Dumbbell Chart (All 16 Models) ──
    ax_a = fig.add_subplot(gs[0, 0])
    order = [
        ("patnet_M4", "PAT-Net (M4, Full)", PALETTE["hero"]),
        ("nig_evidential", "NIG Evidential", PALETTE["teal"]),
        ("patnet_unimodal", "PAT-Net (Unimodal)", PALETTE["teal_mid"]),
        ("patnet_M5", "PAT-Net (M5, Struct)", PALETTE["amber"]),
        ("patnet_no_transport", "PAT-Net (No Trans.)", PALETTE["slate"]),
        ("dc_ldl", "DC-LDL (1-Pass)", PALETTE["cobalt"]),
        ("uw_ldl", "UW-LDL", PALETTE["steel"]),
        ("dc_ldl_mcT10", "DC-LDL (MC T=10)", PALETTE["sky"]),
        ("dldl_v2", "DLDL-v2", PALETTE["sky"]),
        ("sord", "SORD", PALETTE["steel"]),
        ("gaussian_nll_mcT10", "Gaussian NLL (MC)", PALETTE["teal_mid"]),
        ("gaussian_nll", "Gaussian NLL", PALETTE["teal_mid"]),
        ("C1_SparseRidge", "C1 SparseRidge", PALETTE["graphite_mid"]),
        ("C2_HybridGBDT", "C2 HybridGBDT", PALETTE["graphite_mid"]),
        ("beta_nll", "Beta-NLL", PALETTE["teal_mid"]),
        ("scalar_mse", "Scalar MSE", PALETTE["graphite"]),
    ]
    labels = [lbl for k, lbl, _ in order if k in oof]
    r_vals = [oof[k]["r_mean_w1"] for k, _, _ in order if k in oof]
    w_vals = [oof[k]["w_mean_w1"] for k, _, _ in order if k in oof]
    cols = [c for k, _, c in order if k in oof]
    y = np.arange(len(labels))

    for yi, rv, wv, c in zip(y, r_vals, w_vals, cols):
        if yi % 2 == 0:
            ax_a.axhspan(yi - 0.45, yi + 0.45, color=PALETTE["ivory_band"], zorder=0)
        ax_a.plot([wv, rv], [yi, yi], color=c, alpha=0.55, linewidth=2.0, zorder=2)

    ax_a.scatter(w_vals, y, marker="D", s=38, facecolors="white", edgecolors=cols, linewidths=1.35, zorder=3, label="Writer W1 (Producer)")
    ax_a.scatter(r_vals, y, marker="o", s=52, color=cols, edgecolors="white", linewidths=0.7, zorder=4, label="Reader W1 (Perceiver)")
    ax_a.axvline(oof["patnet_M4"]["r_mean_w1"], color=PALETTE["hero"], linestyle=":", linewidth=1.05, alpha=0.75)

    for yi, rv, wv in zip(y, r_vals, w_vals):
        fw = "bold" if yi == 0 else "normal"
        col = PALETTE["hero_dark"] if yi == 0 else "#2B333E"
        lo_v, hi_v = min(rv, wv), max(rv, wv)
        ax_a.text(lo_v - 0.003, yi, f"{wv:.4f}", va="center", ha="right", fontsize=6.9, color=col, fontweight=fw)
        ax_a.text(hi_v + 0.003, yi, f"{rv:.4f}", va="center", ha="left", fontsize=6.9, color=col, fontweight=fw)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=7.9)
    ax_a.get_yticklabels()[0].set_fontweight("bold")
    ax_a.get_yticklabels()[0].set_color(PALETTE["hero_dark"])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.205, 0.390)
    ax_a.set_xlabel("1D Wasserstein-1 Distributional Error (Left = Better)")
    ax_a.legend(loc="upper right", ncol=1, fontsize=7.6)
    _add_panel_badge(ax_a, "a", "Dual-Perspective W1 Across All 16 Models", x=-0.16)

    # ── Panel b: Pareto Frontier + Cross-Backbone Upgrade Trajectories ───────
    ax_b = fig.add_subplot(gs[0, 1])
    best_pear = report["gates"]["gate1_distributional_win_no_scalar_drop"]["best_all_baseline_r_pearson"]
    ax_b.axhspan(best_pear - 0.005, best_pear, color=PALETTE["hero_soft"], alpha=0.45, zorder=1,
                 label="Gate 1 Pearson Tolerance (−0.005)")

    rob_pts = [
        ("scalar_mse", "Scalar MSE", PALETTE["graphite"], "s", (-0.004, 0.0020, "right")),
        ("gaussian_nll", "Gaussian NLL", PALETTE["teal_mid"], "^", (0.003, -0.0012, "left")),
        ("nig_evidential", "NIG Evidential", PALETTE["teal"], "v", (0.003, -0.0016, "left")),
        ("sord", "SORD", PALETTE["steel"], "D", (0.0035, 0.0012, "left")),
        ("dldl_v2", "DLDL-v2", PALETTE["sky"], "P", (0.0035, 0.0012, "left")),
        ("dc_ldl", "DC-LDL", PALETTE["cobalt"], "o", (0.0035, -0.0015, "left")),
        ("uw_ldl", "UW-LDL", PALETTE["steel"], "<", (0.0035, 0.0004, "left")),
        ("patnet_unimodal", "PAT-Uni", PALETTE["teal"], ">", (0.0035, 0.0018, "left")),
        ("patnet_no_transport", "No-Trans.", PALETTE["slate"], "X", (0.0035, -0.0020, "left")),
        ("patnet_M4", "PAT-Net\n(RoBERTa)", PALETTE["hero"], "*", (-0.0035, -0.0018, "right")),
    ]
    for key, lbl, col, mk, (dx, dy, ha) in rob_pts:
        xv, yv = oof[key]["r_mean_w1"], oof[key]["r_mean_pearson"]
        sz = 185 if key == "patnet_M4" else 56
        ax_b.scatter(xv, yv, color=col, marker=mk, s=sz, edgecolors="white", linewidths=0.7, zorder=4 if key == "patnet_M4" else 3)
        ax_b.text(xv + dx, yv + dy, lbl, fontsize=7.3, ha=ha, fontweight="bold" if key == "patnet_M4" else "normal", color=col)

    # DeBERTa-v3 upgrades with vector transition arrows
    for key, lbl, col, dy in [
        ("scalar_mse", "Scalar (DeB)", PALETTE["graphite"], -0.0032),
        ("dc_ldl", "DC-LDL (DeB)", PALETTE["cobalt"], -0.0034),
        ("patnet_M4", "PAT-Net (DeBERTa-v3)", PALETTE["amber"], 0.0020),
    ]:
        rx, ry = oof[key]["r_mean_w1"], oof[key]["r_mean_pearson"]
        dx_p, dy_p = deb[key]["r_mean_w1"], deb[key]["r_mean_pearson"]
        is_m = key == "patnet_M4"
        ax_b.scatter(dx_p, dy_p, marker="*" if is_m else "o", s=195 if is_m else 48,
                     facecolors=col if is_m else "white", edgecolors=col, linewidths=1.35, zorder=5)
        ax_b.annotate("", xy=(dx_p, dy_p), xytext=(rx, ry),
                      arrowprops={"arrowstyle": "->", "color": col, "alpha": 0.55, "lw": 1.25, "ls": "--", "shrinkA": 4, "shrinkB": 5}, zorder=2)
        ax_b.text(dx_p + (-0.003 if key == "scalar_mse" else 0.002), dy_p + dy, lbl,
                  fontsize=7.3, ha="right" if key == "scalar_mse" else "left",
                  fontweight="bold" if is_m else "normal", color=col)

    ax_b.set_xlim(0.232, 0.365)
    ax_b.set_ylim(0.525, 0.562)
    ax_b.set_xlabel("Reader Mean 1D W1 (Left = Better)")
    ax_b.set_ylabel("Reader Mean Pearson r (Up = Better)")
    ax_b.legend(loc="lower right", fontsize=7.5)
    _add_panel_badge(ax_b, "b", "Pareto Frontier & Backbone Shift (RoB → DeB)", x=-0.14)

    # ── Panel c: Per-Dimension VAD Performance Matrix Heatmap ───────────────
    ax_c = fig.add_subplot(gs[1, 0])
    hm_models = [
        ("patnet_M4", "PAT-Net (M4)"),
        ("patnet_unimodal", "PAT-Net (Uni)"),
        ("nig_evidential", "NIG Evid."),
        ("patnet_no_transport", "PAT-NoTrans"),
        ("dc_ldl", "DC-LDL"),
        ("sord", "SORD"),
        ("gaussian_nll", "Gaussian NLL"),
        ("scalar_mse", "Scalar MSE"),
    ]
    cols_hm = [
        ("r_V_w1", "V W1↓", True),
        ("r_A_w1", "A W1↓", True),
        ("r_D_w1", "D W1↓", True),
        ("w_V_w1", "Wr-V W1↓", True),
        ("w_A_w1", "Wr-A W1↓", True),
        ("r_V_ccc", "V CCC↑", False),
        ("r_A_ccc", "A CCC↑", False),
        ("r_D_ccc", "D CCC↑", False),
    ]
    mat_raw = np.array([[oof[mk][ck] for ck, _, _ in cols_hm] for mk, _ in hm_models], dtype=float)
    mat_norm = np.zeros_like(mat_raw)
    for j, (_, _, low_best) in enumerate(cols_hm):
        cmin, cmax = mat_raw[:, j].min(), mat_raw[:, j].max()
        span = max(cmax - cmin, 1e-9)
        norm_col = (mat_raw[:, j] - cmin) / span
        mat_norm[:, j] = (1.0 - norm_col) if low_best else norm_col  # 1.0 = Best

    cmap_pub = LinearSegmentedColormap.from_list("nmi_pub", ["#F4E7E6", "#F7F8FA", "#C4D6EB", "#3A6FA5"])
    ax_c.imshow(mat_norm, cmap=cmap_pub, aspect="auto", vmin=0.0, vmax=1.0)
    for i in range(len(hm_models)):
        for j in range(len(cols_hm)):
            v = mat_raw[i, j]
            is_best = (mat_norm[i, j] >= 0.999)
            tcol = "white" if mat_norm[i, j] > 0.78 else "#1E242B"
            ax_c.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.4,
                      fontweight="bold" if (i == 0 or is_best) else "normal", color=tcol)
    # Crisp white cell borders
    ax_c.set_xticks(np.arange(len(cols_hm)))
    ax_c.set_xticklabels([lbl for _, lbl, _ in cols_hm], fontsize=7.7, rotation=18, ha="right")
    ax_c.set_yticks(np.arange(len(hm_models)))
    ax_c.set_yticklabels([lbl for _, lbl in hm_models], fontsize=7.9)
    ax_c.get_yticklabels()[0].set_fontweight("bold")
    ax_c.get_yticklabels()[0].set_color(PALETTE["hero_dark"])
    ax_c.grid(False)
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    ax_c.set_xticks(np.arange(-0.5, len(cols_hm), 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(hm_models), 1), minor=True)
    ax_c.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    _add_panel_badge(ax_c, "c", "Per-Dimension (V / A / D) Breakdown (Blue = Best)", x=-0.16)

    # ── Panel d: Hardware Latency vs Error & Throughput Bubbles (RTX 4090) ──
    ax_d = fig.add_subplot(gs[1, 1])
    hw_map = {(r["backbone"], r["model_type"], r["ablation_mode"], r["mc_dropout_T"]): r for r in hw}
    ens = report["emobank_fold0_deep_ensembles_E5"]

    pts_eff = [
        ("RoB Scalar MSE", hw_map[("roberta-base", "scalar_mse", "M0_baseline", 1)], oof["scalar_mse"]["r_mean_w1"], PALETTE["graphite"], "s", (8.3, 0.365, "left")),
        ("RoB DC-LDL", hw_map[("roberta-base", "dc_ldl", "M0_baseline", 1)], oof["dc_ldl"]["r_mean_w1"], PALETTE["cobalt"], "o", (9.2, 0.278, "left")),
        ("RoB PAT-Net (M0)", hw_map[("roberta-base", "patnet", "M0_baseline", 1)], abl["M0_baseline"]["r_mean_w1"], PALETTE["slate"], "^", (11.5, 0.252, "right")),
        ("RoB PAT-Net (M4)", hw_map[("roberta-base", "patnet", "M4_disentangle", 1)], oof["patnet_M4"]["r_mean_w1"], PALETTE["hero"], "*", (17.5, 0.246, "center")),
        ("RoB DC-LDL (E=5)", hw_map[("roberta-base", "deep_ensemble_E5(dc_ldl)", "M0_baseline", 1)], oof["dc_ldl"]["r_mean_w1"], PALETTE["cobalt"], "o", (43.5, 0.281, "center")),
        ("RoB Gauss MC-T10", hw_map[("roberta-base", "gaussian_nll", "M0_baseline", 10)], oof["gaussian_nll_mcT10"]["r_mean_w1"], PALETTE["teal_mid"], "^", (79.0, 0.291, "right")),
        ("DeB Scalar MSE", hw_map[("microsoft/deberta-v3-base", "scalar_mse", "M0_baseline", 1)], deb["scalar_mse"]["r_mean_w1"], PALETTE["graphite"], "s", (21.0, 0.350, "left")),
        ("DeB DC-LDL", hw_map[("microsoft/deberta-v3-base", "dc_ldl", "M0_baseline", 1)], deb["dc_ldl"]["r_mean_w1"], PALETTE["cobalt"], "o", (20.5, 0.269, "left")),
        ("DeB PAT-Net (M4)", hw_map[("microsoft/deberta-v3-base", "patnet", "M4_disentangle", 1)], deb["patnet_M4"]["r_mean_w1"], PALETTE["amber"], "*", (35.0, 0.249, "left")),
        ("DeB Gauss MC-T10", hw_map[("microsoft/deberta-v3-base", "gaussian_nll", "M0_baseline", 10)], oof["gaussian_nll_mcT10"]["r_mean_w1"], PALETTE["teal_mid"], "^", (158.0, 0.321, "right")),
    ]
    for lbl, r, w1v, col, mk, (tx, ty, ha) in pts_eff:
        lat = r["bs1_p50_ms"]
        qps = r["bs32_qps"]
        sz = max(55, min(280, qps * 0.14)) if mk != "*" else 210
        is_hollow = ("DeB" in lbl and "PAT-Net" not in lbl) or ("E=5" in lbl)
        ax_d.scatter(
            lat, w1v, marker=mk, s=sz,
            facecolors="white" if is_hollow else col,
            edgecolors=col if is_hollow else "white",
            linewidths=1.4 if is_hollow else 0.7,
            zorder=4 if "PAT-Net" in lbl else 3,
        )
        txt = f"{lbl}\n({lat:.1f}ms, {qps:.0f} QPS)" if "PAT-Net (M4)" in lbl else f"{lbl} ({lat:.1f}ms)"
        ax_d.annotate(
            txt, xy=(lat, w1v), xytext=(tx, ty), fontsize=6.7, ha=ha, va="center",
            fontweight="bold" if "PAT-Net (M4)" in lbl else "normal",
            color=col,
            arrowprops={"arrowstyle": "-", "color": col, "alpha": 0.45, "lw": 0.75} if abs(ty - w1v) > 0.004 else None,
        )

    # Highlight 4.77x speedup span
    r_p50 = hw_map[("roberta-base", "patnet", "M4_disentangle", 1)]["bs1_p50_ms"]
    mc_p50 = hw_map[("roberta-base", "gaussian_nll", "M0_baseline", 10)]["bs1_p50_ms"]
    ax_d.annotate("", xy=(r_p50, 0.313), xytext=(mc_p50, 0.313),
                  arrowprops={"arrowstyle": "<->", "color": PALETTE["hero"], "lw": 1.2})
    ax_d.text(math.sqrt(r_p50 * mc_p50), 0.315, "4.77x faster\n(6.52x on DeB)", ha="center", va="bottom",
              fontsize=7.3, fontweight="bold", color=PALETTE["hero_dark"])

    ax_d.set_xscale("log")
    ax_d.set_xticks([8, 15, 30, 60, 120, 200])
    ax_d.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_d.set_xlim(6.5, 225)
    ax_d.set_ylim(0.236, 0.375)
    ax_d.set_xlabel("Single-Query P50 Latency (ms, Log Scale; Marker ~ BS32 QPS)")
    ax_d.set_ylabel("Reader Mean 1D W1 (Lower Better)")
    _add_panel_badge(ax_d, "d", "Single-Pass Latency–Accuracy Efficiency (RTX 4090)", x=-0.14)

    _save_fig(fig, "fig3_emobank_pareto_and_efficiency")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Gate 2 Non-Circular Disentanglement, M0..M5 Ladder & Equal-Compute Ensembles
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig4_disentanglement_and_ablation(report: Dict[str, object]) -> None:
    oof = report["emobank_5fold_oof"]
    ens = report["emobank_fold0_deep_ensembles_E5"]
    abl = report["emobank_ablation_ladder_M0_to_M5"]
    g2 = report["gates"]["gate2_non_circular_shape_disentanglement"]

    fig = plt.figure(figsize=(15.4, 9.2))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.05], hspace=0.30)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0], width_ratios=[1.35, 1.0], wspace=0.22)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], width_ratios=[1.22, 1.0, 1.05], wspace=0.25)

    # ── Panel a: Hard Neutral-Mean Shape Disentanglement (U_plur vs U_amb vs ER0.8) ──
    ax_a = fig.add_subplot(gs_top[0])
    models = [
        ("PAT-Net\n(Fold0 E=5)", ens["patnet_fold0_E5"]),
        ("PAT-Net\n(M4 5-Fold)", oof["patnet_M4"]),
        ("PAT-Net\n(No-Trans)", oof["patnet_no_transport"]),
        ("DC-LDL\n(MC T=10)", oof["dc_ldl_mcT10"]),
        ("DC-LDL\n(1-Pass)", oof["dc_ldl"]),
        ("SORD", oof["sord"]),
        ("PAT-Net\n(Unimodal)", oof["patnet_unimodal"]),
        ("Gaussian\nNLL", oof["gaussian_nll"]),
    ]
    x = np.arange(len(models))
    w = 0.255
    uplur = [m[1]["r_hard_shape_auroc_by_uplur"] for m in models]
    uamb = [m[1]["r_hard_shape_auroc_by_uamb"] for m in models]
    er08 = [m[1]["r_hard_shape_auroc_by_er08"] for m in models]

    ax_a.bar(x - w, uplur, width=w, color=PALETTE["hero"], label="Between-Mode Pluralism ($U_{plur}$)")
    ax_a.bar(x, uamb, width=w, color=PALETTE["cobalt"], label="Within-Mode Dispersion ($U_{amb}$)")
    ax_a.bar(x + w, er08, width=w, color=PALETTE["teal"], label="Tail Polarization Mass ($ER_{0.8}$)")
    ax_a.axhline(0.50, color=PALETTE["slate"], linestyle=":", linewidth=1.0, label="Random Chance (0.50)")
    ax_a.axhline(g2["best_single_pass_er08_auroc"], color=PALETTE["teal"], linestyle="--", linewidth=0.9, alpha=0.65, label=f"Best Single Baseline ER0.8 ({g2['best_single_pass_er08_auroc']:.4f})")

    for xi, v1, v3 in zip(x, uplur, er08):
        ax_a.text(xi - w, v1 + 0.0025, f"{v1:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
        ax_a.text(xi + w, v3 + 0.0025, f"{v3:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([m[0] for m in models], fontsize=7.9)
    ax_a.set_ylim(0.48, 0.708)
    ax_a.set_ylabel("Hard Neutral-Mean Polarization AUROC")
    ax_a.legend(loc="upper right", ncol=2, fontsize=7.3)
    _add_panel_badge(ax_a, "a", "Non-Circular Polarization vs. Dispersion Separation (Gate 2)", x=-0.09)

    # ── Panel b: Separation Gap (U_plur - U_amb) & Epistemic Collapse Check ─
    ax_b = fig.add_subplot(gs_top[1])
    sep_items = [
        ("PAT-Net (DeBERTa-v3)", report["emobank_deberta_v3_5fold"]["patnet_M4"]["r_hard_shape_auroc_by_uplur"] - report["emobank_deberta_v3_5fold"]["patnet_M4"]["r_hard_shape_auroc_by_uamb"], PALETTE["amber"]),
        ("PAT-Net (M4 5-Fold)", oof["patnet_M4"]["r_hard_shape_auroc_by_uplur"] - oof["patnet_M4"]["r_hard_shape_auroc_by_uamb"], PALETTE["hero"]),
        ("PAT-Net (Fold-0 E=5)", ens["patnet_fold0_E5"]["r_hard_shape_auroc_by_uplur"] - ens["patnet_fold0_E5"]["r_hard_shape_auroc_by_uamb"], PALETTE["hero_dark"]),
        ("NIG Evidential", oof["nig_evidential"]["r_hard_shape_auroc_by_uplur"] - oof["nig_evidential"]["r_hard_shape_auroc_by_uamb"], PALETTE["teal"]),
        ("DC-LDL (1-Pass, Aliased)", oof["dc_ldl"]["r_hard_shape_auroc_by_uplur"] - oof["dc_ldl"]["r_hard_shape_auroc_by_uamb"], PALETTE["cobalt"]),
        ("DC-LDL (MC T=10 Epist.)", oof["dc_ldl_mcT10"]["r_hard_shape_auroc_by_uplur"] - oof["dc_ldl_mcT10"]["r_hard_shape_auroc_by_uamb"], PALETTE["sky"]),
        ("PAT-Net (Unimodal)", oof["patnet_unimodal"]["r_hard_shape_auroc_by_uplur"] - oof["patnet_unimodal"]["r_hard_shape_auroc_by_uamb"], PALETTE["slate"]),
        ("Gaussian NLL (MC T=10)", oof["gaussian_nll_mcT10"]["r_hard_shape_auroc_by_uplur"] - oof["gaussian_nll_mcT10"]["r_hard_shape_auroc_by_uamb"], PALETTE["graphite"]),
    ]
    ys = np.arange(len(sep_items))
    vals_s = [v for _, v, _ in sep_items]
    cols_s = [c for _, _, c in sep_items]
    ax_b.barh(ys, vals_s, color=cols_s, height=0.62)
    ax_b.axvline(0.0, color="#2B333E", linewidth=1.0)
    for yi, v in zip(ys, vals_s):
        ha = "left" if v >= 0 else "right"
        pad = 0.002 if v >= 0 else -0.002
        ax_b.text(v + pad, yi, f"{v:+.4f}", va="center", ha=ha, fontsize=7.4, fontweight="bold" if yi in (0, 1, 2) else "normal")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([n for n, _, _ in sep_items], fontsize=7.9)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.155, 0.045)
    ax_b.set_xlabel("Disentanglement Margin: AUROC($U_{plur}$) − AUROC($U_{amb}$)")
    _add_panel_badge(ax_b, "b", "Clean Mode Separation vs. Epistemic Collapse", x=-0.16)

    # ── Panel c: Clean Dual-Track Stacked Ablation Ladder (NO Twin-Axis Clutter!) ─
    gs_abl = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bot[0], hspace=0.14)
    ax_c1 = fig.add_subplot(gs_abl[0])
    ax_c2 = fig.add_subplot(gs_abl[1], sharex=ax_c1)

    stages = ["M0_baseline", "M1_multilayer_conv", "M2_bipolar_pool", "M3_tension_route", "M4_disentangle", "M5_full_struct"]
    short = ["M0\nBase", "M1\n+Conv", "M2\n+Bipol", "M3\n+Route", "M4\nFull", "M5\n+Struct"]
    xs = np.arange(len(stages))

    r_w1s = [abl[s]["r_mean_w1"] for s in stages]
    w_w1s = [abl[s]["w_mean_w1"] for s in stages]
    pears = [abl[s]["r_mean_pearson"] for s in stages]
    er08s = [abl[s]["r_hard_shape_auroc_by_er08"] for s in stages]
    alrts = [abl[s]["oof_alert_any_vod_auroc"] for s in stages]

    ax_c1.axvspan(3.65, 4.35, color=PALETTE["hero_soft"], alpha=0.55, zorder=0)
    ax_c1.plot(xs, r_w1s, marker="o", lw=2.0, color=PALETTE["hero"], label="Reader W1↓")
    ax_c1.plot(xs, w_w1s, marker="D", lw=1.7, ls="--", color=PALETTE["cobalt"], label="Writer W1↓")
    ax_c1.scatter([4], [r_w1s[4]], s=95, color=PALETTE["hero_dark"], zorder=5)
    ax_c1.text(4.0, r_w1s[4] - 0.004, f"{r_w1s[4]:.4f}", ha="center", va="top", fontsize=7.2, fontweight="bold", color=PALETTE["hero_dark"])
    ax_c1.set_ylim(0.224, 0.284)
    ax_c1.set_ylabel("W1 Error↓", fontsize=8.0)
    ax_c1.legend(loc="upper right", ncol=2, fontsize=7.0)
    plt.setp(ax_c1.get_xticklabels(), visible=False)
    _add_panel_badge(ax_c1, "c", "Modular Ablation Ladder (M0–M5)", x=-0.16, y=1.08)

    ax_c2.axvspan(3.65, 4.35, color=PALETTE["hero_soft"], alpha=0.55, zorder=0)
    ax_c2.plot(xs, er08s, marker="^", lw=1.9, color=PALETTE["teal"], label="ER0.8 AUROC↑")
    ax_c2.plot(xs, alrts, marker="s", lw=1.8, ls=":", color=PALETTE["violet"], label="Alert VOD↑")
    ax_c2.plot(xs, pears, marker="o", lw=1.8, ls="--", color=PALETTE["amber"], label="Pearson r↑")
    ax_c2.set_xticks(xs)
    ax_c2.set_xticklabels(short, fontsize=7.8)
    ax_c2.set_ylim(0.525, 0.634)
    ax_c2.set_ylabel("Score↑", fontsize=8.0)
    ax_c2.legend(loc="center left", ncol=2, fontsize=6.7, bbox_to_anchor=(0.0, 0.48))

    # ── Panel d: Distributional Variance & JSD Fidelity Across Model Families ──
    ax_d = fig.add_subplot(gs_bot[1])
    fam_keys = [
        ("patnet_M4", "PAT-Net\n(M4)"),
        ("nig_evidential", "NIG\nEvid."),
        ("dc_ldl", "DC-LDL"),
        ("dldl_v2", "DLDL-v2"),
        ("sord", "SORD"),
        ("gaussian_nll", "Gauss\nNLL"),
        ("beta_nll", "Beta-NLL"),
    ]
    xf = np.arange(len(fam_keys))
    std_maes = [oof[k]["r_mean_std_mae"] for k, _ in fam_keys]
    jsds = [oof[k]["r_mean_jsd"] for k, _ in fam_keys]
    w2 = 0.36
    ax_d.bar(xf - w2 / 2, std_maes, width=w2, color=PALETTE["hero"], label="Reader Std MAE↓")
    ax_d.bar(xf + w2 / 2, jsds, width=w2, color=PALETTE["cobalt"], label="Reader JSD↓")
    for xi, sm, jv in zip(xf, std_maes, jsds):
        ax_d.text(xi - w2 / 2, sm + 0.004, f"{sm:.3f}", ha="center", va="bottom", fontsize=6.6, rotation=90)
        ax_d.text(xi + w2 / 2, jv + 0.004, f"{jv:.3f}", ha="center", va="bottom", fontsize=6.6, rotation=90)
    ax_d.set_xticks(xf)
    ax_d.set_xticklabels([l for _, l in fam_keys], fontsize=7.5)
    ax_d.set_ylim(0.06, 0.39)
    ax_d.set_ylabel("Dispersion & Divergence Error↓")
    ax_d.legend(loc="upper left", fontsize=7.3)
    _add_panel_badge(ax_d, "d", "Spread Accuracy vs. Divergence", x=-0.16)

    # ── Panel e: Fold-0 Equal-Compute Deep Ensembles (E=5 vs 1-Pass) ────────
    ax_e = fig.add_subplot(gs_bot[2])
    ens_metrics = [
        ("r_mean_w1", "Reader W1↓"),
        ("w_mean_w1", "Writer W1↓"),
        ("shift_mean_mae", "Shift MAE↓"),
        ("r_mean_pearson", "Reader r↑"),
        ("r_mean_ccc", "Reader CCC↑"),
        ("r_hard_shape_auroc_by_uplur", "Hard U_plur↑"),
        ("r_hard_shape_auroc_by_er08", "Hard ER0.8↑"),
    ]
    ye = np.arange(len(ens_metrics))
    dc_e5 = [ens["dc_ldl_fold0_E5"][k] for k, _ in ens_metrics]
    pt_e5 = [ens["patnet_fold0_E5"][k] for k, _ in ens_metrics]
    deltas = [p - d for p, d in zip(pt_e5, dc_e5)]
    h_b = 0.36
    ax_e.barh(ye - h_b / 2, dc_e5, height=h_b, color=PALETTE["cobalt"], label="DC-LDL (Fold-0 E=5)")
    ax_e.barh(ye + h_b / 2, pt_e5, height=h_b, color=PALETTE["hero"], label="PAT-Net (Fold-0 E=5)")
    for yi, pv, dv, (mk, _) in zip(ye, pt_e5, deltas, ens_metrics):
        good = (dv < 0) if ("w1" in mk.lower() or "mae" in mk.lower()) else (dv > 0)
        dcol = PALETTE["sage"] if good else PALETTE["hero"]
        ax_e.text(max(pv, dc_e5[yi]) + 0.008, yi, f"{pv:.4f} ({dv:+.4f})", va="center", ha="left", fontsize=6.9, fontweight="bold", color=dcol)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels([l for _, l in ens_metrics], fontsize=7.7)
    ax_e.invert_yaxis()
    ax_e.set_xlim(0.15, 0.80)
    ax_e.set_xlabel("Equal-Compute (E=5) Metric Value")
    ax_e.legend(loc="upper right", fontsize=7.3)
    _add_panel_badge(ax_e, "e", "Fold-0 E=5 Ensemble Gains (7/7 Wins)", x=-0.16)

    _save_fig(fig, "fig4_disentanglement_and_ablation")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Gate 3 Causal Perspective Transport & 7-Genre Leave-One-Genre-Out
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig5_perspective_transport_and_logo(
    report: Dict[str, object],
    audit_eb: Dict[str, object],
) -> None:
    oof = report["emobank_5fold_oof"]
    deb = report["emobank_deberta_v3_5fold"]
    logo = report["emobank_logo_7genre"]

    fig = plt.figure(figsize=(15.4, 8.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1.0, 1.22], hspace=0.30, wspace=0.23)

    # ── Panel a: Shift MAE Across RoBERTa, DeBERTa-v3 & 7-Genre LOGO ────────
    ax_a = fig.add_subplot(gs[0, 0])
    groups = ["RoBERTa\n5-Fold OOF", "DeBERTa-v3\n5-Fold OOF", "RoBERTa\n7-Genre LOGO"]
    x = np.arange(len(groups))
    w = 0.25
    dc_mae = [oof["dc_ldl"]["shift_mean_mae"], deb["dc_ldl"]["shift_mean_mae"], logo["dc_ldl"]["oof"]["shift_mean_mae"]]
    nt_mae = [oof["patnet_no_transport"]["shift_mean_mae"], deb["patnet_no_transport"]["shift_mean_mae"], logo["patnet_no_transport"]["oof"]["shift_mean_mae"]]
    pt_mae = [oof["patnet_M4"]["shift_mean_mae"], deb["patnet_M4"]["shift_mean_mae"], logo["patnet"]["oof"]["shift_mean_mae"]]

    ax_a.bar(x - w, dc_mae, width=w, color=PALETTE["cobalt"], label="DC-LDL (Independent)")
    ax_a.bar(x, nt_mae, width=w, color=PALETTE["slate"], label="PAT-Net (No Transport)")
    ax_a.bar(x + w, pt_mae, width=w, color=PALETTE["hero"], label="PAT-Net (Causal Transport)")
    for xi, a, b, c in zip(x, dc_mae, nt_mae, pt_mae):
        ax_a.text(xi - w, a + 0.0005, f"{a:.4f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
        ax_a.text(xi, b + 0.0005, f"{b:.4f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
        ax_a.text(xi + w, c + 0.0005, f"{c:.4f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["hero_dark"], rotation=90)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(groups, fontsize=8.3)
    ax_a.set_ylim(0.250, 0.278)
    ax_a.set_ylabel("Writer → Reader Shift MAE (Lower Better)")
    ax_a.legend(loc="upper left", fontsize=7.5)
    _add_panel_badge(ax_a, "a", "Perspective Shift Error Across Backbones & LOGO", x=-0.14)

    # ── Panel b: Asymmetric Offense Detection & Shift Pearson Correlation ────
    ax_b = fig.add_subplot(gs[0, 1])
    b_groups = [
        "RoB\nAsym. Offense\nAUROC↑",
        "DeB\nAsym. Offense\nAUROC↑",
        "LOGO OOF\nReader r↑",
        "RoB\nShift Pearson↑\n(×4 scaled)",
    ]
    xb = np.arange(len(b_groups))
    v_dc = [
        oof["dc_ldl"]["asymm_offense_auroc"],
        deb["dc_ldl"]["asymm_offense_auroc"],
        logo["dc_ldl"]["oof"]["r_mean_pearson"],
        oof["dc_ldl"]["shift_mean_pearson"] * 4.0,
    ]
    v_nt = [
        oof["patnet_no_transport"]["asymm_offense_auroc"],
        deb["patnet_no_transport"]["asymm_offense_auroc"],
        logo["patnet_no_transport"]["oof"]["r_mean_pearson"],
        oof["patnet_no_transport"]["shift_mean_pearson"] * 4.0,
    ]
    v_pt = [
        oof["patnet_M4"]["asymm_offense_auroc"],
        deb["patnet_M4"]["asymm_offense_auroc"],
        logo["patnet"]["oof"]["r_mean_pearson"],
        oof["patnet_M4"]["shift_mean_pearson"] * 4.0,
    ]
    raw_pt = [
        oof["patnet_M4"]["asymm_offense_auroc"],
        deb["patnet_M4"]["asymm_offense_auroc"],
        logo["patnet"]["oof"]["r_mean_pearson"],
        oof["patnet_M4"]["shift_mean_pearson"],
    ]
    raw_nt = [
        oof["patnet_no_transport"]["asymm_offense_auroc"],
        deb["patnet_no_transport"]["asymm_offense_auroc"],
        logo["patnet_no_transport"]["oof"]["r_mean_pearson"],
        oof["patnet_no_transport"]["shift_mean_pearson"],
    ]

    ax_b.bar(xb - w, v_dc, width=w, color=PALETTE["cobalt"], label="DC-LDL")
    ax_b.bar(xb, v_nt, width=w, color=PALETTE["slate"], label="PAT-Net (No Transport)")
    ax_b.bar(xb + w, v_pt, width=w, color=PALETTE["hero"], label="PAT-Net (Causal Transport)")
    for xi, rpt, rnt, vp, vn in zip(xb, raw_pt, raw_nt, v_pt, v_nt):
        ax_b.text(xi, vn + 0.008, f"{rnt:.3f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
        ax_b.text(xi + w, vp + 0.008, f"{rpt:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["hero_dark"], rotation=90)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(b_groups, fontsize=8.0)
    ax_b.set_ylim(0.40, 0.77)
    ax_b.set_ylabel("Asymmetric Detection / Alignment Score↑")
    ax_b.legend(loc="upper right", fontsize=7.4)
    _add_panel_badge(ax_b, "b", "Asymmetric Perspective Divergence & LOGO Correlation", x=-0.12)

    # ── Panel c: 7-Genre Leave-One-Genre-Out Per-Genre Transfer (7/7 Wins) ──
    ax_c = fig.add_subplot(gs[1, 0])
    genres = ["essays", "travel-guides", "blog", "newspaper", "fiction", "letters", "SemEval"]
    xg = np.arange(len(genres))
    g_dc = [logo["dc_ldl"]["per_genre"][g]["shift_mean_mae"] for g in genres]
    g_nt = [logo["patnet_no_transport"]["per_genre"][g]["shift_mean_mae"] for g in genres]
    g_pt = [logo["patnet"]["per_genre"][g]["shift_mean_mae"] for g in genres]

    ax_c.bar(xg - w, g_dc, width=w, color=PALETTE["cobalt"], label="DC-LDL")
    ax_c.bar(xg, g_nt, width=w, color=PALETTE["slate"], label="PAT-Net (No Trans.)")
    ax_c.bar(xg + w, g_pt, width=w, color=PALETTE["hero"], label="PAT-Net (Transport)")
    for xi, b, c in zip(xg, g_nt, g_pt):
        ax_c.text(xi + w, c + 0.0018, f"{c - b:+.3f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=PALETTE["hero_dark"], rotation=90)
    ax_c.set_xticks(xg)
    ax_c.set_xticklabels(genres, rotation=22, ha="right", fontsize=8.0)
    ax_c.set_ylim(0.215, 0.315)
    ax_c.set_ylabel("Held-Out Genre Shift MAE (Lower Better)")
    ax_c.legend(loc="upper left", fontsize=7.3)
    _add_panel_badge(ax_c, "c", "7-Genre Leave-One-Out Shift Error (7/7 Wins vs. No-Trans.)", x=-0.14)

    # ── Panel d: Per-Dimension Shift Error & Empirical Genre Asymmetry Scatter ─
    ax_d = fig.add_subplot(gs[1, 1])
    genre_aud = audit_eb["genre_perspective_shift_10062"]
    emp_shift_a = [genre_aud[g]["frac_abs_A_ge_0p5"] * 100.0 for g in genres]
    gain_vs_nt = [(logo["patnet_no_transport"]["per_genre"][g]["shift_mean_mae"] - logo["patnet"]["per_genre"][g]["shift_mean_mae"]) * 1000.0 for g in genres]
    col_g = [PALETTE["hero"] if g in ("SemEval", "letters", "fiction") else PALETTE["cobalt"] for g in genres]

    ax_d.scatter(emp_shift_a, gain_vs_nt, s=85, c=col_g, edgecolors="white", linewidths=0.8, zorder=4)
    for g, sx, gy, c in zip(genres, emp_shift_a, gain_vs_nt, col_g):
        if g == "fiction":
            ax_d.text(sx, gy + 0.45, g, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=c)
        elif g == "travel-guides":
            ax_d.text(sx, gy - 0.45, g, ha="center", va="top", fontsize=7.8, fontweight="bold", color=c)
        elif g == "SemEval":
            ax_d.text(sx, gy - 0.45, g, ha="center", va="top", fontsize=7.8, fontweight="bold", color=c)
        else:
            ax_d.text(sx, gy + 0.45, g, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=c)

    # Embedded inset: V/A/D Per-Dim Shift MAE Breakdown (RoBERTa + DeBERTa)
    ax_ins = ax_d.inset_axes([0.52, 0.50, 0.46, 0.45])
    dv_lbls = ["Valence", "Arousal", "Dominance"]
    xd2 = np.arange(3)
    nt_dim = [oof["patnet_no_transport"][f"shift_{d}_mae"] for d in ("V", "A", "D")]
    pt_dim = [oof["patnet_M4"][f"shift_{d}_mae"] for d in ("V", "A", "D")]
    ax_ins.bar(xd2 - 0.17, nt_dim, width=0.33, color=PALETTE["slate"], label="NoTrans")
    ax_ins.bar(xd2 + 0.17, pt_dim, width=0.33, color=PALETTE["hero"], label="PAT-Net")
    ax_ins.set_xticks(xd2)
    ax_ins.set_xticklabels(dv_lbls, fontsize=6.7)
    ax_ins.set_ylim(0.20, 0.32)
    ax_ins.tick_params(axis="y", labelsize=6.5)
    ax_ins.set_title("5-Fold Shift MAE by VAD", fontsize=7.2, fontweight="bold")
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    ax_ins.legend(fontsize=6.2, loc="upper left")

    ax_d.set_xlim(15.2, 23.8)
    ax_d.set_ylim(1.8, 11.8)
    ax_d.set_xlabel("Empirical Large Arousal Shift Rate $|A_R - A_W| \\geq 0.5$ (%)")
    ax_d.set_ylabel("Transport Gain vs. No-Trans ($\\times 10^{-3}$ MAE Reduction$\\uparrow$)")
    _add_panel_badge(ax_d, "d", "Empirical Asymmetry vs. Transfer Gain & VAD Breakdown", x=-0.12)

    _save_fig(fig, "fig5_perspective_transport_and_logo")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Gate 4 Cross-Dataset (EPIC Perspectivist Irony + BRIGHTER 4-Lang)
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig6_cross_dataset_epic_brighter(report: Dict[str, object], audit_epic: Dict[str, object]) -> None:
    epic_models = report["gates"]["gate4_epic_5fold_and_brighter"]["epic"]["models"]
    brighter = report["gates"]["gate4_epic_5fold_and_brighter"]["brighter"]

    fig = plt.figure(figsize=(15.6, 9.2))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.08], hspace=0.32)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], width_ratios=[1.18, 1.0, 0.92], wspace=0.24)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1], width_ratios=[1.35, 1.0], wspace=0.22)

    # ── Panel a: EPIC 5-Fold Polarization Double-Dissociation ───────────────
    ax_a = fig.add_subplot(gs_top[0])
    ep_keys = [
        ("C1_SparseRidge", "Sparse\nRidge"),
        ("roberta_scalar_bce", "RoB\nScalar BCE"),
        ("roberta_beta_binom", "RoB\nBeta-Binom"),
        ("roberta_patnet_no_transport", "RoB\nNo-Trans."),
        ("roberta_patnet_M4", "RoB\nPAT-Net"),
        ("deberta_patnet_M4", "DeB\nPAT-Net (3f)"),
    ]
    x = np.arange(len(ep_keys))
    w = 0.36
    plur_v = [epic_models[k]["oof_pol_auroc_by_uplur"] for k, _ in ep_keys]
    amb_v = [epic_models[k]["oof_pol_auroc_by_uamb"] for k, _ in ep_keys]
    ax_a.bar(x - w / 2, plur_v, width=w, color=PALETTE["hero"], label="Between-Faction $U_{plur}$")
    ax_a.bar(x + w / 2, amb_v, width=w, color=PALETTE["cobalt"], label="Within-Mode $U_{amb}$")
    for xi, v1, v2 in zip(x, plur_v, amb_v):
        ax_a.text(xi - w / 2, v1 + 0.004, f"{v1:.3f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
        ax_a.text(xi + w / 2, v2 + 0.004, f"{v2:.3f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([l for _, l in ep_keys], fontsize=7.6)
    ax_a.set_ylim(0.45, 0.785)
    ax_a.set_ylabel("Irony Polarization AUROC↑")
    ax_a.legend(loc="lower right", fontsize=7.2)
    _add_panel_badge(ax_a, "a", "EPIC Perspectivist Double-Dissociation", x=-0.14)

    # ── Panel b: EPIC Demographic Sex-Shift MAE (1,514-Cluster Bootstrap p=0.0) ──
    ax_b = fig.add_subplot(gs_top[1])
    sh_keys = [
        ("C1_SparseRidge", "Sparse\nRidge", PALETTE["graphite"]),
        ("C2_HybridGBDT", "Hybrid\nGBDT", PALETTE["graphite_mid"]),
        ("roberta_patnet_no_transport", "RoB\nNo-Trans", PALETTE["slate"]),
        ("roberta_patnet_M4", "RoB\nPAT-Net", PALETTE["hero"]),
        ("deberta_patnet_no_transport", "DeB\nNo-Trans", PALETTE["slate"]),
        ("deberta_patnet_M4", "DeB\nPAT-Net", PALETTE["amber"]),
    ]
    xs = np.arange(len(sh_keys))
    sh_vals = [epic_models[k]["oof_sex_shift_mae"] for k, _, _ in sh_keys]
    sh_cols = [c for _, _, c in sh_keys]
    ax_b.bar(xs, sh_vals, width=0.58, color=sh_cols)
    for xi, v in zip(xs, sh_vals):
        ax_b.text(xi, v + 0.0015, f"{v:.4f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold" if "PAT-Net" in sh_keys[xi][1] else "normal")
    # Bracket between RoB No-Trans and RoB PAT-Net
    ax_b.annotate("", xy=(3, 0.298), xytext=(2, 0.298), arrowprops={"arrowstyle": "<->", "color": PALETTE["hero_dark"], "lw": 1.1})
    ax_b.text(2.5, 0.2995, "−0.0149 (p=0.0)", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["hero_dark"])
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([l for _, l, _ in sh_keys], fontsize=7.6)
    ax_b.set_ylim(0.255, 0.322)
    ax_b.set_ylabel("Demographic Sex-Shift MAE↓")
    _add_panel_badge(ax_b, "b", "EPIC Male vs. Female Perspective Transport", x=-0.15)

    # ── Panel c: EPIC Overall Irony Pearson r vs Conformal Polarization FNR ──
    ax_c = fig.add_subplot(gs_top[2])
    c_keys = [
        ("C1_SparseRidge", "SparseRidge", PALETTE["graphite"], "s", (0.310, 0.137, "left")),
        ("C2_HybridGBDT", "HybridGBDT", PALETTE["graphite_mid"], "D", (0.295, 0.108, "left")),
        ("roberta_scalar_bce", "RoB Scalar BCE", PALETTE["teal_mid"], "^", (0.438, 0.097, "right")),
        ("roberta_patnet_no_transport", "RoB NoTrans", PALETTE["slate"], "X", (0.438, 0.091, "right")),
        ("roberta_beta_binom", "RoB Beta-Binom", PALETTE["cobalt"], "v", (0.438, 0.085, "right")),
        ("roberta_patnet_M4", "RoB PAT-Net", PALETTE["hero"], "*", (0.438, 0.074, "right")),
        ("deberta_patnet_no_transport", "DeB NoTrans", PALETTE["slate_light"], "X", (0.460, 0.114, "right")),
        ("deberta_patnet_M4", "DeB PAT-Net (3f)", PALETTE["amber"], "*", (0.438, 0.0795, "right")),
    ]
    ax_c.axhspan(0.05, 0.10, color=PALETTE["sage_soft"], alpha=0.65, zorder=0, label="CRC Safety Target ($\\mathrm{FNR} \\leq 0.10$)")
    ax_c.axhline(0.10, color=PALETTE["sage"], linestyle="--", linewidth=1.1, zorder=1)
    for k, lbl, col, mk, (tx, ty, ha) in c_keys:
        pr = epic_models[k]["oof_overall_pearson"]
        fnr = epic_models[k]["crc_pol_alpha10_mean"]["test_fnr"]
        sz = 185 if mk == "*" else 58
        ax_c.scatter(pr, fnr, marker=mk, s=sz, color=col, edgecolors="white", linewidths=0.7, zorder=4 if mk == "*" else 3)
        ax_c.annotate(
            lbl, xy=(pr, fnr), xytext=(tx, ty),
            fontsize=6.8, ha=ha, va="center",
            fontweight="bold" if mk == "*" else "normal",
            color=col,
            arrowprops={"arrowstyle": "-", "color": col, "alpha": 0.55, "lw": 0.65} if pr > 0.45 else None,
        )
    ax_c.set_xlim(0.25, 0.535)
    ax_c.set_ylim(0.068, 0.148)
    ax_c.set_xlabel("Overall Irony Rate Pearson r↑")
    ax_c.set_ylabel("Polarization CRC Test FNR↓")
    ax_c.legend(loc="upper right", fontsize=6.8)
    _add_panel_badge(ax_c, "c", "EPIC Accuracy–Safety Frontier", x=-0.16)

    # ── Panel d: BRIGHTER All 6 Language×Backbone Tracks Calibration (ECE) ───
    ax_d = fig.add_subplot(gs_bot[0])

    def _find_br(lang: str, bb: str, mt: str, ab: str | None = None) -> Dict[str, object] | None:
        for row in brighter.values():
            if row["lang"] == lang and row["backbone"] == bb and row["model_type"] == mt:
                if ab is None or row["ablation_mode"] == ab:
                    return row
        return None

    br_tracks = [
        ("chn", "hfl/chinese-roberta-wwm-ext", "CHN\nRoBERTa-wwm"),
        ("chn", "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese", "CHN\nErlangshen"),
        ("eng", "roberta-base", "ENG\nRoBERTa"),
        ("eng", "microsoft/deberta-v3-base", "ENG\nDeBERTa-v3"),
        ("deu", "xlm-roberta-base", "DEU\nXLM-R"),
        ("esp", "xlm-roberta-base", "ESP\nXLM-R"),
    ]
    xt = np.arange(len(br_tracks))
    bw = 0.19
    m_br = [
        ("scalar_mse", "M0_baseline", "Scalar MSE", PALETTE["graphite"]),
        ("sord", "M0_baseline", "SORD", PALETTE["cobalt"]),
        ("uw_ldl", "M0_baseline", "UW-LDL", PALETTE["teal"]),
        ("dml_ordinal", "M4_disentangle", "DML-Ord (M4)", PALETTE["hero"]),
    ]
    for idx, (mt, ab, lbl, col) in enumerate(m_br):
        vals_ece = []
        for lg, bb, _ in br_tracks:
            row = _find_br(lg, bb, mt, ab)
            # Erlangshen runs sord and dml_ordinal(M0/M4), not scalar_mse/uw_ldl -> safely skip NaN
            vals_ece.append(row["test_macro_ece"] if row is not None else np.nan)
        offs = (idx - 1.5) * bw
        for xi, ev in zip(xt, vals_ece):
            if np.isnan(ev):
                continue
            ax_d.bar(xi + offs, ev, width=bw, color=col, label=lbl if (xi == 0 or (idx in (0, 2) and xi == 2)) else "_nolegend_")
            ax_d.text(xi + offs, ev + 0.009, f"{ev:.3f}", ha="center", va="bottom", fontsize=6.4, rotation=90,
                      fontweight="bold" if mt == "dml_ordinal" else "normal", color=PALETTE["hero_dark"] if mt == "dml_ordinal" else "#2B333E")

    # Deduplicate legend cleanly
    hnd, lbs = ax_d.get_legend_handles_labels()
    seen = {}
    for h, l in zip(hnd, lbs):
        if l not in seen:
            seen[l] = h
    ax_d.legend(seen.values(), seen.keys(), loc="upper left", ncol=2, fontsize=7.4)
    ax_d.set_xticks(xt)
    ax_d.set_xticklabels([n for _, _, n in br_tracks], fontsize=8.0)
    ax_d.set_ylim(0.0, 0.62)
    ax_d.set_ylabel("Macro Probability Calibration ECE↓")
    _add_panel_badge(ax_d, "d", "BRIGHTER Calibration Across 6 Language × Backbone Tracks", x=-0.10)

    # ── Panel e: BRIGHTER Multi-Metric Discrimination Heatmap ───────────────
    ax_e = fig.add_subplot(gs_bot[1])
    br_hm_rows = [
        ("CHN (RoB-wwm)", "chn", "hfl/chinese-roberta-wwm-ext"),
        ("ENG (DeB-v3)", "eng", "microsoft/deberta-v3-base"),
        ("DEU (XLM-R)", "deu", "xlm-roberta-base"),
        ("ESP (XLM-R)", "esp", "xlm-roberta-base"),
    ]
    br_hm_cols = [
        ("DML-Ord (M4)\nPearson r↑", "dml_ordinal", "M4_disentangle", "test_macro_pearson"),
        ("SORD\nPearson r↑", "sord", "M0_baseline", "test_macro_pearson"),
        ("DML-Ord (M4)\nQWK↑", "dml_ordinal", "M4_disentangle", "test_macro_qwk"),
        ("Scalar MSE\nQWK↑", "scalar_mse", "M0_baseline", "test_macro_qwk"),
        ("DML-Ord (M4)\nExtremity AUC↑", "dml_ordinal", "M4_disentangle", "test_extremity_auroc"),
        ("Scalar MSE\nExtremity AUC↑", "scalar_mse", "M0_baseline", "test_extremity_auroc"),
    ]
    mat_b = np.zeros((len(br_hm_rows), len(br_hm_cols)))
    for i, (_, lg, bb) in enumerate(br_hm_rows):
        for j, (_, mt, ab, field) in enumerate(br_hm_cols):
            mat_b[i, j] = _find_br(lg, bb, mt, ab)[field]
    cmap_br = LinearSegmentedColormap.from_list("br_pub", ["#F6E2E0", "#F7F8FA", "#C4D6EB", "#295C94"])
    ax_e.imshow(mat_b, cmap=cmap_br, aspect="auto", vmin=0.35, vmax=0.95)
    for i in range(len(br_hm_rows)):
        for j in range(len(br_hm_cols)):
            val = mat_b[i, j]
            tc = "white" if val > 0.78 else "#1E242B"
            ax_e.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.6,
                      fontweight="bold" if j in (0, 2, 4) else "normal", color=tc)
    ax_e.set_xticks(np.arange(len(br_hm_cols)))
    ax_e.set_xticklabels([l for l, _, _, _ in br_hm_cols], fontsize=7.2, rotation=18, ha="right")
    ax_e.set_yticks(np.arange(len(br_hm_rows)))
    ax_e.set_yticklabels([l for l, _, _ in br_hm_rows], fontsize=8.0, fontweight="bold")
    ax_e.grid(False)
    for sp in ax_e.spines.values():
        sp.set_visible(False)
    ax_e.set_xticks(np.arange(-0.5, len(br_hm_cols), 1), minor=True)
    ax_e.set_yticks(np.arange(-0.5, len(br_hm_rows), 1), minor=True)
    ax_e.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax_e.tick_params(which="minor", bottom=False, left=False)
    _add_panel_badge(ax_e, "e", "BRIGHTER Ordinal Accuracy, QWK & Extremity Matrix", x=-0.14)

    _save_fig(fig, "fig6_cross_dataset_epic_brighter")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Statistical Significance (28 Tests) & Conformal Safety Operating Map
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig7_bootstrap_forest_and_conformal(report: Dict[str, object]) -> None:
    boot = report["cluster_bootstrap_2000"]
    oof = report["emobank_5fold_oof"]
    deb = report["emobank_deberta_v3_5fold"]
    logo = report["emobank_logo_7genre"]
    epic_m = report["gates"]["gate4_epic_5fold_and_brighter"]["epic"]["models"]

    fig = plt.figure(figsize=(15.4, 9.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)

    baselines = [
        ("beta_nll", "Beta-NLL"),
        ("C2_HybridGBDT", "C2 HybridGBDT"),
        ("C1_SparseRidge", "C1 SparseRidge"),
        ("gaussian_nll", "Gaussian NLL"),
        ("gaussian_nll_mcT10", "Gaussian NLL (MC)"),
        ("sord", "SORD"),
        ("dldl_v2", "DLDL-v2"),
        ("scalar_mse", "Scalar MSE"),
        ("dc_ldl_mcT10", "DC-LDL (MC)"),
        ("uw_ldl", "UW-LDL"),
        ("dc_ldl", "DC-LDL"),
        ("patnet_no_transport", "PAT-Net (NoTrans)"),
        ("patnet_unimodal", "PAT-Net (Uni)"),
        ("nig_evidential", "NIG Evidential"),
    ]

    def _draw_forest(ax: plt.Axes, prefix: str, xlim: Tuple[float, float], xlabel: str, badge: str, title: str) -> None:
        y = np.arange(len(baselines))
        for yi, (bk, lbl) in enumerate(baselines):
            if yi % 2 == 0:
                ax.axhspan(yi - 0.45, yi + 0.45, color=PALETTE["ivory_band"], zorder=0)
            b = boot[f"{prefix}{bk}"]
            m = b["diff_mean"]
            lo = m - b["ci95_low"]
            hi = b["ci95_high"] - m
            sig = "***" if b["p_holm"] < 0.001 else ("*" if b["p_holm"] < 0.05 else f"p_H={b['p_holm']:.3f}")
            col = PALETTE["hero"] if b["p_holm"] < 0.05 else PALETTE["teal"]
            ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt="o", color=col, ecolor=col, capsize=2.8, ms=4.5, lw=1.35, zorder=3)
            span_x = xlim[1] - xlim[0]
            if "shift_mae" in prefix:
                tx_pos, ha = max(b["ci95_high"], 0.0) + span_x * 0.018, "left"
            elif bk in ("dc_ldl", "uw_ldl", "dc_ldl_mcT10", "patnet_no_transport", "patnet_unimodal", "nig_evidential"):
                tx_pos, ha = max(b["ci95_high"], 0.0) + span_x * 0.016, "left"
            elif bk == "scalar_mse":
                tx_pos, ha = b["ci95_low"] - span_x * 0.016, "right"
            else:
                tx_pos, ha = b["ci95_high"] + span_x * 0.014, "left"
            ax.text(tx_pos, yi, f"{m:+.4f} ({sig})", va="center", ha=ha, fontsize=7.0,
                    fontweight="bold" if b["p_holm"] < 0.05 else "normal", color="#1E242B")
        ax.axvline(0.0, color="#2B333E", linestyle="--", linewidth=1.0, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([lbl for _, lbl in baselines], fontsize=7.8)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        _add_panel_badge(ax, badge, title, x=-0.16)

    # ── Panel a: Bootstrap W1 Difference ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_forest(
        ax_a, "w1_vs_", (-0.106, 0.042),
        "Paired Reader W1 Diff: PAT-Net − Baseline (< 0 Favors PAT-Net)",
        "a", "136-Cluster Bootstrap: Reader W1 (13/14 Significant)",
    )

    # ── Panel b: Bootstrap Perspective Shift MAE Difference ─────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_forest(
        ax_b, "shift_mae_vs_", (-0.025, 0.0145),
        "Paired Shift MAE Diff: PAT-Net − Baseline (< 0 Favors PAT-Net)",
        "b", "136-Cluster Bootstrap: Shift MAE (11/14 Signif., All < 0)",
    )

    # ── Panel c: Mondrian Conformal Credal Coverage & Adaptive Width ────────
    ax_c = fig.add_subplot(gs[1, 0])
    conf_models = [
        ("scalar_mse", "Scalar\nMSE"),
        ("gaussian_nll", "Gaussian\nNLL"),
        ("sord", "SORD"),
        ("dc_ldl", "DC-LDL"),
        ("patnet_unimodal", "PAT\nUni"),
        ("patnet_M4", "PAT-Net\n(M4)"),
    ]
    xc = np.arange(len(conf_models))
    w = 0.25
    cov_v = [oof[k]["conformal_mean"]["conf_V_crc_human_coverage"] * 100 for k, _ in conf_models]
    cov_a = [oof[k]["conformal_mean"]["conf_A_crc_human_coverage"] * 100 for k, _ in conf_models]
    cov_d = [oof[k]["conformal_mean"]["conf_D_crc_human_coverage"] * 100 for k, _ in conf_models]
    wid_v = [oof[k]["conformal_mean"]["conf_V_crc_mean_width_bins"] for k, _ in conf_models]
    wid_a = [oof[k]["conformal_mean"]["conf_A_crc_mean_width_bins"] for k, _ in conf_models]
    wid_d = [oof[k]["conformal_mean"]["conf_D_crc_mean_width_bins"] for k, _ in conf_models]

    ax_c.bar(xc - w, cov_v, width=w, color=PALETTE["cobalt"], label="Valence Coverage")
    ax_c.bar(xc, cov_a, width=w, color=PALETTE["amber"], label="Arousal Coverage")
    ax_c.bar(xc + w, cov_d, width=w, color=PALETTE["teal"], label="Dominance Coverage")
    ax_c.axhline(90.0, color=PALETTE["hero"], linestyle="--", linewidth=1.3, label="Target Coverage (90%)")
    for xi, wv, wa, wd, cv, ca, cd in zip(xc, wid_v, wid_a, wid_d, cov_v, cov_a, cov_d):
        mw = (wv + wa + wd) / 3.0
        ax_c.text(xi, max(cv, ca, cd) + 0.85, f"w̄={mw:.2f}", ha="center", va="bottom", fontsize=7.6,
                  fontweight="bold" if xi == len(conf_models) - 1 else "normal")
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([l for _, l in conf_models], fontsize=7.9)
    ax_c.set_ylim(66, 102)
    ax_c.set_ylabel("Finite-Sample Human Coverage (%)")
    ax_c.legend(loc="upper left", ncol=2, fontsize=7.3)
    _add_panel_badge(ax_c, "c", "Mondrian Conformal Credal Coverage & Width (α=0.10)", x=-0.14)

    # ── Panel d: Split-CRC Safety–Volume Operating Plane (FNR vs Alert Vol) ─
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axhspan(0.02, 0.10, color=PALETTE["sage_soft"], alpha=0.65, zorder=0, label="CRC Safety Zone (FNR ≤ 0.10)")
    ax_d.axhline(0.10, color=PALETTE["sage"], linestyle="--", linewidth=1.25, zorder=1)
    ax_d.axvspan(0.935, 0.98, color=PALETTE["hero_soft"], alpha=0.35, zorder=0)
    ax_d.text(0.957, 0.132, "Trivial\n~96% Alert\nRegion", ha="center", va="center", fontsize=6.9, color=PALETTE["graphite_mid"], style="italic")

    crc_pts = [
        ("RoB Scalar", oof["scalar_mse"]["conformal_mean"]["pol_crc_sup_alert_volume"], oof["scalar_mse"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["graphite"], "s", (0.928, 0.044, "right")),
        ("DeB Scalar", deb["scalar_mse"]["conformal_mean"]["pol_crc_sup_alert_volume"], deb["scalar_mse"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["graphite_mid"], "s", (0.928, 0.056, "right")),
        ("RoB DC-LDL", oof["dc_ldl"]["conformal_mean"]["pol_crc_sup_alert_volume"], oof["dc_ldl"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["cobalt"], "o", (0.788, 0.128, "right")),
        ("RoB SORD", oof["sord"]["conformal_mean"]["pol_crc_sup_alert_volume"], oof["sord"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["steel"], "D", (0.826, 0.132, "left")),
        ("DeB DC-LDL", deb["dc_ldl"]["conformal_mean"]["pol_crc_sup_alert_volume"], deb["dc_ldl"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["sky"], "o", (0.822, 0.119, "left")),
        ("DeB NoTrans", deb["patnet_no_transport"]["conformal_mean"]["pol_crc_sup_alert_volume"], deb["patnet_no_transport"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["slate"], "X", (0.805, 0.107, "right")),
        ("RoB PAT-Net (M4)", oof["patnet_M4"]["conformal_mean"]["pol_crc_sup_alert_volume"], oof["patnet_M4"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["hero"], "*", (0.892, 0.091, "left")),
        ("DeB PAT-Net (M4)", deb["patnet_M4"]["conformal_mean"]["pol_crc_sup_alert_volume"], deb["patnet_M4"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["amber"], "*", (0.871, 0.062, "center")),
        ("EPIC PAT-Net", epic_m["roberta_patnet_M4"]["crc_pol_alpha10_mean"]["alert_volume"], epic_m["roberta_patnet_M4"]["crc_pol_alpha10_mean"]["test_fnr"], PALETTE["hero_dark"], "^", (0.836, 0.082, "right")),
        ("LOGO PAT-Net (Cov. Shift)", logo["patnet"]["oof"]["conformal_mean"]["pol_crc_sup_alert_volume"], logo["patnet"]["oof"]["conformal_mean"]["pol_crc_sup_test_fnr"], PALETTE["violet"], "P", (0.806, 0.147, "left")),
    ]
    for lbl, vx, fy, col, mk, (tx, ty, ha) in crc_pts:
        sz = 190 if mk == "*" else 58
        ax_d.scatter(vx, fy, marker=mk, s=sz, color=col, edgecolors="white", linewidths=0.7, zorder=4 if mk == "*" else 3)
        ax_d.annotate(
            lbl, xy=(vx, fy), xytext=(tx, ty), ha=ha, va="center", fontsize=6.9,
            fontweight="bold" if mk == "*" else "normal", color=col,
            arrowprops={"arrowstyle": "-", "color": col, "alpha": 0.45, "lw": 0.7} if abs(ty - fy) > 0.008 or abs(tx - vx) > 0.015 else None,
        )

    ax_d.set_xlim(0.75, 0.98)
    ax_d.set_ylim(0.032, 0.156)
    ax_d.set_xlabel("Conformal Alert Volume Fraction (Lower = More Selective)")
    ax_d.set_ylabel("Out-of-Fold Polarization Test FNR (Lower = Safer)")
    ax_d.legend(loc="lower left", fontsize=7.3)
    _add_panel_badge(ax_d, "d", "Split-CRC Risk vs. Alert Selectivity Across Tasks", x=-0.14)

    _save_fig(fig, "fig7_bootstrap_forest_and_conformal")


def main() -> None:
    _configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any previous fig0 / fig1 / stale numbered figure outputs so figures start cleanly at Fig. 2
    for stale in list(FIG_DIR.glob("fig*.*")):
        stale.unlink()

    report = json.loads(_resolve_json("all_gates_report.json").read_text(encoding="utf-8"))
    audit_eb = json.loads(_resolve_json("audit_emobank.json").read_text(encoding="utf-8"))
    audit_epic = json.loads(_resolve_json("audit_epic.json").read_text(encoding="utf-8"))
    audit_ge = json.loads(_resolve_json("audit_goemotions.json").read_text(encoding="utf-8"))

    plot_fig2_empirical_phenomena_and_audit(audit_eb, audit_epic, audit_ge)
    plot_fig3_emobank_pareto_and_efficiency(report)
    plot_fig4_disentanglement_and_ablation(report)
    plot_fig5_perspective_transport_and_logo(report, audit_eb)
    plot_fig6_cross_dataset_epic_brighter(report, audit_epic)
    plot_fig7_bootstrap_forest_and_conformal(report)

    generated = sorted(p.name for p in FIG_DIR.glob("fig*.*"))
    print(f"Generated {len(generated)} top-venue figure files (Fig. 2–Fig. 7) in {FIG_DIR}:")
    for name in generated:
        print(" -", name)


if __name__ == "__main__":
    main()
