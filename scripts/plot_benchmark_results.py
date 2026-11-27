"""Plot model benchmark results across EmoBank, GoEmotions Raw, EPIC, and BRIGHTER."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
FIG_DIR = OUT_ROOT / "figures"


def plot_emobank_comparison() -> Path | None:
    eb_dir = OUT_ROOT / "emobank"
    rows = []
    for sp in sorted(eb_dir.glob("official_roberta-base_*_seed*/summary.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        rows.append({"model_type": d["model_type"], "seed": d["seed"], **d["test_metrics"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    order = [
        "scalar_mse",
        "gaussian_nll",
        "nig_evidential",
        "softmax_ldl",
        "patnet_unimodal",
        "patnet_no_transport",
        "patnet",
    ]
    labels_map = {
        "scalar_mse": "Scalar MSE",
        "gaussian_nll": "Gaussian NLL",
        "nig_evidential": "Evidential NIG",
        "softmax_ldl": "Softmax LDL",
        "patnet_unimodal": "PAT-Net (Unimodal)",
        "patnet_no_transport": "PAT-Net (No Trans.)",
        "patnet": "PAT-Net (Full)",
    }
    df = df[df["model_type"].isin(order)]
    agg = df.groupby("model_type").agg(["mean", "std"]).reindex([m for m in order if m in set(df["model_type"])])

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.4), dpi=200)
    names = [labels_map[m] for m in agg.index]
    x = np.arange(len(names))
    colors = ["#7F7F7F" if m != "patnet" else "#D62728" for m in agg.index]

    pol_col = "r_mean_pol_auroc_by_std" if ("r_mean_pol_auroc_by_std", "mean") in agg.columns else "r_mean_pol_auroc"
    specs = [
        ("r_mean_w1", "A. Reader 1D Wasserstein-1 (Lower Better)", "Wasserstein-1 (W1)", True),
        (pol_col, "B. Polarized vs Neutral AUROC (Higher Better)", "AUROC", False),
        ("r_mean_pearson", "C. Scalar Mean Pearson r (Higher Better)", "Pearson r", False),
        ("shift_mean_mae", "D. Writer->Reader Shift MAE (Lower Better)", "Shift MAE", True),
    ]
    for ax, (col, title, ylab, lower_best) in zip(axes, specs):
        means = agg[(col, "mean")].to_numpy(dtype=float)
        stds = np.nan_to_num(agg[(col, "std")].to_numpy(dtype=float), nan=0.0)
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.62)
        for xi, v in enumerate(means):
            if np.isfinite(v):
                ax.text(xi, v + max(stds[xi], 0.004) + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8.3)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.0, fontweight="bold")
        valid = means[np.isfinite(means)]
        if len(valid):
            lo = max(0.0, valid.min() - 0.08)
            hi = valid.max() + 0.08
            ax.set_ylim(lo, hi)

    plt.tight_layout()
    out_p = FIG_DIR / "fig3_emobank_results.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def plot_goemotions_dissociation() -> Path | None:
    ge_dir = OUT_ROOT / "goemotions"
    rows = []
    for sp in sorted(ge_dir.glob("*/summary.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        rows.append({"target_mode": d["target_mode"], "model_type": d["model_type"], "seed": d["seed"], **d["test_metrics"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    sub = df[df["target_mode"] == "total"].groupby("model_type").mean(numeric_only=True)
    if len(sub) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=200)
    models = [m for m in ["flat_bce", "mc_dropout_bce", "apm_no_cooccur", "apm_net"] if m in sub.index]
    x = np.arange(len(models))
    w = 0.34

    # Panel A: Unclear AUROC by U_amb vs U_plur vs Entropy
    ax = axes[0]
    u_amb_auc = [sub.loc[m, "unclear_auroc_by_uamb"] for m in models]
    u_plur_auc = [sub.loc[m, "unclear_auroc_by_uplur"] for m in models]
    ax.bar(x - w / 2, u_amb_auc, width=w, color="#4C78A8", label="By U_amb (Ambiguity)")
    ax.bar(x + w / 2, u_plur_auc, width=w, color="#E45756", label="By U_plur (Pluralism)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, fontsize=8.5)
    ax.set_ylabel("AUROC Predicting Ground-Truth Unclear")
    ax.set_title("A. Detecting Linguistic Unclarity (No Unclear Labels)", fontsize=10.0, fontweight="bold")
    ax.set_ylim(0.45, 0.92)
    ax.legend(fontsize=8.5)

    # Panel B: Clear Pos-Neg Conflict AUROC by U_plur vs U_amb
    ax = axes[1]
    pn_plur = [sub.loc[m, "clear_posneg_auroc_by_uplur"] for m in models]
    pn_amb = [sub.loc[m, "clear_posneg_auroc_by_uamb"] for m in models]
    ax.bar(x - w / 2, pn_amb, width=w, color="#4C78A8", label="By U_amb (Ambiguity)")
    ax.bar(x + w / 2, pn_plur, width=w, color="#E45756", label="By U_plur (Pluralism)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, fontsize=8.5)
    ax.set_ylabel("AUROC Predicting Clear Pos-Neg Polarization")
    ax.set_title("B. Detecting Clear Pluralistic Polarization (Double Dissociation)", fontsize=10.0, fontweight="bold")
    ax.set_ylim(0.45, 0.92)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    out_p = FIG_DIR / "fig4_goemotions_dissociation.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p3 = plot_emobank_comparison()
    p4 = plot_goemotions_dissociation()
    print("Generated:", p3, p4)


if __name__ == "__main__":
    main()
