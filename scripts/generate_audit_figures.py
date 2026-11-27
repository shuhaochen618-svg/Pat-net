"""Generate publication-quality empirical audit figures for EmoBank and GoEmotions Raw gaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"


def make_emobank_gap_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=200)

    # Panel A: Inside neutral-mean band [2.5, 3.5], show actual std distribution per dimension
    ax = axes[0]
    dims = ["V", "A", "D"]
    cats = ["Unanimous Neutral\n(std < 0.05)", "Moderate Diffuse\n(0.05 <= std < 0.8)", "Polarized / Bimodal\n(std >= 0.8 or both tails)"]
    mat = np.zeros((3, 3))
    for i, d in enumerate(dims):
        m = df[f"r_{d}_mean"]
        s = df[f"r_{d}_std"]
        bp = df[f"r_{d}_bipole_min"]
        mid = (m >= 2.5) & (m <= 3.5)
        n_mid = float(mid.sum())
        c0 = (mid & (s < 0.05)).sum() / n_mid * 100.0
        c2 = (mid & ((s >= 0.8) | (bp > 0))).sum() / n_mid * 100.0
        c1 = 100.0 - c0 - c2
        mat[i] = [c0, c1, c2]

    x = np.arange(3)
    colors = ["#4C78A8", "#F58518", "#E45756"]
    bottom = np.zeros(3)
    for k in range(3):
        bars = ax.bar(x, mat[:, k], bottom=bottom, color=colors[k], width=0.58, label=cats[k])
        for xi, h, b in zip(x, mat[:, k], bottom):
            if h >= 6.0:
                ax.text(xi, b + h * 0.5, f"{h:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        bottom += mat[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels(["Valence (V)", "Arousal (A)", "Dominance (D)"])
    ax.set_ylabel("% of Mean-Neutral Items [2.5, 3.5]")
    ax.set_title("A. Hidden Polarization Inside 'Neutral' Means", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper center", ncol=1, fontsize=7.3, framealpha=0.95)
    ax.set_ylim(0, 145)

    # Panel B: Example histograms sharing identical scalar mean = 3.0 in Reader Arousal
    ax = axes[1]
    scores = np.array([1, 2, 3, 4, 5])
    cand_unanimous = df[(np.abs(df["r_A_mean"] - 3.0) < 0.01) & (df["r_A_std"] < 0.01)].iloc[0]
    cand_polarized = df[(np.abs(df["r_A_mean"] - 3.0) < 0.05) & (df["r_A_bipole_min"] >= 0.35)].sort_values("r_A_std", ascending=False).iloc[0]
    p_u = [cand_unanimous[f"r_A_p{k}"] for k in range(1, 6)]
    p_p = [cand_polarized[f"r_A_p{k}"] for k in range(1, 6)]
    w = 0.36
    ax.bar(scores - w / 2, p_u, width=w, color="#4C78A8", label=f"Unanimous (std={cand_unanimous['r_A_std']:.2f})")
    ax.bar(scores + w / 2, p_p, width=w, color="#E45756", label=f"Bimodal Conflict (std={cand_polarized['r_A_std']:.2f})")
    ax.set_xticks(scores)
    ax.set_xlabel("Discrete Rating Scale (1-5)")
    ax.set_ylabel("Human Rating Probability")
    ax.set_title("B. Identical Mean (3.0) Collides Opposite Realities", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.12)

    # Panel C: Genre-wise Writer -> Reader Divergence across all 7 genres
    ax = axes[2]
    genres = ["SemEval", "letters", "fiction", "newspaper", "essays", "blog", "travel-guides"]
    vals_v = [((df[df["category"] == g]["shift_abs_V"] >= 0.5).mean() * 100.0) for g in genres]
    vals_a = [((df[df["category"] == g]["shift_abs_A"] >= 0.5).mean() * 100.0) for g in genres]
    y = np.arange(len(genres))
    h = 0.36
    ax.barh(y - h / 2, vals_v, height=h, color="#54A24B", label="|V_W - V_R| >= 0.5")
    ax.barh(y + h / 2, vals_a, height=h, color="#B279A2", label="|A_W - A_R| >= 0.5")
    ax.set_yticks(y)
    ax.set_yticklabels(genres, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of Sentences with Large Perspective Shift")
    ax.set_title("C. Genre-Dependent Writer–Reader Asymmetry", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 29)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_emobank_gap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_goemotions_gap_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=200)

    # Panel A: Why total Shannon entropy fails — Clear Polarized > Unclear!
    ax = axes[0]
    clear_agree = df[(~df["has_any_unclear"]) & (df["clear_disagree_jaccard"] == 0)]["total_bernoulli_entropy"]
    unclear = df[df["has_any_unclear"]]["total_bernoulli_entropy"]
    clear_polar = df[(~df["has_any_unclear"]) & (df["pos_neg_conflict"] > 0)]["total_bernoulli_entropy"]

    means = [clear_agree.mean(), unclear.mean(), clear_polar.mean()]
    errs = [clear_agree.std() / np.sqrt(len(clear_agree)), unclear.std() / np.sqrt(len(unclear)), clear_polar.std() / np.sqrt(len(clear_polar))]
    labels = [
        f"Clear Consensus\n(n={len(clear_agree):,})",
        f"Linguistic Unclear\n(n={len(unclear):,})",
        f"Clear Pos-Neg Conflict\n(n={len(clear_polar):,})",
    ]
    bars = ax.bar(range(3), means, yerr=1.96 * np.array(errs), capsize=4, color=["#4C78A8", "#F58518", "#E45756"], width=0.55)
    for idx, val in enumerate(means):
        ax.text(idx, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Total Multi-Label Bernoulli Entropy (bits)")
    ax.set_title("A. Total Entropy Misranks Pluralism Above Unclarity", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.22)

    # Panel B: Vote Sum (Clarity Mass) vs Cross-Valence Conflict cleanly separate the regimes
    ax = axes[1]
    sub = df.sample(min(5000, len(df)), random_state=42)
    emo_cols = [c for c in df.columns if c.startswith("p_total_")]
    df_sum_clear = df[~df["has_any_unclear"]][emo_cols].sum(axis=1)
    df_sum_unclear = df[df["has_any_unclear"]][emo_cols].sum(axis=1)
    bins = np.linspace(0.0, 2.2, 23)
    ax.hist(df_sum_clear, bins=bins, density=True, alpha=0.55, color="#4C78A8", label="Clear Rater Consensus / Conflict")
    ax.hist(df_sum_unclear, bins=bins, density=True, alpha=0.65, color="#F58518", label="Flagged Very Unclear (>0 raters)")
    ax.set_xlabel("Expected Total Emotion Votes per Rater")
    ax.set_ylabel("Density")
    ax.set_title("B. Ground-Truth Clarity Signal vs Vote Conservation", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_goemotions_paradox.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eb = pd.read_parquet(PROC_DIR / "emobank_full.parquet")
    ge = pd.read_parquet(PROC_DIR / "goemotions_raw_agg.parquet")
    p1 = make_emobank_gap_figure(eb)
    p2 = make_goemotions_gap_figure(ge)
    print(f"Generated:\n- {p1}\n- {p2}")


if __name__ == "__main__":
    main()
