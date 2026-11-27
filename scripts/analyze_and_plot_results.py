"""Post-hoc Conformal calibration, genre/regime breakdowns, and publication tables/charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal_sets import conformal_histogram_band


OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def summarize_conformal_emobank(emobank_dir: Path, protocol: str = "official", backbone: str = "roberta-base") -> pd.DataFrame:
    """Compute conformal W1 credal balls & 90% human-rating coverage sets across models/seeds."""
    b_slug = backbone.replace("/", "_")
    rows = []
    for run_dir in sorted(emobank_dir.glob(f"{protocol}_{b_slug}_*_seed*")):
        npz_path = run_dir / "test_predictions.npz"
        sum_path = run_dir / "summary.json"
        if not npz_path.exists() or not sum_path.exists():
            continue
        meta = json.loads(sum_path.read_text(encoding="utf-8"))
        mt = meta["model_type"]
        sd = meta["seed"]
        dat = np.load(npz_path, allow_pickle=True)
        dev_npz = run_dir / "dev_predictions.npz"
        r_pred = dat["r_probs"]  # (N, 3, 5)
        r_tgt = dat["tgt_r_probs"]
        if dev_npz.exists():
            dv = np.load(dev_npz, allow_pickle=True)
            cal_pred, cal_tgt = dv["r_probs"], dv["tgt_r_probs"]
            ev_pred, ev_tgt = r_pred, r_tgt
        else:
            N = len(r_pred)
            cal_idx = np.arange(0, N, 2)
            ev_idx = np.arange(1, N, 2)
            cal_pred, cal_tgt = r_pred[cal_idx], r_tgt[cal_idx]
            ev_pred, ev_tgt = r_pred[ev_idx], r_tgt[ev_idx]
        for j, d in enumerate(["V", "A", "D"]):
            res = conformal_histogram_band(
                cal_pred[:, j, :],
                cal_tgt[:, j, :],
                ev_pred[:, j, :],
                ev_tgt[:, j, :],
                alpha=0.10,
            )
            rows.append({"model_type": mt, "seed": sd, "dim": d, **res})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    agg = (
        df.groupby("model_type")[
            ["conformal_w1_radius", "test_w1_ball_coverage", "credal_set_mean_human_coverage", "credal_set_mean_width_bins"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    return agg


def generate_full_report() -> None:
    report_lines = ["# 002idea Experimental Summary\n"]

    for fname, sec in [
        ("main_official_summary.csv", "## 1. EmoBank Dual-Perspective Official Benchmark (roberta-base)\n"),
        ("main_official_microsoft_mpnet-base_summary.csv", "## 1b. EmoBank Cross-Backbone Generalization (microsoft/mpnet-base)\n"),
        ("doc_disjoint_summary.csv", "## 1c. EmoBank Document-Disjoint Leakage-Free Benchmark\n"),
        ("logo_summary.csv", "## 1d. EmoBank Leave-One-Genre-Out (LOGO) Transfer\n"),
        ("ablation_summary.csv", "## 1e. EmoBank PAT-Net Component & Loss Ablations\n"),
    ]:
        p = OUT_ROOT / "emobank" / fname
        if p.exists():
            report_lines.append(sec)
            report_lines.append(pd.read_csv(p).to_markdown(index=False))
            report_lines.append("\n")

    sig_path = OUT_ROOT / "emobank" / "main_official_significance.json"
    if sig_path.exists():
        report_lines.append("## 2. Paired Significance vs PAT-Net\n```json\n" + sig_path.read_text(encoding="utf-8") + "\n```\n")

    conf_df = summarize_conformal_emobank(OUT_ROOT / "emobank")
    if len(conf_df) > 0:
        conf_df.to_csv(OUT_ROOT / "emobank" / "conformal_summary.csv", index=False)
        report_lines.append("## 3. Conformal Credal Prediction Sets (alpha = 0.10)\n")
        report_lines.append(conf_df.to_markdown(index=False))
        report_lines.append("\n")

    for subdir, title in [
        ("goemotions", "## 4. GoEmotions Raw Ambiguity vs Pluralism Double Dissociation\n"),
        ("epic", "## 5. EPIC Perspectivist Irony Disentanglement\n"),
        ("brighter", "## 6. BRIGHTER Zero-Inflated Ordinal Emotion Intensity\n"),
    ]:
        s_dir = OUT_ROOT / subdir
        if s_dir.exists():
            ms_p = s_dir / "multi_seed_summary.csv"
            rows = []
            for sp in sorted(s_dir.glob("*/summary.json")):
                d = json.loads(sp.read_text(encoding="utf-8"))
                rec = {k: v for k, v in d.items() if k in ("lang", "target_mode", "model_type", "seed")}
                rec.update(d["test_metrics"])
                rows.append(rec)
            if rows:
                df_all = pd.DataFrame(rows)
                df_all.to_csv(s_dir / "summary_all.csv", index=False)
                report_lines.append(title)
                if ms_p.exists():
                    report_lines.append("### Multi-Seed Aggregated Summary\n")
                    report_lines.append(pd.read_csv(ms_p).to_markdown(index=False))
                    report_lines.append("\n### Per-Seed Detail\n")
                report_lines.append(df_all.to_markdown(index=False))
                report_lines.append("\n")

    out_md = OUT_ROOT / "EXPERIMENT_REPORT.md"
    out_md.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    generate_full_report()
