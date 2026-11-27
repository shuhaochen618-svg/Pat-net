"""Deep stratified diagnostic of EmoBank predictions across polarization, ambiguity, perspective shift, and genres."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import discrete_jsd, discrete_wasserstein1, safe_pearson


OUT_EB = Path(__file__).resolve().parents[1] / "outputs" / "emobank"


def two_point_dirac_w1(mu: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    """Exact 1D W1 between deterministic scalar prediction mu in [1,5] and true rating histogram."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float).reshape(1, 1, 5)
    return np.sum(p_true * np.abs(v - mu[..., None]), axis=-1)


def main() -> None:
    rows = []
    for run_dir in sorted(OUT_EB.glob("official_roberta-base_*_seed42")):
        npz_p = run_dir / "test_predictions.npz"
        sum_p = run_dir / "summary.json"
        if not npz_p.exists() or not sum_p.exists():
            continue
        meta = json.loads(sum_p.read_text(encoding="utf-8"))
        mt = meta["model_type"]
        dat = np.load(npz_p, allow_pickle=True)

        p_t = dat["tgt_r_probs"]  # (N, 3, 5)
        p_q = dat["r_probs"]      # (N, 3, 5)
        m_t = dat["tgt_r_mean"]
        m_q = dat["r_disc_mean"]
        s_t = dat["tgt_r_std"]
        s_q = dat["r_disc_std"]
        wm_t = dat["tgt_w_mean"]
        wm_q = dat["w_disc_mean"]
        u_amb = dat["r_u_amb"]
        u_plur = dat["r_u_plur"]

        # True W1: for scalar_mse, also report exact deterministic scalar W1!
        w1 = discrete_wasserstein1(p_t, p_q)
        w1_dirac = two_point_dirac_w1(m_q, p_t) if mt == "scalar_mse" else w1
        jsd = discrete_jsd(p_t, p_q)

        # Stratified masks across all (N, 3) flattened dims
        p_low = p_t[..., 0] + p_t[..., 1]
        p_high = p_t[..., 3] + p_t[..., 4]
        bipole = np.minimum(p_low, p_high) > 0
        neutral_band = (m_t >= 2.5) & (m_t <= 3.5)
        unan_neutral = neutral_band & (s_t < 0.05)
        polar_neutral = neutral_band & ((s_t >= 0.8) | bipole)
        diffuse_unimodal = neutral_band & (s_t >= 0.35) & (s_t < 0.8) & (~bipole)

        # High perspective-shift subset (|R - W| >= 0.5)
        high_shift = np.abs(m_t - wm_t) >= 0.5
        shift_mae_high = float(np.mean(np.abs((m_t - wm_t)[high_shift] - (m_q - wm_q)[high_shift])))

        # Polarization AUROC using total spread (s_q) vs u_plur vs u_amb
        eval_np = unan_neutral | polar_neutral
        auc_std = float(roc_auc_score(polar_neutral[eval_np].astype(int), s_q[eval_np])) if mt != "scalar_mse" else 0.5

        # Harder test: Bimodal Conflict vs Diffuse Unimodal!
        eval_bd = bipole | diffuse_unimodal
        auc_bd_plur = float(roc_auc_score(bipole[eval_bd].astype(int), u_plur[eval_bd])) if (u_plur[eval_bd].std() > 1e-8) else 0.5
        auc_bd_std = float(roc_auc_score(bipole[eval_bd].astype(int), s_q[eval_bd])) if (s_q[eval_bd].std() > 1e-8) else 0.5

        rows.append(
            {
                "model": mt,
                "r_w1_all": round(float(w1_dirac.mean()), 4),
                "r_w1_bimodal": round(float(w1_dirac[bipole].mean()), 4),
                "r_jsd_all": round(float(jsd.mean()), 4),
                "r_jsd_bimodal": round(float(jsd[bipole].mean()), 4),
                "r_pearson": round(float(meta["test_metrics"]["r_mean_pearson"]), 4),
                "r_std_mae": round(float(np.mean(np.abs(s_t - s_q))), 4),
                "pol_auc_std": round(auc_std, 4),
                "bimod_vs_diff_auc_std": round(auc_bd_std, 4),
                "bimod_vs_diff_auc_uplur": round(auc_bd_plur, 4),
                "shift_mae_all": round(float(np.mean(np.abs((m_t - wm_t) - (m_q - wm_q)))), 4),
                "shift_mae_high": round(shift_mae_high, 4),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
