"""Clean distributional, axiomatic polarization, perspective-shift, operational alerting, and cluster-bootstrap metrics."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def concordance_cc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient (CCC)."""
    if len(y_true) < 2:
        return float("nan")
    mu_t, mu_p = float(np.mean(y_true)), float(np.mean(y_pred))
    var_t, var_p = float(np.var(y_true)), float(np.var(y_pred))
    cov = float(np.mean((y_true - mu_t) * (y_pred - mu_p)))
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    return (2.0 * cov / denom) if denom > 1e-12 else 0.0


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def discrete_wasserstein1(
    p: np.ndarray,
    q: np.ndarray,
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """Exact 1D W1 along arbitrary monotonic bin grid; p, q shape (..., K)."""
    steps = np.diff(np.asarray(score_values, dtype=float))
    cdf_p = np.cumsum(p, axis=-1)[..., :-1]
    cdf_q = np.cumsum(q, axis=-1)[..., :-1]
    return np.sum(np.abs(cdf_p - cdf_q) * steps, axis=-1)


def dirac_wasserstein1(
    mu: np.ndarray,
    tgt_probs: np.ndarray,
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """Exact continuous 1D W1 between deterministic Dirac scalar mu and discrete histogram tgt_probs."""
    v = np.asarray(score_values, dtype=float).reshape(*([1] * mu.ndim), -1)
    return np.sum(tgt_probs * np.abs(v - mu[..., None]), axis=-1)


def discrete_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Base-2 Jensen-Shannon Divergence bounded in [0, 1]."""
    p_s = np.clip(p, eps, 1.0)
    p_s = p_s / p_s.sum(axis=-1, keepdims=True)
    q_s = np.clip(q, eps, 1.0)
    q_s = q_s / q_s.sum(axis=-1, keepdims=True)
    m = 0.5 * (p_s + q_s)
    kl_pm = np.sum(p_s * (np.log2(p_s) - np.log2(m)), axis=-1)
    kl_qm = np.sum(q_s * (np.log2(q_s) - np.log2(m)), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def esteban_ray_polarization(
    p: np.ndarray,
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
    alpha: float = 0.8,
) -> np.ndarray:
    """Axiomatic Esteban-Ray (1994) polarization functional: sum_{i,j} p_i^{1+alpha} p_j |v_i - v_j|."""
    v = np.asarray(score_values, dtype=float)
    d = np.abs(v[:, None] - v[None, :])
    p_c = np.clip(p, 0.0, 1.0)
    return np.einsum("...i,ij,...j->...", p_c ** (1.0 + alpha), d, p_c)


def gap_gated_divisiveness(
    p: np.ndarray,
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """Metric-aware divisiveness 0.5 * sum_{i,j} p_i p_j max(0, |v_i - v_j| - 1)^2."""
    v = np.asarray(score_values, dtype=float)
    psi = np.maximum(np.abs(v[:, None] - v[None, :]) - 1.0, 0.0) ** 2
    return 0.5 * np.einsum("...i,ij,...j->...", p, psi, p)


def linear_divisiveness(
    p: np.ndarray,
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """DC-LDL pairwise divisiveness 0.5 * sum_{i,j} p_i p_j |v_i - v_j|."""
    v = np.asarray(score_values, dtype=float)
    d = np.abs(v[:, None] - v[None, :])
    return 0.5 * np.einsum("...i,ij,...j->...", p, d, p)


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_c = np.clip(p, eps, 1.0)
    return -np.sum(np.where(p > 0, p_c * np.log2(p_c), 0.0), axis=-1)


def extract_clean_polarization_masks(
    p_t: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Classify ground-truth ordinal histograms without the bell-trap bug ([2,3,3,3,4])."""
    p_low = p_t[:, :2].sum(-1)
    p_mid = p_t[:, 2] if p_t.shape[-1] == 5 else np.zeros(len(p_t))
    p_high = p_t[:, -2:].sum(-1)
    neut = (tgt_mean >= 2.5) & (tgt_mean <= 3.5)

    bell_trap = (
        (p_t[:, 0] == 0)
        & (p_t[:, -1] == 0)
        & (p_t[:, 1] > 0)
        & (p_t[:, -2] > 0)
        & (p_mid >= np.maximum(p_low, p_high))
    )
    valley_strict = (p_low >= 0.20) & (p_high >= 0.20) & (p_mid < np.minimum(p_low, p_high))
    valley_or_dip = (p_low >= 0.20) & (p_high >= 0.20) & ((p_mid < np.minimum(p_low, p_high)) | (tgt_std >= 0.75))
    diffuse_clean = neut & (~valley_or_dip) & (~bell_trap) & (tgt_std >= 0.45) & (tgt_std < 0.95) & (p_mid >= 0.20)
    nonvod_high_std = neut & (~valley_or_dip) & (tgt_std >= 0.60)
    unanimous_neut = neut & (tgt_std < 0.05)
    distress_neut = neut & (p_low >= 0.20)

    return {
        "neut": neut,
        "bell_trap": bell_trap,
        "valley_strict": valley_strict,
        "valley_or_dip": valley_or_dip,
        "diffuse_clean": diffuse_clean,
        "nonvod_high_std": nonvod_high_std,
        "unanimous_neut": unanimous_neut,
        "distress_neut": distress_neut,
    }


def _safe_auc(y_bin: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_bin, scores)) if (len(np.unique(y_bin)) == 2 and np.std(scores) > 1e-12) else 0.5


def _safe_ap(y_bin: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_bin, scores)) if (len(np.unique(y_bin)) == 2 and np.std(scores) > 1e-12) else float(np.mean(y_bin))


def evaluate_binary_alert_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    budgets: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    cost_ratios: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
) -> Dict[str, float]:
    """Compute operational alerting metrics: Recall/Precision@Budget, FNR@FPR, FPR@Recall, Asymmetric Cost."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    out: Dict[str, float] = {
        "base_rate": float(n_pos / max(n, 1)),
        "auroc": _safe_auc(y, s),
        "auprc": _safe_ap(y, s),
    }
    if n_pos == 0 or n_neg == 0 or np.std(s) < 1e-12:
        return out

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]

    for b in budgets:
        k = max(1, min(n, int(round(b * n))))
        tp_k = int(y_sorted[:k].sum())
        pct = int(round(b * 100))
        out[f"recall_at_{pct}pct"] = float(tp_k / n_pos)
        out[f"precision_at_{pct}pct"] = float(tp_k / k)

    # Sweep all thresholds via cumulative counts
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    tpr = tp_cum / float(n_pos)
    fpr = fp_cum / float(n_neg)
    fnr = 1.0 - tpr

    for target_fpr in (0.05, 0.10):
        idx = np.where(fpr <= target_fpr)[0]
        val = float(fnr[idx[-1]]) if len(idx) > 0 else 1.0
        out[f"fnr_at_fpr_{int(target_fpr * 100)}pct"] = val

    for target_rec in (0.80, 0.90):
        idx = np.where(tpr >= target_rec)[0]
        val = float(fpr[idx[0]]) if len(idx) > 0 else 1.0
        out[f"fpr_at_rec_{int(target_rec * 100)}pct"] = val

    for ratio in cost_ratios:
        # Normalized cost per item: (ratio * FN + FP) / (ratio * n_pos)
        costs = (ratio * (n_pos - tp_cum) + fp_cum) / max(ratio * n_pos, 1.0)
        out[f"min_norm_cost_r{int(ratio)}"] = float(np.min(costs))
    return out


def evaluate_perspective_predictions(
    tgt_probs: np.ndarray,
    pred_probs: np.ndarray,
    tgt_mean: np.ndarray,
    pred_mean: np.ndarray,
    tgt_std: np.ndarray,
    pred_std: np.ndarray,
    has_hist: np.ndarray,
    u_plur: np.ndarray | None = None,
    u_amb: np.ndarray | None = None,
    dim_names: Tuple[str, ...] = ("V", "A", "D"),
    score_values: Tuple[float, ...] = (1, 2, 3, 4, 5),
    prefix: str = "r",
) -> Dict[str, float]:
    """Compute scalar, distributional, clean valley-bimodal, and high-std shape disentanglement metrics."""
    out: Dict[str, float] = {}
    w1_list, bimod_w1_list, jsd_list, std_mae_list, r_list, ccc_list = [], [], [], [], [], []
    pol_auc_std_list, pol_ap_std_list, pol_auc_plur_list, pol_auc_amb_list = [], [], [], []
    bd_auc_plur_list, bd_auc_std_list, bd_auc_amb_list = [], [], []
    hs_auc_std_list, hs_auc_plur_list, hs_ap_plur_list, hs_auc_uamb_list, hs_auc_er_list, hs_ap_er_list = [], [], [], [], [], []

    hmask = has_hist.astype(bool)
    is_dirac = bool(np.all(pred_std < 1e-6))

    for j, d in enumerate(dim_names):
        r_val = safe_pearson(tgt_mean[:, j], pred_mean[:, j])
        ccc_val = concordance_cc(tgt_mean[:, j], pred_mean[:, j])
        std_mae = float(np.mean(np.abs(tgt_std[:, j] - pred_std[:, j])))
        std_r = safe_pearson(tgt_std[:, j], pred_std[:, j])
        out[f"{prefix}_{d}_pearson"] = r_val
        out[f"{prefix}_{d}_ccc"] = ccc_val
        out[f"{prefix}_{d}_std_mae"] = std_mae
        out[f"{prefix}_{d}_std_pearson"] = std_r
        r_list.append(r_val)
        ccc_list.append(ccc_val)
        std_mae_list.append(std_mae)

        if np.any(hmask):
            p_t = tgt_probs[hmask, j, :]
            p_q = pred_probs[hmask, j, :]
            w1_vec = (
                dirac_wasserstein1(pred_mean[hmask, j], p_t, score_values)
                if is_dirac
                else discrete_wasserstein1(p_t, p_q, score_values)
            )
            w1 = float(np.mean(w1_vec))
            jsd = float("nan") if is_dirac else float(np.mean(discrete_jsd(p_t, p_q)))
            out[f"{prefix}_{d}_w1"] = w1
            out[f"{prefix}_{d}_jsd"] = jsd
            w1_list.append(w1)
            if not np.isnan(jsd):
                jsd_list.append(jsd)

            masks = extract_clean_polarization_masks(p_t, tgt_mean[hmask, j], tgt_std[hmask, j])
            vod = masks["valley_or_dip"]
            if np.any(vod):
                bw1 = float(np.mean(w1_vec[vod]))
                out[f"{prefix}_{d}_bimod_w1"] = bw1
                bimod_w1_list.append(bw1)

            # 1) Neutral-band valley_or_dip vs unanimous neutral
            eval_un = masks["unanimous_neut"] | (masks["neut"] & vod)
            if eval_un.sum() >= 10 and (masks["neut"] & vod).sum() >= 3 and masks["unanimous_neut"].sum() >= 3:
                y_bin = (masks["neut"] & vod)[eval_un].astype(int)
                auc_s = _safe_auc(y_bin, pred_std[hmask, j][eval_un])
                ap_s = _safe_ap(y_bin, pred_std[hmask, j][eval_un])
                out[f"{prefix}_{d}_pol_auroc_by_std"] = auc_s
                out[f"{prefix}_{d}_pol_auprc_by_std"] = ap_s
                pol_auc_std_list.append(auc_s)
                pol_ap_std_list.append(ap_s)
                if u_plur is not None:
                    auc_p = _safe_auc(y_bin, u_plur[hmask, j][eval_un])
                    out[f"{prefix}_{d}_pol_auroc_by_uplur"] = auc_p
                    out[f"{prefix}_{d}_pol_auprc_by_uplur"] = _safe_ap(y_bin, u_plur[hmask, j][eval_un])
                    pol_auc_plur_list.append(auc_p)
                if u_amb is not None:
                    auc_a = _safe_auc(y_bin, u_amb[hmask, j][eval_un])
                    out[f"{prefix}_{d}_pol_auroc_by_uamb"] = auc_a
                    pol_auc_amb_list.append(auc_a)

            # 2) Clean Bimodal vs Clean Diffuse Unimodal (bell_trap excluded)
            bd_idx = (masks["neut"] & vod) | masks["diffuse_clean"]
            if bd_idx.sum() >= 10 and (masks["neut"] & vod).sum() >= 3 and masks["diffuse_clean"].sum() >= 3:
                y_bd = (masks["neut"] & vod)[bd_idx].astype(int)
                bd_auc_std_list.append(_safe_auc(y_bd, pred_std[hmask, j][bd_idx]))
                if u_plur is not None:
                    bd_auc_plur_list.append(_safe_auc(y_bd, u_plur[hmask, j][bd_idx]))
                if u_amb is not None:
                    bd_auc_amb_list.append(_safe_auc(y_bd, u_amb[hmask, j][bd_idx]))

            # 3) HARD Matched High-Std Shape Test (std>=0.60 nonvod vs valley_or_dip)
            hs_idx = (masks["neut"] & vod) | masks["nonvod_high_std"]
            if hs_idx.sum() >= 10 and (masks["neut"] & vod).sum() >= 3 and masks["nonvod_high_std"].sum() >= 3:
                y_hs = (masks["neut"] & vod)[hs_idx].astype(int)
                hs_auc_std_list.append(_safe_auc(y_hs, pred_std[hmask, j][hs_idx]))
                if not is_dirac:
                    er_sc = esteban_ray_polarization(p_q[hs_idx], score_values=score_values, alpha=0.8)
                    hs_auc_er_list.append(_safe_auc(y_hs, er_sc))
                    hs_ap_er_list.append(_safe_ap(y_hs, er_sc))
                if u_plur is not None:
                    hs_auc_plur_list.append(_safe_auc(y_hs, u_plur[hmask, j][hs_idx]))
                    hs_ap_plur_list.append(_safe_ap(y_hs, u_plur[hmask, j][hs_idx]))
                if u_amb is not None:
                    hs_auc_uamb_list.append(_safe_auc(y_hs, u_amb[hmask, j][hs_idx]))

    out[f"{prefix}_mean_pearson"] = float(np.mean(r_list))
    out[f"{prefix}_mean_ccc"] = float(np.mean(ccc_list))
    out[f"{prefix}_mean_std_mae"] = float(np.mean(std_mae_list))
    if w1_list:
        out[f"{prefix}_mean_w1"] = float(np.mean(w1_list))
    out[f"{prefix}_mean_jsd"] = float(np.mean(jsd_list)) if jsd_list else float("nan")
    if bimod_w1_list:
        out[f"{prefix}_bimod_w1"] = float(np.mean(bimod_w1_list))
    if pol_auc_std_list:
        out[f"{prefix}_mean_pol_auroc"] = float(np.mean(pol_auc_std_list))
        out[f"{prefix}_mean_pol_auprc"] = float(np.mean(pol_ap_std_list))
        out[f"{prefix}_mean_pol_auroc_by_std"] = float(np.mean(pol_auc_std_list))
    if pol_auc_plur_list:
        out[f"{prefix}_mean_pol_auroc_by_uplur"] = float(np.mean(pol_auc_plur_list))
    if pol_auc_amb_list:
        out[f"{prefix}_mean_pol_auroc_by_uamb"] = float(np.mean(pol_auc_amb_list))
    if bd_auc_std_list:
        out[f"{prefix}_bimod_vs_diff_auroc_by_std"] = float(np.mean(bd_auc_std_list))
    if bd_auc_plur_list:
        out[f"{prefix}_bimod_vs_diff_auroc_by_uplur"] = float(np.mean(bd_auc_plur_list))
    if bd_auc_amb_list:
        out[f"{prefix}_bimod_vs_diff_auroc_by_uamb"] = float(np.mean(bd_auc_amb_list))
    if hs_auc_std_list:
        out[f"{prefix}_hard_shape_auroc_by_std"] = float(np.mean(hs_auc_std_list))
    if hs_auc_plur_list:
        out[f"{prefix}_hard_shape_auroc_by_uplur"] = float(np.mean(hs_auc_plur_list))
        out[f"{prefix}_hard_shape_auprc_by_uplur"] = float(np.mean(hs_ap_plur_list))
    if hs_auc_uamb_list:
        out[f"{prefix}_hard_shape_auroc_by_uamb"] = float(np.mean(hs_auc_uamb_list))
    if hs_auc_er_list:
        out[f"{prefix}_hard_shape_auroc_by_er08"] = float(np.mean(hs_auc_er_list))
        out[f"{prefix}_hard_shape_auprc_by_er08"] = float(np.mean(hs_ap_er_list))
    return out


def evaluate_perspective_shift(
    w_mean_true: np.ndarray,
    r_mean_true: np.ndarray,
    w_mean_pred: np.ndarray,
    r_mean_pred: np.ndarray,
    dim_names: Tuple[str, ...] = ("V", "A", "D"),
) -> Dict[str, float]:
    """Evaluate Writer->Reader perspective shift & Writer-calm / Reader-shock offense risk."""
    true_diff = r_mean_true - w_mean_true
    pred_diff = r_mean_pred - w_mean_pred
    out: Dict[str, float] = {}
    maes, rs, high_aucs = [], [], []
    for j, d in enumerate(dim_names):
        mae = float(np.mean(np.abs(true_diff[:, j] - pred_diff[:, j])))
        r_val = safe_pearson(true_diff[:, j], pred_diff[:, j])
        high_lbl = (np.abs(true_diff[:, j]) >= 0.5).astype(int)
        out[f"shift_{d}_mae"] = mae
        out[f"shift_{d}_pearson"] = r_val
        maes.append(mae)
        rs.append(r_val)
        if 3 <= high_lbl.sum() <= len(high_lbl) - 3:
            auc = _safe_auc(high_lbl, np.abs(pred_diff[:, j]))
            out[f"shift_{d}_high_auroc"] = auc
            high_aucs.append(auc)
    out["shift_mean_mae"] = float(np.mean(maes))
    out["shift_mean_pearson"] = float(np.mean(rs))
    if high_aucs:
        out["shift_mean_high_auroc"] = float(np.mean(high_aucs))

    # Asymmetric offense alert: Writer calm (V_W >= 2.8) while Reader distressed (V_R <= 2.3) or W_V - R_V >= 0.5
    off_lbl = ((w_mean_true[:, 0] >= 2.8) & (r_mean_true[:, 0] <= 2.3)) | ((w_mean_true[:, 0] - r_mean_true[:, 0]) >= 0.5)
    off_score = w_mean_pred[:, 0] - r_mean_pred[:, 0]
    if 3 <= off_lbl.sum() <= len(off_lbl) - 3:
        out["asymm_offense_auroc"] = _safe_auc(off_lbl.astype(int), off_score)
        out["asymm_offense_auprc"] = _safe_ap(off_lbl.astype(int), off_score)
    return out


def cluster_bootstrap_paired_test(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    groups: Sequence[str],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided cluster-paired bootstrap over document/thread groups (fixes i.i.d. sentence inflation)."""
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    diff = a - b
    g_arr = np.asarray(groups)
    uniq_g = np.unique(g_arr)
    g_indices = [np.where(g_arr == g)[0] for g in uniq_g]
    n_g = len(g_indices)

    obs = float(stat_fn(diff))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for bi in range(n_boot):
        draw = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([g_indices[j] for j in draw])
        boots[bi] = float(stat_fn(diff[idx]))

    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p_left = float(np.mean(boots <= 0.0))
    p_right = float(np.mean(boots >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    return {
        "diff_mean": obs,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_two_sided": p_two,
        "n_clusters": n_g,
    }


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Apply exact step-down Holm-Bonferroni family-wise error rate correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj
