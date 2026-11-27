"""Locally-adaptive Mondrian Credal Calibration and Conformal Risk Control (CRC) for Alerting."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import discrete_wasserstein1, esteban_ray_polarization, gap_gated_divisiveness


def _fit_scale_predictor(
    pred_probs: np.ndarray,
    tgt_probs: np.ndarray,
    u_amb: np.ndarray,
    u_plur: np.ndarray,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Tuple[np.ndarray, float]:
    """Fit non-negative heteroscedastic error scale s_hat(x) strictly on dev_select."""
    w1 = discrete_wasserstein1(tgt_probs, pred_probs, score_values)
    X = np.column_stack([np.sqrt(np.clip(u_amb, 0.0, None)), np.sqrt(np.clip(u_plur, 0.0, None))])
    reg = Ridge(alpha=1.0).fit(X, w1)
    floor = max(0.03, 0.25 * float(np.median(w1)))
    return np.array([reg.intercept_, reg.coef_[0], reg.coef_[1]], dtype=float), floor


def _predict_scale(coef: np.ndarray, floor: float, u_amb: np.ndarray, u_plur: np.ndarray) -> np.ndarray:
    raw = coef[0] + coef[1] * np.sqrt(np.clip(u_amb, 0.0, None)) + coef[2] * np.sqrt(np.clip(u_plur, 0.0, None))
    return np.maximum(floor, raw)


def _regime_strata(
    pred_probs: np.ndarray,
    u_amb: np.ndarray,
    u_plur: np.ndarray,
    q85: float,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> np.ndarray:
    """Assign each prediction to observable Mondrian strata using a fixed dev_select q85 threshold."""
    v = np.asarray(score_values, dtype=float)
    mu = (pred_probs * v[None, :]).sum(axis=-1, keepdims=True)
    std = np.sqrt(np.clip((pred_probs * (v[None, :] - mu) ** 2).sum(axis=-1), 0.0, None))
    d_psi2 = gap_gated_divisiveness(pred_probs, score_values=score_values)
    is_polar = (u_plur >= q85) | (d_psi2 >= 0.06)
    is_low = (~is_polar) & (std < 0.35)
    out = np.ones(len(pred_probs), dtype=int)
    out[is_low] = 0
    out[is_polar] = 2
    return out


def mondrian_conformal_credal_sets(
    select_pred_probs: np.ndarray,
    select_tgt_probs: np.ndarray,
    select_u_amb: np.ndarray,
    select_u_plur: np.ndarray,
    cal_pred_probs: np.ndarray,
    cal_tgt_probs: np.ndarray,
    cal_u_amb: np.ndarray,
    cal_u_plur: np.ndarray,
    test_pred_probs: np.ndarray,
    test_tgt_probs: np.ndarray,
    test_u_amb: np.ndarray,
    test_u_plur: np.ndarray,
    test_categories: Sequence[str] | None = None,
    cal_categories: Sequence[str] | None = None,
    alpha: float = 0.10,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Dict[str, float]:
    """Evaluate locally-adaptive Mondrian W1 credal balls AND strictly nested CRC rating sets."""
    coef, floor = _fit_scale_predictor(select_pred_probs, select_tgt_probs, select_u_amb, select_u_plur, score_values)
    cal_s = _predict_scale(coef, floor, cal_u_amb, cal_u_plur)
    test_s = _predict_scale(coef, floor, test_u_amb, test_u_plur)

    cal_w1 = discrete_wasserstein1(cal_tgt_probs, cal_pred_probs, score_values)
    test_w1 = discrete_wasserstein1(test_tgt_probs, test_pred_probs, score_values)
    cal_r = cal_w1 / cal_s
    test_r = test_w1 / test_s

    # Fit regime threshold strictly on dev_select to preserve split-conformal exchangeability
    q85_sel = float(np.quantile(select_u_plur, 0.85)) if np.std(select_u_plur) > 1e-9 else float("inf")
    cal_strata = _regime_strata(cal_pred_probs, cal_u_amb, cal_u_plur, q85=q85_sel, score_values=score_values)
    test_strata = _regime_strata(test_pred_probs, test_u_amb, test_u_plur, q85=q85_sel, score_values=score_values)

    # Exact finite-sample split-conformal rank k = ceil((n+1)*(1-alpha)) per Mondrian stratum
    def _split_q(scores: np.ndarray, a: float) -> float:
        n = len(scores)
        if n == 0:
            return float("inf")
        k = int(np.ceil((n + 1) * (1.0 - a)))
        if k > n:
            return float("inf")
        return float(np.sort(scores)[max(k - 1, 0)])

    q_global = _split_q(cal_r, alpha)
    cov_global = float(np.mean(test_r <= q_global))
    rad_global = float(np.mean(q_global * test_s))

    q_raw = {g: _split_q(cal_r[cal_strata == g], alpha) for g in (0, 1, 2)}
    fallback_strata = int(sum(1 for qg in q_raw.values() if not np.isfinite(qg)))
    q_strata = {g: (qg if np.isfinite(qg) else q_global) for g, qg in q_raw.items()}
    test_q = np.array([q_strata[int(g)] for g in test_strata], dtype=float)
    cov_mondrian = float(np.mean(test_r <= test_q))
    rad_mondrian = float(np.mean(test_q * test_s))

    res: Dict[str, float] = {
        "target_coverage": round(1.0 - alpha, 3),
        "adaptive_w1_coverage": round(cov_global, 4),
        "adaptive_w1_mean_radius": round(rad_global, 4),
        "mondrian_w1_coverage": round(cov_mondrian, 4),
        "mondrian_w1_mean_radius": round(rad_mondrian, 4),
        "mondrian_fallback_strata": fallback_strata,
    }
    for g, name in ((0, "low"), (1, "diffuse"), (2, "polar_cand")):
        m = test_strata == g
        if np.any(m):
            res[f"mondrian_cov_{name}"] = round(float(np.mean(test_r[m] <= test_q[m])), 4)
            res[f"mondrian_rad_{name}"] = round(float(np.mean(test_q[m] * test_s[m])), 4)

    if test_categories is not None and cal_categories is not None:
        cal_cat = np.asarray(cal_categories)
        te_cat = np.asarray(test_categories)
        for cat in np.unique(te_cat):
            cm_cal = cal_cat == cat
            cm_te = te_cat == cat
            if np.sum(cm_cal) >= 20 and np.sum(cm_te) >= 10:
                qc = _split_q(cal_r[cm_cal], alpha)
                res[f"genre_cov_{cat}"] = round(float(np.mean(test_r[cm_te] <= qc)), 4)

    # Strictly nested CRC discrete rating sets covering >= (1 - alpha) expected human rater mass
    def _nested_missed_mass(p_hat: np.ndarray, p_true: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(score_values, dtype=float)
        mu_hat = (p_hat * v[None, :]).sum(axis=-1)
        widths = np.zeros(len(p_hat), dtype=float)
        missed = np.zeros(len(p_hat), dtype=float)
        for i in range(len(p_hat)):
            # Deterministic tie-break by proximity to predicted mean
            keys = (-p_hat[i], np.abs(v - mu_hat[i]))
            order = np.lexsort((keys[1], keys[0]))
            cum = 0.0
            sel_cov = 0.0
            w = 0
            for idx in order:
                cum += p_hat[i, idx]
                sel_cov += p_true[i, idx]
                w += 1
                if cum >= lam:
                    break
            widths[i] = w
            missed[i] = 1.0 - sel_cov
        return missed, widths

    lam_grid = np.linspace(0.50, 1.00, 101)
    n_cal = len(cal_pred_probs)
    lam_crc = 1.0
    for lam in lam_grid:
        missed_cal, _ = _nested_missed_mass(cal_pred_probs, cal_tgt_probs, float(lam))
        risk_bound = (float(missed_cal.sum()) + 1.0) / float(n_cal + 1)
        if risk_bound <= alpha:
            lam_crc = float(lam)
            break

    missed_te, widths_te = _nested_missed_mass(test_pred_probs, test_tgt_probs, lam_crc)
    res["crc_lambda"] = round(lam_crc, 4)
    res["crc_human_coverage"] = round(float(np.mean(1.0 - missed_te)), 4)
    res["crc_mean_width_bins"] = round(float(np.mean(widths_te)), 4)
    return res


def conformal_histogram_band(
    dev_pred_probs: np.ndarray,
    dev_tgt_probs: np.ndarray,
    test_pred_probs: np.ndarray,
    test_tgt_probs: np.ndarray,
    alpha: float = 0.10,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Dict[str, float]:
    """Backwards-compatible wrapper using exact (n+1) split conformal + nested CRC sets."""
    n = len(dev_pred_probs)
    half = max(1, n // 2)
    z = np.zeros(n, dtype=float)
    z_te = np.zeros(len(test_pred_probs), dtype=float)
    return mondrian_conformal_credal_sets(
        dev_pred_probs[:half], dev_tgt_probs[:half], z[:half], z[:half],
        dev_pred_probs[half:], dev_tgt_probs[half:], z[half:], z[half:],
        test_pred_probs, test_tgt_probs, z_te, z_te,
        alpha=alpha, score_values=score_values,
    )


def build_risk_features(
    pred_probs: np.ndarray,
    u_amb: np.ndarray,
    u_plur: np.ndarray,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> np.ndarray:
    """Build 7D interpretable feature matrix from single-pass distribution + latent uncertainties."""
    v = np.asarray(score_values, dtype=float)
    mu = (pred_probs * v[None, :]).sum(axis=-1)
    p_low = pred_probs[:, :2].sum(axis=-1)
    p_mid = pred_probs[:, 2] if pred_probs.shape[-1] == 5 else np.zeros(len(pred_probs))
    p_high = pred_probs[:, -2:].sum(axis=-1)
    d_psi2 = gap_gated_divisiveness(pred_probs, score_values)
    er08 = esteban_ray_polarization(pred_probs, score_values, alpha=0.8)
    return np.column_stack(
        [
            np.log1p(np.clip(u_plur, 0.0, None) * 100.0),
            np.sqrt(np.clip(u_amb, 0.0, None)),
            d_psi2,
            er08,
            np.minimum(p_low, p_high) - 0.5 * p_mid,
            p_low,
            np.abs(mu - 3.0),
        ]
    ).astype(float)


def fit_and_calibrate_alert_crc(
    select_X: np.ndarray,
    select_y: np.ndarray,
    cal_X: np.ndarray,
    cal_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.10,
    cal_severity: np.ndarray | None = None,
    test_severity: np.ndarray | None = None,
    use_supervised_combiner: bool = True,
) -> Dict[str, float]:
    """Calibrate Conditional-FNR CRC threshold on dev_cal and evaluate strictly out-of-sample on test."""
    sy = np.asarray(select_y, dtype=int)
    cy = np.asarray(cal_y, dtype=int)
    ty = np.asarray(test_y, dtype=int)

    if use_supervised_combiner and select_X.ndim == 2 and sy.sum() >= 3 and (len(sy) - sy.sum()) >= 3:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)).fit(select_X, sy)
        s_cal = clf.predict_proba(cal_X)[:, 1]
        s_te = clf.predict_proba(test_X)[:, 1]
    elif cal_X.ndim == 1:
        s_cal = np.asarray(cal_X, dtype=float)
        s_te = np.asarray(test_X, dtype=float)
    else:
        s_cal = cal_X[:, 0] + cal_X[:, 3]
        s_te = test_X[:, 0] + test_X[:, 3]

    cal_pos_scores = np.sort(s_cal[cy == 1])
    n_pos = len(cal_pos_scores)
    k_pos = int(np.floor((n_pos + 1) * alpha))
    tau_fnr = float(cal_pos_scores[k_pos - 1]) if (n_pos > 0 and k_pos >= 1) else -float("inf")

    alert_te = s_te >= tau_fnr
    te_pos = ty == 1
    fnr_te = float(np.mean(~alert_te[te_pos])) if np.any(te_pos) else 0.0
    vol_te = float(np.mean(alert_te))
    prec_te = float(np.mean(ty[alert_te])) if np.any(alert_te) else 0.0

    out: Dict[str, float] = {
        "target_fnr": alpha,
        "cal_n_pos": int(n_pos),
        "tau_fnr": round(tau_fnr, 5) if np.isfinite(tau_fnr) else -999.0,
        "test_fnr": round(fnr_te, 4),
        "test_recall": round(1.0 - fnr_te, 4),
        "alert_volume": round(vol_te, 4),
        "alert_precision": round(prec_te, 4),
    }

    if cal_severity is not None and test_severity is not None:
        # Unconditional Continuous Severity CRC across full dev_cal (e.g. L_sev in [0, 1])
        cal_sev = np.asarray(cal_severity, dtype=float)
        te_sev = np.asarray(test_severity, dtype=float)
        n_cal = len(s_cal)
        tau_grid = np.unique(np.concatenate([[-1.0], s_cal, [2.0]]))
        tau_sev = -float("inf")
        for tau in reversed(tau_grid):
            missed = cal_sev * (s_cal < tau)
            if (float(missed.sum()) + 1.0) / float(n_cal + 1) <= alpha:
                tau_sev = float(tau)
                break
        al_sev = s_te >= tau_sev
        out["sev_crc_test_risk"] = round(float(np.mean(te_sev * (~al_sev))), 4)
        out["sev_crc_alert_volume"] = round(float(np.mean(al_sev)), 4)
    return out
