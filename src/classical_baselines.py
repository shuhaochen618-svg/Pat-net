"""Non-deep-learning baselines: C1_SparseRidge and C2_HybridGBDT with OOF token-polarity dispersion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.brighter_exp import ORD_SCORES as BR_SCORES
from src.brighter_exp import _fit_temperature_on_sel, _get_brighter_4way, _temp_scale_probs, evaluate_brighter
from src.conformal_sets import fit_and_calibrate_alert_crc
from src.train_emobank import evaluate_operational_and_conformal
from src.datasets import DIMS, fit_struct_scaler_for_splits, get_emobank_4way_splits, load_emobank_dataframe
from src.epic_exp import evaluate_epic, get_epic_4way_splits
from src.metrics import (
    esteban_ray_polarization,
    evaluate_binary_alert_metrics,
    evaluate_perspective_predictions,
    evaluate_perspective_shift,
    extract_clean_polarization_masks,
)
from src.paths import OUT_DIR, PROC_DIR
from src.struct_features import extract_struct_matrix

_EN_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def _jieba_or_regex_tokenize(text: str) -> List[str]:
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        try:
            import jieba  # type: ignore

            return [t.strip() for t in jieba.lcut(text) if t.strip()]
        except Exception:
            return list(text.strip())
    return _EN_WORD_RE.findall(text.lower())


def _project_simplex(probs: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(probs, eps, None)
    return p / p.sum(axis=-1, keepdims=True)


def _moments_from_probs(probs: np.ndarray, v: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)) -> Tuple[np.ndarray, np.ndarray]:
    gv = np.asarray(v, dtype=float).reshape(1, 1, -1)
    mean = (probs * gv).sum(axis=-1)
    var = np.clip((probs * (gv - mean[..., None]) ** 2).sum(axis=-1), 0.0, None)
    return mean, np.sqrt(var)


def _build_sparse_design(
    tr_texts: Sequence[str],
    *other_texts: Sequence[str],
    word_max: int = 25000,
    char_max: int = 20000,
):
    w_vec = TfidfVectorizer(
        tokenizer=_jieba_or_regex_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=word_max,
    )
    c_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=char_max,
    )
    Xw_tr = w_vec.fit_transform(tr_texts)
    Xc_tr = c_vec.fit_transform(tr_texts)
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
    outs = [X_tr]
    for ts in other_texts:
        outs.append(hstack([w_vec.transform(ts), c_vec.transform(ts)]).tocsr())
    return tuple(outs)


def _token_polarity_dispersion(
    X_csr,
    coefs_matrix: np.ndarray,  # (D_feat, T)
) -> np.ndarray:
    """Compute signed positive/negative/co-activation/variance token polarity statistics without external lexicons."""
    X_abs = X_csr
    n_samples = X_abs.shape[0]
    n_t = coefs_matrix.shape[1]
    feats = np.zeros((n_samples, n_t * 6), dtype=np.float32)

    indptr = X_abs.indptr
    indices = X_abs.indices
    data = X_abs.data
    for i in range(n_samples):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        cols = indices[s:e]
        w = data[s:e]  # (nnz,)
        c = w[:, None] * coefs_matrix[cols, :]  # (nnz, T)
        pos = np.sum(np.maximum(c, 0.0), axis=0)
        neg = np.sum(np.maximum(-c, 0.0), axis=0)
        coact = np.minimum(pos, neg)
        prod = np.sqrt(np.maximum(pos * neg, 0.0))
        spread = np.max(c, axis=0) - np.min(c, axis=0)
        std_c = np.std(c, axis=0)
        base = np.concatenate([pos, neg, coact, prod, spread, std_c], axis=0)
        feats[i] = base
    return feats


def _extract_oof_and_gbdt_features_emobank(
    X_tr,
    X_sel,
    X_cal,
    X_te,
    df_tr: pd.DataFrame,
    sf_tr: np.ndarray,
    sf_sel: np.ndarray,
    sf_cal: np.ndarray,
    sf_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build leakage-free OOF sparse Ridge + token dispersion + SVD + structural matrix."""
    n_tr = len(df_tr)
    groups = df_tr["document"].fillna(df_tr["id"].astype(str)).astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)

    y_mean_r = np.stack([df_tr[f"r_{d}_pub_mean"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    y_std_r = np.stack([df_tr[f"r_{d}_pub_std"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    y_mean_w = np.stack([df_tr[f"w_{d}_pub_mean"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    y_std_w = np.stack([df_tr[f"w_{d}_pub_std"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)

    r_probs_tr = np.stack(
        [np.stack([df_tr[f"r_{d}_p{k}"].fillna(0.2).to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1) for d in DIMS],
        axis=1,
    )
    w_probs_tr = np.stack(
        [np.stack([df_tr[f"w_{d}_p{k}"].fillna(0.2).to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1) for d in DIMS],
        axis=1,
    )
    r_has = df_tr["r_V_has_hist"].astype(bool).to_numpy()
    w_has = df_tr["w_V_has_hist"].astype(bool).to_numpy()

    # Mean-subtract center for linear polarity weights so positive vs negative token signs are well-defined!
    y_center_r = y_mean_r - y_mean_r.mean(axis=0, keepdims=True)
    y_center_w = y_mean_w - y_mean_w.mean(axis=0, keepdims=True)
    Y_lin = np.hstack([y_center_r, y_center_w])  # (n_tr, 6)

    # 12 moment targets + 30 histogram targets
    Y_mom = np.hstack([y_mean_w, y_std_w, y_mean_r, y_std_r])  # 12

    oof_mom = np.zeros((n_tr, 12), dtype=np.float32)
    oof_disp = np.zeros((n_tr, 36), dtype=np.float32)  # 6*6
    oof_w_probs = np.zeros((n_tr, 15), dtype=np.float32)
    oof_r_probs = np.zeros((n_tr, 15), dtype=np.float32)

    # Concatenate struct feats to sparse for Ridge C1 accuracy
    X_tr_s = hstack([X_tr, sf_tr]).tocsr()
    X_sel_s = hstack([X_sel, sf_sel]).tocsr()
    X_cal_s = hstack([X_cal, sf_cal]).tocsr()
    X_te_s = hstack([X_te, sf_te]).tocsr()

    for tr_i, va_i in gkf.split(X_tr_s, groups=groups):
        reg_mom = Ridge(alpha=3.0).fit(X_tr_s[tr_i], Y_mom[tr_i])
        oof_mom[va_i] = reg_mom.predict(X_tr_s[va_i])

        reg_pol = Ridge(alpha=3.0).fit(X_tr[tr_i], Y_lin[tr_i])
        oof_disp[va_i] = _token_polarity_dispersion(X_tr[va_i], reg_pol.coef_.T)

        rm_mask = r_has[tr_i]
        wm_mask = w_has[tr_i]
        reg_rp = Ridge(alpha=3.0).fit(X_tr_s[tr_i][rm_mask], r_probs_tr[tr_i][rm_mask].reshape(-1, 15))
        reg_wp = Ridge(alpha=3.0).fit(X_tr_s[tr_i][wm_mask], w_probs_tr[tr_i][wm_mask].reshape(-1, 15))
        oof_r_probs[va_i] = _project_simplex(reg_rp.predict(X_tr_s[va_i]).reshape(-1, 3, 5)).reshape(-1, 15)
        oof_w_probs[va_i] = _project_simplex(reg_wp.predict(X_tr_s[va_i]).reshape(-1, 3, 5)).reshape(-1, 15)

    # Full-train fits for sel/cal/te
    full_mom = Ridge(alpha=3.0).fit(X_tr_s, Y_mom)
    full_pol = Ridge(alpha=3.0).fit(X_tr, Y_lin)
    full_rp = Ridge(alpha=3.0).fit(X_tr_s[r_has], r_probs_tr[r_has].reshape(-1, 15))
    full_wp = Ridge(alpha=3.0).fit(X_tr_s[w_has], w_probs_tr[w_has].reshape(-1, 15))

    ridge_preds = {}
    for tag, Xs in (("sel", X_sel_s), ("cal", X_cal_s), ("te", X_te_s)):
        ridge_preds[f"{tag}_mom"] = full_mom.predict(Xs).astype(np.float32)
        ridge_preds[f"{tag}_r_probs"] = _project_simplex(full_rp.predict(Xs).reshape(-1, 3, 5)).astype(np.float32)
        ridge_preds[f"{tag}_w_probs"] = _project_simplex(full_wp.predict(Xs).reshape(-1, 3, 5)).astype(np.float32)

    svd = TruncatedSVD(n_components=48, random_state=42)
    Z_tr = svd.fit_transform(X_tr).astype(np.float32)
    Z_sel = svd.transform(X_sel).astype(np.float32)
    Z_cal = svd.transform(X_cal).astype(np.float32)
    Z_te = svd.transform(X_te).astype(np.float32)

    coef_T = full_pol.coef_.T
    G_tr = np.hstack([oof_mom, oof_w_probs, oof_r_probs, oof_disp, Z_tr, sf_tr]).astype(np.float32)
    G_sel = np.hstack([ridge_preds["sel_mom"], ridge_preds["sel_w_probs"].reshape(-1, 15), ridge_preds["sel_r_probs"].reshape(-1, 15), _token_polarity_dispersion(X_sel, coef_T), Z_sel, sf_sel]).astype(np.float32)
    G_cal = np.hstack([ridge_preds["cal_mom"], ridge_preds["cal_w_probs"].reshape(-1, 15), ridge_preds["cal_r_probs"].reshape(-1, 15), _token_polarity_dispersion(X_cal, coef_T), Z_cal, sf_cal]).astype(np.float32)
    G_te = np.hstack([ridge_preds["te_mom"], ridge_preds["te_w_probs"].reshape(-1, 15), ridge_preds["te_r_probs"].reshape(-1, 15), _token_polarity_dispersion(X_te, coef_T), Z_te, sf_te]).astype(np.float32)

    return G_tr, G_sel, G_cal, G_te, ridge_preds


def _df_to_emobank_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = len(df)
    w_m = np.stack([df[f"w_{d}_pub_mean"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    w_s = np.stack([df[f"w_{d}_pub_std"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    r_m = np.stack([df[f"r_{d}_pub_mean"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    r_s = np.stack([df[f"r_{d}_pub_std"].to_numpy(dtype=np.float32) for d in DIMS], axis=-1)
    w_has = df["w_V_has_hist"].astype(bool).to_numpy()
    r_has = df["r_V_has_hist"].astype(bool).to_numpy()
    r_p = np.full((n, 3, 5), 0.2, dtype=np.float32)
    w_p = np.full((n, 3, 5), 0.2, dtype=np.float32)
    for j, d in enumerate(DIMS):
        rp = np.stack([df[f"r_{d}_p{k}"].to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1)
        wp = np.stack([df[f"w_{d}_p{k}"].fillna(0.2).to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1)
        r_p[r_has, j, :] = rp[r_has]
        w_p[w_has, j, :] = wp[w_has]
    return {"w_mean": w_m, "w_std": w_s, "r_mean": r_m, "r_std": r_s, "w_probs": w_p, "r_probs": r_p, "w_has_hist": w_has, "r_has_hist": r_has}


def run_emobank_classical(
    model_name: str = "C1_SparseRidge",
    protocol: str = "cv_fold_0",
    seed: int = 42,
    out_dir_override: str | None = None,
) -> Dict[str, object]:
    np.random.seed(seed)
    df = load_emobank_dataframe()
    df_tr, df_sel, df_cal, df_te = get_emobank_4way_splits(df, protocol=protocol, seed=seed)
    sf_tr, sf_sel, sf_cal, sf_te = fit_struct_scaler_for_splits(df_tr, df_sel, df_cal, df_te)

    X_tr, X_sel, X_cal, X_te = _build_sparse_design(
        df_tr["text"].astype(str).tolist(),
        df_sel["text"].astype(str).tolist(),
        df_cal["text"].astype(str).tolist(),
        df_te["text"].astype(str).tolist(),
    )
    G_tr, G_sel, G_cal, G_te, ridge_preds = _extract_oof_and_gbdt_features_emobank(
        X_tr, X_sel, X_cal, X_te, df_tr, sf_tr, sf_sel, sf_cal, sf_te
    )

    tgt_tr = _df_to_emobank_targets(df_tr)
    tgt_sel = _df_to_emobank_targets(df_sel)
    tgt_cal = _df_to_emobank_targets(df_cal)
    tgt_te = _df_to_emobank_targets(df_te)

    def _pack_ridge(tag: str) -> Dict[str, np.ndarray]:
        mom = ridge_preds[f"{tag}_mom"]
        wp = ridge_preds[f"{tag}_w_probs"]
        rp = ridge_preds[f"{tag}_r_probs"]
        _, ws = _moments_from_probs(wp)
        _, rs = _moments_from_probs(rp)
        # Moment regression mean clipped to [1,5], plus distribution std + ER0.8 score
        return {
            "w_probs": wp,
            "r_probs": rp,
            "w_mean_hat": np.clip(mom[:, 0:3], 1.0, 5.0),
            "w_std_hat": np.clip(mom[:, 3:6], 0.01, 2.0),
            "r_mean_hat": np.clip(mom[:, 6:9], 1.0, 5.0),
            "r_std_hat": np.clip(mom[:, 9:12], 0.01, 2.0),
            "w_u_amb": ws ** 2,
            "w_u_plur": esteban_ray_polarization(wp),
            "r_u_amb": rs ** 2,
            "r_u_plur": esteban_ray_polarization(rp),
        }

    if model_name == "C1_SparseRidge":
        p_sel = _pack_ridge("sel")
        p_cal = _pack_ridge("cal")
        p_te = _pack_ridge("te")
    elif model_name == "C2_HybridGBDT":
        p_sel = {"w_probs": np.zeros((len(df_sel), 3, 5)), "r_probs": np.zeros((len(df_sel), 3, 5))}
        p_cal = {"w_probs": np.zeros((len(df_cal), 3, 5)), "r_probs": np.zeros((len(df_cal), 3, 5))}
        p_te = {"w_probs": np.zeros((len(df_te), 3, 5)), "r_probs": np.zeros((len(df_te), 3, 5))}

        for p_key, tg_key in (("w_probs", "w"), ("r_probs", "r")):
            mask_tr = tgt_tr[f"{tg_key}_has_hist"]
            mask_sel = tgt_sel[f"{tg_key}_has_hist"]
            probs_tr_tgt = tgt_tr[f"{tg_key}_probs"][mask_tr]
            probs_sel_tgt = tgt_sel[f"{tg_key}_probs"][mask_sel]
            cdf_tr = np.cumsum(probs_tr_tgt, axis=-1)[..., :-1]
            cdf_sel = np.cumsum(probs_sel_tgt, axis=-1)[..., :-1]

            # Cumulative ordinal GBDT predictors P(Y <= k) for k=1..4 on each dimension
            cdf_hat = {"sel": np.zeros((len(df_sel), 3, 4)), "cal": np.zeros((len(df_cal), 3, 4)), "te": np.zeros((len(df_te), 3, 4))}
            for j in range(3):
                for b in range(4):
                    gbm = lgb.LGBMRegressor(
                        n_estimators=220,
                        learning_rate=0.05,
                        num_leaves=23,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        random_state=seed + j * 10 + b,
                        n_jobs=4,
                        verbose=-1,
                    )
                    gbm.fit(
                        G_tr[mask_tr],
                        cdf_tr[:, j, b],
                        eval_set=[(G_sel[mask_sel], cdf_sel[:, j, b])],
                        callbacks=[lgb.early_stopping(25, verbose=False)],
                    )
                    cdf_hat["sel"][:, j, b] = np.clip(gbm.predict(G_sel), 0.0, 1.0)
                    cdf_hat["cal"][:, j, b] = np.clip(gbm.predict(G_cal), 0.0, 1.0)
                    cdf_hat["te"][:, j, b] = np.clip(gbm.predict(G_te), 0.0, 1.0)

            for tag, store in (("sel", p_sel), ("cal", p_cal), ("te", p_te)):
                c_arr = np.maximum.accumulate(cdf_hat[tag], axis=-1)
                c_full = np.concatenate([np.zeros((*c_arr.shape[:2], 1)), c_arr, np.ones((*c_arr.shape[:2], 1))], axis=-1)
                probs = _project_simplex(np.diff(c_full, axis=-1))
                store[p_key] = probs
                _, std_d = _moments_from_probs(probs)
                store[f"{tg_key}_u_amb"] = std_d ** 2
                store[f"{tg_key}_u_plur"] = esteban_ray_polarization(probs)

        # Direct GBDT mean & std regressors over G_tr
        for p_key in ("w", "r"):
            m_sel, m_cal, m_te = [], [], []
            s_sel, s_cal, s_te = [], [], []
            for j in range(3):
                gm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=23, subsample=0.85, colsample_bytree=0.75, random_state=seed + 100 + j, n_jobs=4, verbose=-1)
                gm.fit(G_tr, tgt_tr[f"{p_key}_mean"][:, j], eval_set=[(G_sel, tgt_sel[f"{p_key}_mean"][:, j])], callbacks=[lgb.early_stopping(25, verbose=False)])
                m_sel.append(np.clip(gm.predict(G_sel), 1.0, 5.0))
                m_cal.append(np.clip(gm.predict(G_cal), 1.0, 5.0))
                m_te.append(np.clip(gm.predict(G_te), 1.0, 5.0))

                gs = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=23, subsample=0.85, colsample_bytree=0.75, random_state=seed + 200 + j, n_jobs=4, verbose=-1)
                gs.fit(G_tr, tgt_tr[f"{p_key}_std"][:, j], eval_set=[(G_sel, tgt_sel[f"{p_key}_std"][:, j])], callbacks=[lgb.early_stopping(25, verbose=False)])
                s_sel.append(np.clip(gs.predict(G_sel), 0.01, 2.0))
                s_cal.append(np.clip(gs.predict(G_cal), 0.01, 2.0))
                s_te.append(np.clip(gs.predict(G_te), 0.01, 2.0))

            for tag, st, ms, ss in (("sel", p_sel, m_sel, s_sel), ("cal", p_cal, m_cal, s_cal), ("te", p_te, m_te, s_te)):
                st[f"{p_key}_mean_hat"] = np.column_stack(ms)
                st[f"{p_key}_std_hat"] = np.column_stack(ss)
    else:
        raise ValueError(f"Unknown classical model_name: {model_name}")

    # Evaluate cleanly using the IDENTICAL functions and saving identical npz/summary.json formats!
    res_r = evaluate_perspective_predictions(
        tgt_te["r_probs"], p_te["r_probs"], tgt_te["r_mean"], p_te["r_mean_hat"], tgt_te["r_std"], p_te["r_std_hat"],
        tgt_te["r_has_hist"], u_plur=p_te["r_u_plur"], u_amb=p_te["r_u_amb"], prefix="r",
    )
    res_w = evaluate_perspective_predictions(
        tgt_te["w_probs"], p_te["w_probs"], tgt_te["w_mean"], p_te["w_mean_hat"], tgt_te["w_std"], p_te["w_std_hat"],
        tgt_te["w_has_hist"], u_plur=p_te["w_u_plur"], u_amb=p_te["w_u_amb"], prefix="w",
    )
    res_shift = evaluate_perspective_shift(tgt_te["w_mean"], tgt_te["r_mean"], p_te["w_mean_hat"], p_te["r_mean_hat"])
    test_metrics = {**res_r, **res_w, **res_shift}

    # Operational Alerting & Mondrian Conformal Risk Control using IDENTICAL evaluator as neural models
    def _to_eval_pack(p_dict, t_dict, df_d):
        return {
            "ids": df_d["id"].astype(str).tolist(),
            "categories": df_d["category"].astype(str).tolist(),
            "documents": df_d["document"].astype(str).tolist(),
            "r_probs": p_dict["r_probs"],
            "w_probs": p_dict["w_probs"],
            "r_disc_mean": p_dict["r_mean_hat"],
            "w_disc_mean": p_dict["w_mean_hat"],
            "r_disc_std": p_dict["r_std_hat"],
            "w_disc_std": p_dict["w_std_hat"],
            "r_u_amb": p_dict["r_u_amb"],
            "r_u_plur": p_dict["r_u_plur"],
            "w_u_amb": p_dict["w_u_amb"],
            "w_u_plur": p_dict["w_u_plur"],
            "tgt_r_probs": t_dict["r_probs"],
            "tgt_w_probs": t_dict["w_probs"],
            "tgt_r_mean": t_dict["r_mean"],
            "tgt_w_mean": t_dict["w_mean"],
            "tgt_r_std": t_dict["r_std"],
            "tgt_w_std": t_dict["w_std"],
            "w_has_hist": t_dict["w_has_hist"],
            "r_has_hist": t_dict["r_has_hist"],
        }

    conformal_res = evaluate_operational_and_conformal(
        _to_eval_pack(p_sel, tgt_sel, df_sel),
        _to_eval_pack(p_cal, tgt_cal, df_cal),
        _to_eval_pack(p_te, tgt_te, df_te),
        alpha=0.10,
    )

    out_dir = Path(out_dir_override) if out_dir_override else (OUT_DIR / "emobank" / f"{protocol}_{model_name}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, p_dict, tg_dict, df_d in (
        ("cal_predictions.npz", p_cal, tgt_cal, df_cal),
        ("test_predictions.npz", p_te, tgt_te, df_te),
    ):
        np.savez_compressed(
            out_dir / fname,
            ids=np.array(df_d["id"].astype(str).tolist(), dtype=object),
            categories=np.array(df_d["category"].astype(str).tolist(), dtype=object),
            documents=np.array(df_d["document"].astype(str).tolist(), dtype=object),
            r_probs=p_dict["r_probs"],
            w_probs=p_dict["w_probs"],
            r_disc_mean=p_dict["r_mean_hat"],
            w_disc_mean=p_dict["w_mean_hat"],
            r_disc_std=p_dict["r_std_hat"],
            w_disc_std=p_dict["w_std_hat"],
            r_u_amb=p_dict["r_u_amb"],
            r_u_plur=p_dict["r_u_plur"],
            w_u_amb=p_dict["w_u_amb"],
            w_u_plur=p_dict["w_u_plur"],
            tgt_r_probs=tg_dict["r_probs"],
            tgt_w_probs=tg_dict["w_probs"],
            tgt_r_mean=tg_dict["r_mean"],
            tgt_w_mean=tg_dict["w_mean"],
            tgt_r_std=tg_dict["r_std"],
            tgt_w_std=tg_dict["w_std"],
            w_has_hist=tg_dict["w_has_hist"],
            r_has_hist=tg_dict["r_has_hist"],
        )
    summary = {
        "protocol": protocol,
        "backbone": "classical_tfidf_struct",
        "model_type": model_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "conformal_and_alert_metrics": conformal_res,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Classical][{protocol} {model_name}] r_w1={test_metrics['r_mean_w1']:.4f} r_pearson={test_metrics['r_mean_pearson']:.4f} hard_auc={test_metrics.get('r_hard_shape_auroc_by_uplur', 0.0):.4f}")
    return summary


def run_epic_classical(model_name: str = "C2_HybridGBDT", cv_fold: int = -1, seed: int = 42) -> Dict[str, object]:
    df = pd.read_parquet(PROC_DIR / "epic_agg.parquet")
    df_tr, df_sel, df_cal, df_te = get_epic_4way_splits(df, cv_fold=cv_fold)
    pairs = lambda d: [f"{p} {r}".strip() for p, r in zip(d["parent_text"].fillna(""), d["text"].fillna(""))]
    tr_s = extract_struct_matrix(pairs(df_tr))
    sc = StandardScaler().fit(tr_s)
    sf_tr, sf_sel, sf_cal, sf_te = sc.transform(tr_s), sc.transform(extract_struct_matrix(pairs(df_sel))), sc.transform(extract_struct_matrix(pairs(df_cal))), sc.transform(extract_struct_matrix(pairs(df_te)))
    X_tr, X_sel, X_cal, X_te = _build_sparse_design(pairs(df_tr), pairs(df_sel), pairs(df_cal), pairs(df_te), word_max=15000, char_max=12000)

    nat_codes = ("uk", "us", "au", "ie", "in")
    if model_name == "C1_SparseRidge":
        X_trs = hstack([X_tr, sf_tr]).tocsr()
        m_ov = Ridge(alpha=3.0).fit(X_trs, df_tr["p_iro"].to_numpy())
        m_mask = df_tr["n_male"].to_numpy() > 0
        f_mask = df_tr["n_female"].to_numpy() > 0
        rm = Ridge(alpha=3.0).fit(X_trs[m_mask], df_tr["p_iro_male"].to_numpy()[m_mask])
        rf = Ridge(alpha=3.0).fit(X_trs[f_mask], df_tr["p_iro_female"].to_numpy()[f_mask])
        nat_models = [
            Ridge(alpha=3.0).fit(X_trs[df_tr[f"n_nat_{c}"].to_numpy() > 0], df_tr[f"p_nat_{c}"].to_numpy()[df_tr[f"n_nat_{c}"].to_numpy() > 0])
            if np.any(df_tr[f"n_nat_{c}"].to_numpy() > 0)
            else None
            for c in nat_codes
        ]

        def _pred_sparse(X_sp, sf_sp):
            Xs = hstack([X_sp, sf_sp]).tocsr()
            p_ov = np.clip(m_ov.predict(Xs), 0.0, 1.0)
            p_m = np.clip(rm.predict(Xs), 0.0, 1.0)
            p_f = np.clip(rf.predict(Xs), 0.0, 1.0)
            p_nat = np.column_stack([np.clip(nm.predict(Xs), 0.0, 1.0) if nm is not None else p_ov for nm in nat_models])
            return p_ov, p_m, p_f, p_nat

        split_preds = {
            "sel": _pred_sparse(X_sel, sf_sel),
            "cal": _pred_sparse(X_cal, sf_cal),
            "te": _pred_sparse(X_te, sf_te),
        }
    else:
        svd = TruncatedSVD(n_components=32, random_state=42)
        G_tr = np.hstack([svd.fit_transform(X_tr), sf_tr])
        G_sel = np.hstack([svd.transform(X_sel), sf_sel])
        G_cal = np.hstack([svd.transform(X_cal), sf_cal])
        G_te = np.hstack([svd.transform(X_te), sf_te])
        g_ov = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1)
        g_ov.fit(G_tr, df_tr["p_iro"].to_numpy(), eval_set=[(G_sel, df_sel["p_iro"].to_numpy())], callbacks=[lgb.early_stopping(20, verbose=False)])

        def _fit_sub_gbdt(col_n: str, col_p: str, sd: int):
            tr_m = df_tr[col_n].to_numpy() > 0
            sel_m = df_sel[col_n].to_numpy() > 0
            if np.sum(tr_m) < 10:
                return None
            mod = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=sd, verbose=-1)
            if np.sum(sel_m) >= 5:
                mod.fit(G_tr[tr_m], df_tr[col_p].to_numpy()[tr_m], eval_set=[(G_sel[sel_m], df_sel[col_p].to_numpy()[sel_m])], callbacks=[lgb.early_stopping(20, verbose=False)])
            else:
                mod.fit(G_tr[tr_m], df_tr[col_p].to_numpy()[tr_m])
            return mod

        gm = _fit_sub_gbdt("n_male", "p_iro_male", seed + 1)
        gf = _fit_sub_gbdt("n_female", "p_iro_female", seed + 2)
        nat_models = [_fit_sub_gbdt(f"n_nat_{c}", f"p_nat_{c}", seed + 10 + idx) for idx, c in enumerate(nat_codes)]

        def _pred_gbdt(G_sp):
            p_ov = np.clip(g_ov.predict(G_sp), 0.0, 1.0)
            p_m = np.clip(gm.predict(G_sp), 0.0, 1.0) if gm is not None else p_ov
            p_f = np.clip(gf.predict(G_sp), 0.0, 1.0) if gf is not None else p_ov
            p_nat = np.column_stack([np.clip(nm.predict(G_sp), 0.0, 1.0) if nm is not None else p_ov for nm in nat_models])
            return p_ov, p_m, p_f, p_nat

        split_preds = {"sel": _pred_gbdt(G_sel), "cal": _pred_gbdt(G_cal), "te": _pred_gbdt(G_te)}

    def _pack_epic(df_d: pd.DataFrame, tup: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        p_ov, p_m, p_f, p_nat = tup
        return {
            "p_ov_hat": p_ov.astype(np.float32),
            "p_m_hat": p_m.astype(np.float32),
            "p_f_hat": p_f.astype(np.float32),
            "p_nat_hat": p_nat.astype(np.float32),
            "u_plur": (np.abs(p_f - p_m) ** 2).astype(np.float32),
            "u_amb": (p_ov * (1.0 - p_ov)).astype(np.float32),
            "k_iro": df_d["k_iro"].to_numpy(dtype=np.float32),
            "n_annot": df_d["n_annot"].to_numpy(dtype=np.float32),
            "k_male": df_d["k_male"].to_numpy(dtype=np.float32),
            "n_male": df_d["n_male"].to_numpy(dtype=np.float32),
            "k_female": df_d["k_female"].to_numpy(dtype=np.float32),
            "n_female": df_d["n_female"].to_numpy(dtype=np.float32),
            "k_nat": np.stack([df_d[f"k_nat_{c}"].to_numpy(dtype=np.float32) for c in nat_codes], axis=-1),
            "n_nat": np.stack([df_d[f"n_nat_{c}"].to_numpy(dtype=np.float32) for c in nat_codes], axis=-1),
            "is_pol": df_d["is_polarized"].to_numpy(dtype=bool),
            "is_una": df_d["is_unanimous"].to_numpy(dtype=bool),
            "group_keys": df_d["group_key"].fillna("").astype(str).to_numpy(dtype=object),
        }

    sel_pred = _pack_epic(df_sel, split_preds["sel"])
    cal_pred = _pack_epic(df_cal, split_preds["cal"])
    te_pred = _pack_epic(df_te, split_preds["te"])

    res = evaluate_epic(te_pred, prefix="test")

    def _risk_X(p: Dict[str, np.ndarray]) -> np.ndarray:
        p_h = p["p_ov_hat"]
        return np.column_stack([
            np.abs(p["p_f_hat"] - p["p_m_hat"]),
            np.sqrt(np.clip(p["u_plur"], 0.0, None)),
            np.sqrt(np.clip(p["u_amb"], 0.0, None)),
            np.sqrt(np.clip(p_h * (1.0 - p_h), 0.0, None)),
            np.nanstd(p["p_nat_hat"], axis=-1),
        ]).astype(float)

    def _alert_y(p: Dict[str, np.ndarray]) -> np.ndarray:
        ge2 = (p["n_male"] >= 2) & (p["n_female"] >= 2)
        s_div = np.zeros(len(p["n_male"]), dtype=bool)
        if np.any(ge2):
            s_div[ge2] = np.abs(p["k_female"][ge2] / p["n_female"][ge2] - p["k_male"][ge2] / p["n_male"][ge2]) >= 0.35
        return (p["is_pol"] | s_div).astype(int)

    te_y = _alert_y(te_pred)
    crc_pol_10 = fit_and_calibrate_alert_crc(
        _risk_X(sel_pred), _alert_y(sel_pred),
        _risk_X(cal_pred), _alert_y(cal_pred),
        _risk_X(te_pred), te_y,
        alpha=0.10, use_supervised_combiner=True,
    )
    alert_raw = evaluate_binary_alert_metrics(te_y, np.abs(te_pred["p_f_hat"] - te_pred["p_m_hat"]) + np.sqrt(np.clip(te_pred["u_plur"], 0.0, None)))

    fold_tag = "split0" if cv_fold < 0 else f"cv{cv_fold}"
    out_dir = OUT_DIR / "epic" / f"{fold_tag}_classical_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "cal_predictions.npz", **cal_pred)
    np.savez_compressed(out_dir / "test_predictions.npz", **te_pred)
    summary = {
        "task": "epic_classical",
        "model_type": model_name,
        "cv_fold": cv_fold,
        "seed": seed,
        "test": res,
        "alert_raw": alert_raw,
        "crc_pol_alpha10": crc_pol_10,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[EPIC Classical {model_name}] ov_r={res['test_overall_pearson']:.4f} sex_div_auc={res.get('test_between_sex_div_auroc', 0.0):.4f}")
    return summary


def run_brighter_classical(lang: str = "chn", model_name: str = "C1_SparseRidge", cv_fold: int = -1, seed: int = 42) -> Dict[str, object]:
    df_tr, df_sel, df_cal, df_te, active = _get_brighter_4way(lang, cv_fold=cv_fold)
    sc = StandardScaler().fit(extract_struct_matrix(df_tr["text"].fillna("").astype(str).tolist()))
    sf_tr = sc.transform(extract_struct_matrix(df_tr["text"].fillna("").astype(str).tolist()))
    sf_sel = sc.transform(extract_struct_matrix(df_sel["text"].fillna("").astype(str).tolist()))
    sf_cal = sc.transform(extract_struct_matrix(df_cal["text"].fillna("").astype(str).tolist()))
    sf_te = sc.transform(extract_struct_matrix(df_te["text"].fillna("").astype(str).tolist()))

    X_tr, X_sel, X_cal, X_te = _build_sparse_design(
        df_tr["text"].fillna("").astype(str).tolist(),
        df_sel["text"].fillna("").astype(str).tolist(),
        df_cal["text"].fillna("").astype(str).tolist(),
        df_te["text"].fillna("").astype(str).tolist(),
        word_max=15000,
        char_max=15000,
    )
    X_trs = hstack([X_tr, sf_tr]).tocsr()
    X_sels = hstack([X_sel, sf_sel]).tocsr()
    X_cals = hstack([X_cal, sf_cal]).tocsr()
    X_tes = hstack([X_te, sf_te]).tocsr()

    def _fit_probs(df_d: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vm = df_d[active].notna().to_numpy(dtype=bool)
        yc = np.clip(df_d[active].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 3.0)
        return yc, np.round(yc).astype(np.int64), vm

    yc_tr, yi_tr, vm_tr = _fit_probs(df_tr)
    yc_sel, yi_sel, vm_sel = _fit_probs(df_sel)
    yc_cal, yi_cal, vm_cal = _fit_probs(df_cal)
    yc_te, yi_te, vm_te = _fit_probs(df_te)

    probs_sel = np.zeros((len(df_sel), len(active), 4), dtype=np.float32)
    probs_cal = np.zeros((len(df_cal), len(active), 4), dtype=np.float32)
    probs_te = np.zeros((len(df_te), len(active), 4), dtype=np.float32)

    svd = TruncatedSVD(n_components=32, random_state=42) if model_name == "C2_HybridGBDT" else None
    G_tr = np.hstack([svd.fit_transform(X_tr), sf_tr]) if svd else None
    G_sel = np.hstack([svd.transform(X_sel), sf_sel]) if svd else None
    G_cal = np.hstack([svd.transform(X_cal), sf_cal]) if svd else None
    G_te = np.hstack([svd.transform(X_te), sf_te]) if svd else None

    for j in range(len(active)):
        m = vm_tr[:, j]
        y_oh = np.eye(4, dtype=np.float32)[yi_tr[m, j]]
        if model_name == "C1_SparseRidge":
            reg = Ridge(alpha=3.0).fit(X_trs[m], y_oh)
            probs_sel[:, j, :] = _project_simplex(reg.predict(X_sels))
            probs_cal[:, j, :] = _project_simplex(reg.predict(X_cals))
            probs_te[:, j, :] = _project_simplex(reg.predict(X_tes))
        else:
            for k in range(4):
                gb = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=seed + j * 4 + k, verbose=-1)
                ms = vm_sel[:, j]
                if np.sum(ms) >= 5:
                    y_sel_oh = np.eye(4, dtype=np.float32)[yi_sel[ms, j]]
                    gb.fit(G_tr[m], y_oh[:, k], eval_set=[(G_sel[ms], y_sel_oh[:, k])], callbacks=[lgb.early_stopping(20, verbose=False)])
                else:
                    gb.fit(G_tr[m], y_oh[:, k])
                probs_sel[:, j, k] = np.clip(gb.predict(G_sel), 0.0, 1.0)
                probs_cal[:, j, k] = np.clip(gb.predict(G_cal), 0.0, 1.0)
                probs_te[:, j, k] = np.clip(gb.predict(G_te), 0.0, 1.0)
            probs_sel[:, j, :] = _project_simplex(probs_sel[:, j, :])
            probs_cal[:, j, :] = _project_simplex(probs_cal[:, j, :])
            probs_te[:, j, :] = _project_simplex(probs_te[:, j, :])

    tau = _fit_temperature_on_sel(probs_sel, yi_sel, vm_sel)
    mean_sel, std_sel = _moments_from_probs(probs_sel, v=BR_SCORES)
    mean_cal, std_cal = _moments_from_probs(probs_cal, v=BR_SCORES)
    mean_te, std_te = _moments_from_probs(probs_te, v=BR_SCORES)

    sel_pack = {"probs": probs_sel, "disc_mean": mean_sel, "disc_std": std_sel, "y_cont": yc_sel, "y_int": yi_sel, "valid_mask": vm_sel}
    cal_pack = {"probs": probs_cal, "disc_mean": mean_cal, "disc_std": std_cal, "y_cont": yc_cal, "y_int": yi_cal, "valid_mask": vm_cal}
    te_pack = {"probs": probs_te, "disc_mean": mean_te, "disc_std": std_te, "y_cont": yc_te, "y_int": yi_te, "valid_mask": vm_te}
    res_cal = evaluate_brighter(cal_pack, active, tau=tau, prefix="cal")
    res_te = evaluate_brighter(te_pack, active, tau=tau, prefix="test")

    def _ext_pair(p_d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        ps = _temp_scale_probs(p_d["probs"], tau)
        sc_ext = np.max(np.where(p_d["valid_mask"], ps[..., 2] + ps[..., 3], 0.0), axis=-1)
        y_vals = np.where(p_d["valid_mask"], p_d["y_int"], 0)
        return sc_ext, (np.max(y_vals, axis=-1) >= 2).astype(int)

    sx, sy = _ext_pair(sel_pack)
    cx, cy = _ext_pair(cal_pack)
    tx, ty = _ext_pair(te_pack)
    crc_ext_10 = fit_and_calibrate_alert_crc(sx, sy, cx, cy, tx, ty, alpha=0.10, use_supervised_combiner=False)

    fold_tag = "official" if cv_fold < 0 else f"cv{cv_fold}"
    out_dir = OUT_DIR / "brighter" / f"{lang}_{fold_tag}_classical_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "cal_predictions.npz", **cal_pack)
    np.savez_compressed(out_dir / "test_predictions.npz", **te_pack)
    summary = {
        "task": "brighter_classical",
        "lang": lang,
        "model_type": model_name,
        "calibrated_temp": round(tau, 4),
        "cal": res_cal,
        "test": res_te,
        "crc_extremity_alpha10": crc_ext_10,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[BRIGHTER {lang} Classical {model_name}] r={res_te['test_macro_pearson']:.4f} nll={res_te['test_macro_nll']:.4f} qwk={res_te['test_macro_qwk']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="emobank", choices=["emobank", "epic", "brighter"])
    parser.add_argument("--model-name", type=str, default="C1_SparseRidge", choices=["C1_SparseRidge", "C2_HybridGBDT"])
    parser.add_argument("--protocol", type=str, default="cv_fold_0")
    parser.add_argument("--lang", type=str, default="chn")
    parser.add_argument("--cv-fold", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task == "emobank":
        run_emobank_classical(model_name=args.model_name, protocol=args.protocol, seed=args.seed)
    elif args.task == "epic":
        run_epic_classical(model_name=args.model_name, cv_fold=args.cv_fold, seed=args.seed)
    else:
        run_brighter_classical(lang=args.lang, model_name=args.model_name, cv_fold=args.cv_fold, seed=args.seed)


if __name__ == "__main__":
    main()
