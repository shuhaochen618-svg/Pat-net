"""Aggregate EmoBank 5-fold OOF, M0-M5 ablations, DeBERTa, LOGO, Ensembles, EPIC 5-fold, BRIGHTER 4-lang, Hardware, & Gate 1-4 checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.epic_exp import evaluate_epic
from src.metrics import (
    cluster_bootstrap_paired_test,
    discrete_wasserstein1,
    evaluate_binary_alert_metrics,
    evaluate_perspective_predictions,
    evaluate_perspective_shift,
    extract_clean_polarization_masks,
    holm_bonferroni,
)
from src.paths import OUT_DIR


def _round_metrics(d: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, (int, bool)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, 5) if np.isfinite(v) else None
        else:
            out[k] = v
    return out


def _summarize_emobank_folds(dirs: List[Path]) -> Dict[str, object] | None:
    folds_te: List[Dict[str, np.ndarray]] = []
    sums: List[Dict[str, object]] = []
    for d in dirs:
        if not (d / "test_predictions.npz").exists():
            return None
        folds_te.append(dict(np.load(d / "test_predictions.npz", allow_pickle=True)))
        if (d / "summary.json").exists():
            sums.append(json.loads((d / "summary.json").read_text(encoding="utf-8")))

    oof = {k: np.concatenate([x[k] for x in folds_te], axis=0) for k in folds_te[0].keys()}
    ev_r = evaluate_perspective_predictions(
        oof["tgt_r_probs"], oof["r_probs"], oof["tgt_r_mean"], oof["r_disc_mean"],
        oof["tgt_r_std"], oof["r_disc_std"], oof["r_has_hist"],
        u_plur=oof["r_u_plur"], u_amb=oof["r_u_amb"], prefix="r",
    )
    ev_w = evaluate_perspective_predictions(
        oof["tgt_w_probs"], oof["w_probs"], oof["tgt_w_mean"], oof["w_disc_mean"],
        oof["tgt_w_std"], oof["w_disc_std"], oof["w_has_hist"],
        u_plur=oof["w_u_plur"], u_amb=oof["w_u_amb"], prefix="w",
    )
    ev_sh = evaluate_perspective_shift(oof["tgt_w_mean"], oof["tgt_r_mean"], oof["w_disc_mean"], oof["r_disc_mean"])

    conf_agg: Dict[str, float] = {}
    if sums and "conformal_and_alert_metrics" in sums[0]:
        for ck in sums[0]["conformal_and_alert_metrics"].keys():
            vals = [
                float(s["conformal_and_alert_metrics"][ck])
                for s in sums
                if ck in s.get("conformal_and_alert_metrics", {}) and np.isfinite(s["conformal_and_alert_metrics"][ck])
            ]
            if vals:
                conf_agg[ck] = round(float(np.mean(vals)), 5)
    elif sums and "conformal" in sums[0]:
        buf: Dict[str, List[float]] = {}
        for s in sums:
            c = s.get("conformal", {})
            for grp, gd in c.items():
                if isinstance(gd, dict):
                    for k2, v2 in gd.items():
                        if isinstance(v2, (int, float)) and np.isfinite(v2):
                            buf.setdefault(f"{grp}_{k2}", []).append(float(v2))
        conf_agg = {k: round(float(np.mean(v)), 5) for k, v in buf.items() if v}

    hm = oof["r_has_hist"].astype(bool)
    any_vod = np.zeros(hm.sum(), dtype=bool)
    for j in range(3):
        m_j = extract_clean_polarization_masks(oof["tgt_r_probs"][hm, j], oof["tgt_r_mean"][hm, j], oof["tgt_r_std"][hm, j])
        any_vod |= m_j["neut"] & m_j["valley_or_dip"]
    m_v = extract_clean_polarization_masks(oof["tgt_r_probs"][hm, 0], oof["tgt_r_mean"][hm, 0], oof["tgt_r_std"][hm, 0])
    distress_v = m_v["distress_neut"]

    al_vod = evaluate_binary_alert_metrics(any_vod.astype(int), np.max(oof["r_u_plur"][hm], axis=-1))
    al_dis = evaluate_binary_alert_metrics(distress_v.astype(int), oof["r_u_plur"][hm, 0])
    return {
        "oof": oof,
        "metrics": {**ev_r, **ev_w, **ev_sh},
        "conformal_mean": conf_agg,
        "oof_alert_any_vod_uplur": al_vod,
        "oof_alert_distress_v_uplur": al_dis,
    }


def _load_oof_emobank(exact_suffix: str, eb_dir: Path, n_folds: int = 5) -> Dict[str, object] | None:
    dirs = [eb_dir / f"cv_fold_{f}_{exact_suffix}" for f in range(n_folds)]
    return _summarize_emobank_folds(dirs)


def _format_emobank_row(d: Dict[str, object]) -> Dict[str, object]:
    return {
        **_round_metrics(d["metrics"]),
        "conformal_mean": d["conformal_mean"],
        "oof_alert_any_vod_auroc": round(d["oof_alert_any_vod_uplur"]["auroc"], 4),
        "oof_alert_distress_v_auroc": round(d["oof_alert_distress_v_uplur"]["auroc"], 4),
    }


def _load_deep_ensemble_fold0(eb_dir: Path, mt: str, ab: str, seeds=(42, 43, 44, 45, 46)) -> Dict[str, object] | None:
    preds = []
    for s in seeds:
        d = eb_dir / f"cv_fold_0_roberta-base_{mt}_{ab}_seed{s}" / "test_predictions.npz"
        if not d.exists():
            return None
        preds.append(dict(np.load(d, allow_pickle=True)))
    p0 = preds[0]
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, 1, -1)
    r_probs = np.mean([p["r_probs"] for p in preds], axis=0)
    w_probs = np.mean([p["w_probs"] for p in preds], axis=0)
    r_mean = (r_probs * v).sum(axis=-1)
    w_mean = (w_probs * v).sum(axis=-1)
    r_std = np.sqrt(np.clip((r_probs * (v - r_mean[..., None]) ** 2).sum(axis=-1), 0.0, None))
    w_std = np.sqrt(np.clip((w_probs * (v - w_mean[..., None]) ** 2).sum(axis=-1), 0.0, None))
    # Epistemic disagreement across ensemble seeds + mean predicted aleatoric/plur
    r_epist = np.var([p["r_disc_mean"] for p in preds], axis=0)
    w_epist = np.var([p["w_disc_mean"] for p in preds], axis=0)
    r_uplur = np.mean([p["r_u_plur"] for p in preds], axis=0) + (0.0 if mt == "patnet" else r_epist)
    w_uplur = np.mean([p["w_u_plur"] for p in preds], axis=0) + (0.0 if mt == "patnet" else w_epist)
    ev_r = evaluate_perspective_predictions(
        p0["tgt_r_probs"], r_probs, p0["tgt_r_mean"], r_mean, p0["tgt_r_std"], r_std,
        p0["r_has_hist"], u_plur=r_uplur, u_amb=np.mean([p["r_u_amb"] for p in preds], axis=0), prefix="r",
    )
    ev_w = evaluate_perspective_predictions(
        p0["tgt_w_probs"], w_probs, p0["tgt_w_mean"], w_mean, p0["tgt_w_std"], w_std,
        p0["w_has_hist"], u_plur=w_uplur, u_amb=np.mean([p["w_u_amb"] for p in preds], axis=0), prefix="w",
    )
    ev_sh = evaluate_perspective_shift(p0["tgt_w_mean"], p0["tgt_r_mean"], w_mean, r_mean)
    return _round_metrics({**ev_r, **ev_w, **ev_sh})


def _aggregate_epic(ep_dir: Path) -> Dict[str, object]:
    epic_configs = [
        ("C1_SparseRidge", "classical_C1_SparseRidge_seed42", 5),
        ("C2_HybridGBDT", "classical_C2_HybridGBDT_seed42", 5),
        ("roberta_scalar_bce", "roberta-base_scalar_bce_M0_baseline_seed42", 5),
        ("roberta_beta_binom", "roberta-base_beta_binom_M0_baseline_seed42", 5),
        ("roberta_patnet_no_transport", "roberta-base_patnet_no_transport_M4_disentangle_seed42", 5),
        ("roberta_patnet_M4", "roberta-base_patnet_M4_disentangle_seed42", 5),
        ("deberta_patnet_no_transport", "microsoft__deberta-v3-base_patnet_no_transport_M4_disentangle_seed42", 3),
        ("deberta_patnet_M4", "microsoft__deberta-v3-base_patnet_M4_disentangle_seed42", 3),
    ]
    oof_store: Dict[str, Dict[str, np.ndarray]] = {}
    summary_store: Dict[str, object] = {}
    for label, suffix, n_f in epic_configs:
        preds, sums = [], []
        for f in range(n_f):
            d = ep_dir / f"cv{f}_{suffix}"
            if (d / "test_predictions.npz").exists():
                preds.append(dict(np.load(d / "test_predictions.npz", allow_pickle=True)))
            if (d / "summary.json").exists():
                sums.append(json.loads((d / "summary.json").read_text(encoding="utf-8")))
        if len(preds) == n_f:
            oof = {k: np.concatenate([x[k] for x in preds], axis=0) for k in preds[0].keys()}
            oof_store[label] = oof
            ev = evaluate_epic(oof, prefix="oof")
            crc_means = {}
            if sums and "crc_pol_alpha10" in sums[0]:
                for ck in sums[0]["crc_pol_alpha10"].keys():
                    vals = [float(s["crc_pol_alpha10"][ck]) for s in sums if ck in s.get("crc_pol_alpha10", {}) and np.isfinite(s["crc_pol_alpha10"][ck])]
                    if vals:
                        crc_means[ck] = round(float(np.mean(vals)), 4)
            summary_store[label] = {**_round_metrics(ev), "crc_pol_alpha10_mean": crc_means, "n_folds": n_f}
        elif sums:
            # Fallback to averaging test dicts if classical runs lack npz
            t_buf: Dict[str, List[float]] = {}
            for s in sums:
                for k, v in s.get("test", {}).items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        t_buf.setdefault(k, []).append(float(v))
            summary_store[label] = {k: round(float(np.mean(v)), 5) for k, v in t_buf.items() if v}

    epic_boot: Dict[str, object] = {}
    if "roberta_patnet_M4" in oof_store and "roberta_patnet_no_transport" in oof_store:
        p_o = oof_store["roberta_patnet_M4"]
        nt_o = oof_store["roberta_patnet_no_transport"]
        ge2 = (p_o["n_male"] >= 2) & (p_o["n_female"] >= 2)
        if "group_keys" in p_o and np.any(ge2):
            true_sh = (p_o["k_female"][ge2] / p_o["n_female"][ge2]) - (p_o["k_male"][ge2] / p_o["n_male"][ge2])
            sh_err_p = np.abs((p_o["p_f_hat"][ge2] - p_o["p_m_hat"][ge2]) - true_sh)
            sh_err_nt = np.abs((nt_o["p_f_hat"][ge2] - nt_o["p_m_hat"][ge2]) - true_sh)
            epic_boot["sex_shift_mae_vs_no_transport"] = cluster_bootstrap_paired_test(
                sh_err_p, sh_err_nt, p_o["group_keys"][ge2], n_boot=2000, seed=42
            )
    return {"models": summary_store, "cluster_bootstrap_2000": epic_boot}


def _aggregate_brighter(br_dir: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    if not br_dir.exists():
        return rows
    for p in sorted(br_dir.glob("*/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows[p.parent.name] = {
            "lang": d.get("lang"),
            "backbone": d.get("backbone", "classical"),
            "model_type": d.get("model_type"),
            "ablation_mode": d.get("ablation_mode", "classical"),
            **_round_metrics(d.get("test", {})),
            "crc_extremity_alpha10": d.get("crc_extremity_alpha10"),
        }
    return rows


def evaluate_all_gates(
    emobank_root: Path | None = None,
    epic_root: Path | None = None,
    brighter_root: Path | None = None,
) -> Dict[str, object]:
    eb_dir = emobank_root or (OUT_DIR / "emobank")
    ep_dir = epic_root or (OUT_DIR / "epic")
    br_dir = brighter_root or (OUT_DIR / "brighter")

    model_specs = [
        ("C1_SparseRidge", "C1_SparseRidge_seed42"),
        ("C2_HybridGBDT", "C2_HybridGBDT_seed42"),
        ("scalar_mse", "roberta-base_scalar_mse_M0_baseline_seed42"),
        ("gaussian_nll", "roberta-base_gaussian_nll_M0_baseline_seed42"),
        ("gaussian_nll_mcT10", "roberta-base_gaussian_nll_M0_baseline_seed42_mcT10"),
        ("nig_evidential", "roberta-base_nig_evidential_M0_baseline_seed42"),
        ("sord", "roberta-base_sord_M0_baseline_seed42"),
        ("dldl_v2", "roberta-base_dldl_v2_M0_baseline_seed42"),
        ("dc_ldl", "roberta-base_dc_ldl_M0_baseline_seed42"),
        ("dc_ldl_mcT10", "roberta-base_dc_ldl_M0_baseline_seed42_mcT10"),
        ("uw_ldl", "roberta-base_uw_ldl_M0_baseline_seed42"),
        ("beta_nll", "roberta-base_beta_nll_M0_baseline_seed42"),
        ("patnet_unimodal", "roberta-base_patnet_unimodal_M4_disentangle_seed42"),
        ("patnet_no_transport", "roberta-base_patnet_no_transport_M4_disentangle_seed42"),
        ("patnet_M4", "roberta-base_patnet_M4_disentangle_seed42"),
        ("patnet_M5", "roberta-base_patnet_M5_full_struct_seed42"),
    ]
    results: Dict[str, Dict[str, object]] = {}
    for label, suffix in model_specs:
        loaded = _load_oof_emobank(suffix, eb_dir)
        if loaded is not None:
            results[label] = loaded

    summary_rows = {label: _format_emobank_row(d) for label, d in results.items()}

    # Stage 2: M0..M5 ablation ladder, DeBERTa-v3 5-fold, LOGO 7-genre, & Deep Ensemble E=5
    ablation_ladder = {}
    for ab in ("M0_baseline", "M1_multilayer_conv", "M2_bipolar_pool", "M3_tension_route", "M4_disentangle", "M5_full_struct"):
        ld = _load_oof_emobank(f"roberta-base_patnet_{ab}_seed42", eb_dir)
        if ld is not None:
            ablation_ladder[ab] = _format_emobank_row(ld)

    deberta_rows = {}
    for label, mt, ab in (
        ("scalar_mse", "scalar_mse", "M0_baseline"),
        ("sord", "sord", "M0_baseline"),
        ("dc_ldl", "dc_ldl", "M0_baseline"),
        ("beta_nll", "beta_nll", "M0_baseline"),
        ("patnet_no_transport", "patnet_no_transport", "M4_disentangle"),
        ("patnet_M4", "patnet", "M4_disentangle"),
    ):
        ld = _load_oof_emobank(f"microsoft_deberta-v3-base_{mt}_{ab}_seed42", eb_dir)
        if ld is not None:
            deberta_rows[label] = _format_emobank_row(ld)

    genres = ("fiction", "letters", "blog", "newspaper", "SemEval", "essays", "travel-guides")
    logo_rows: Dict[str, object] = {}
    for mt, ab in (("dc_ldl", "M0_baseline"), ("patnet_no_transport", "M4_disentangle"), ("patnet", "M4_disentangle")):
        dirs = [eb_dir / f"logo_{g}_roberta-base_{mt}_{ab}_seed42" for g in genres]
        if all((d / "summary.json").exists() for d in dirs):
            per_g = {}
            for g, d in zip(genres, dirs):
                per_g[g] = _round_metrics(json.loads((d / "summary.json").read_text(encoding="utf-8")).get("test_metrics", {}))
            oof_g = _summarize_emobank_folds(dirs)
            logo_rows[mt] = {
                "per_genre": per_g,
                "oof": _format_emobank_row(oof_g) if oof_g is not None else {},
            }

    ensembles = {}
    for mt, ab in (("dc_ldl", "M0_baseline"), ("patnet", "M4_disentangle")):
        ens = _load_deep_ensemble_fold0(eb_dir, mt, ab)
        if ens is not None:
            ensembles[f"{mt}_fold0_E5"] = ens

    # Cluster-paired bootstrap comparing primary patnet_M4 vs all baselines
    boot_report: Dict[str, object] = {}
    primary_key = "patnet_M4" if "patnet_M4" in results else ("patnet_M5" if "patnet_M5" in results else None)
    if primary_key is not None:
        p_oof = results[primary_key]["oof"]
        hm = p_oof["r_has_hist"].astype(bool)
        docs = p_oof["documents"][hm]
        ids_p = p_oof["ids"]
        w1_p = discrete_wasserstein1(p_oof["tgt_r_probs"][hm], p_oof["r_probs"][hm]).mean(axis=-1)
        sh_p = np.abs(p_oof["r_disc_mean"] - p_oof["w_disc_mean"] - (p_oof["tgt_r_mean"] - p_oof["tgt_w_mean"])).mean(axis=-1)

        raw_pvals: List[float] = []
        raw_names: List[str] = []
        comp_stats: Dict[str, Dict[str, object]] = {}
        cmp_list = (
            "C1_SparseRidge", "C2_HybridGBDT", "scalar_mse", "gaussian_nll", "gaussian_nll_mcT10",
            "nig_evidential", "sord", "dldl_v2", "dc_ldl", "dc_ldl_mcT10", "uw_ldl", "beta_nll",
            "patnet_unimodal", "patnet_no_transport",
        )
        for cmp_name in cmp_list:
            if cmp_name not in results:
                continue
            b_oof = results[cmp_name]["oof"]
            if not np.array_equal(ids_p, b_oof["ids"]):
                order_map = {k: idx for idx, k in enumerate(b_oof["ids"])}
                perm = np.array([order_map[k] for k in ids_p], dtype=int)
                b_r_probs, b_rm, b_wm = b_oof["r_probs"][perm], b_oof["r_disc_mean"][perm], b_oof["w_disc_mean"][perm]
            else:
                b_r_probs, b_rm, b_wm = b_oof["r_probs"], b_oof["r_disc_mean"], b_oof["w_disc_mean"]

            w1_b = discrete_wasserstein1(p_oof["tgt_r_probs"][hm], b_r_probs[hm]).mean(axis=-1)
            bt_w1 = cluster_bootstrap_paired_test(w1_p, w1_b, docs, n_boot=2000, seed=42)
            raw_names.append(f"w1_vs_{cmp_name}")
            raw_pvals.append(bt_w1["p_two_sided"])
            comp_stats[f"w1_vs_{cmp_name}"] = bt_w1

            sh_b = np.abs(b_rm - b_wm - (p_oof["tgt_r_mean"] - p_oof["tgt_w_mean"])).mean(axis=-1)
            bt_sh = cluster_bootstrap_paired_test(sh_p, sh_b, p_oof["documents"], n_boot=2000, seed=42)
            raw_names.append(f"shift_mae_vs_{cmp_name}")
            raw_pvals.append(bt_sh["p_two_sided"])
            comp_stats[f"shift_mae_vs_{cmp_name}"] = bt_sh

        if raw_pvals:
            adj = holm_bonferroni(raw_pvals)
            for idx, nm in enumerate(raw_names):
                comp_stats[nm]["p_holm"] = float(adj[idx])
        boot_report = comp_stats

    # Formal Gate 1-4 automated evaluation
    gates: Dict[str, object] = {}
    dist_baselines = [b for b in ("sord", "dldl_v2", "dc_ldl", "uw_ldl", "beta_nll", "patnet_unimodal", "dc_ldl_mcT10") if b in results]
    all_baselines = [b for b in results.keys() if not b.startswith("patnet_M")]
    if primary_key is not None and dist_baselines:
        pm = results[primary_key]["metrics"]
        best_dist_w1 = min(results[b]["metrics"].get("r_mean_w1", 99.0) for b in dist_baselines)
        best_all_pear = max(results[b]["metrics"].get("r_mean_pearson", 0.0) for b in all_baselines)
        gates["gate1_distributional_win_no_scalar_drop"] = {
            "primary_r_w1": round(pm.get("r_mean_w1", 99.0), 4),
            "best_dist_baseline_r_w1": round(best_dist_w1, 4),
            "primary_r_pearson": round(pm.get("r_mean_pearson", 0.0), 4),
            "best_all_baseline_r_pearson": round(best_all_pear, 4),
            "passed": bool((pm.get("r_mean_w1", 99.0) < best_dist_w1) and (pm.get("r_mean_pearson", 0.0) >= best_all_pear - 0.005)),
        }
        hs_plur = pm.get("r_hard_shape_auroc_by_uplur", 0.0)
        hs_uamb = pm.get("r_hard_shape_auroc_by_uamb", 1.0)
        hs_std = pm.get("r_hard_shape_auroc_by_std", 1.0)
        hs_er08 = pm.get("r_hard_shape_auroc_by_er08", 0.0)
        single_pass_dist = [b for b in dist_baselines if b != "dc_ldl_mcT10"]
        er08_bases = {b: round(results[b]["metrics"].get("r_hard_shape_auroc_by_er08", 0.0), 4) for b in dist_baselines}
        best_single_er08 = max((results[b]["metrics"].get("r_hard_shape_auroc_by_er08", 0.0) for b in single_pass_dist), default=0.0)
        best_base_er08 = max((results[b]["metrics"].get("r_hard_shape_auroc_by_er08", 0.0) for b in dist_baselines), default=0.0)
        dc_var_confounded = round(results["dc_ldl"]["metrics"].get("r_hard_shape_auroc_by_uplur", 0.0), 4) if "dc_ldl" in results else None
        raw_mc_uplur = results["dc_ldl_mcT10"]["metrics"].get("r_hard_shape_auroc_by_uplur", 0.0) if "dc_ldl_mcT10" in results else None
        mc_epistemic_uplur = round(raw_mc_uplur, 4) if raw_mc_uplur is not None else None
        disentangle_ok = bool(hs_plur > hs_uamb)
        if raw_mc_uplur is not None:
            disentangle_ok = bool(disentangle_ok and (hs_plur > raw_mc_uplur))
        er08_ok = bool((hs_er08 > best_single_er08) and (hs_er08 >= best_base_er08 - 0.002))
        gates["gate2_non_circular_shape_disentanglement"] = {
            "hard_shape_uplur_auroc": round(hs_plur, 4),
            "hard_shape_uamb_auroc": round(hs_uamb, 4),
            "hard_shape_std_auroc": round(hs_std, 4),
            "uplur_minus_uamb": round(hs_plur - hs_uamb, 4),
            "uplur_minus_std": round(hs_plur - hs_std, 4),
            "hard_shape_er08_auroc": round(hs_er08, 4),
            "best_single_pass_er08_auroc": round(best_single_er08, 4),
            "best_all_baseline_er08_auroc": round(best_base_er08, 4),
            "baseline_er08_aurocs": er08_bases,
            "dc_ldl_var_confounded_uplur": dc_var_confounded,
            "dc_ldl_mcT10_epistemic_uplur": mc_epistemic_uplur,
            "passed": bool(disentangle_ok and er08_ok),
        }

    if primary_key is not None and "patnet_no_transport" in results:
        pm = results[primary_key]["metrics"]
        nt = results["patnet_no_transport"]["metrics"]
        p_sh_holm = boot_report.get("shift_mae_vs_patnet_no_transport", {}).get("p_holm", 1.0)
        logo_ok = True
        logo_info: Dict[str, object] = {}
        if "patnet" in logo_rows and "patnet_no_transport" in logo_rows:
            lp = logo_rows["patnet"]["oof"]
            lnt = logo_rows["patnet_no_transport"]["oof"]
            logo_ok = bool(lp.get("shift_mean_mae", 99.0) < lnt.get("shift_mean_mae", 99.0))
            logo_info = {"logo_patnet_shift_mae": lp.get("shift_mean_mae"), "logo_no_transport_shift_mae": lnt.get("shift_mean_mae"), "logo_win": logo_ok}
        gates["gate3_causal_perspective_transport"] = {
            "primary_shift_mae": round(pm.get("shift_mean_mae", 99.0), 4),
            "no_transport_shift_mae": round(nt.get("shift_mean_mae", 99.0), 4),
            "shift_mae_p_holm": p_sh_holm,
            "primary_shift_pearson": round(pm.get("shift_mean_pearson", 0.0), 4),
            "no_transport_shift_pearson": round(nt.get("shift_mean_pearson", 0.0), 4),
            "primary_asymm_offense_auroc": round(pm.get("asymm_offense_auroc", 0.0), 4),
            "no_transport_asymm_offense_auroc": round(nt.get("asymm_offense_auroc", 0.0), 4),
            **logo_info,
            "passed": bool((pm.get("shift_mean_mae", 99.0) < nt.get("shift_mean_mae", 99.0)) and (p_sh_holm < 0.05) and logo_ok),
        }

    epic_agg = _aggregate_epic(ep_dir)
    brighter_agg = _aggregate_brighter(br_dir)

    em = epic_agg["models"]
    epic_gate_passed = False
    if "roberta_patnet_M4" in em and "roberta_beta_binom" in em and "roberta_patnet_no_transport" in em:
        ep_p = em["roberta_patnet_M4"]
        ep_bb = em["roberta_beta_binom"]
        ep_nt = em["roberta_patnet_no_transport"]
        fnr_mean = ep_p.get("crc_pol_alpha10_mean", {}).get("test_fnr", 1.0)
        epic_gate_passed = bool(
            (ep_p.get("oof_overall_pearson", 0.0) >= ep_bb.get("oof_overall_pearson", 0.0) - 0.01)
            and (ep_p.get("oof_pol_auroc_by_uplur", 0.0) > ep_p.get("oof_pol_auroc_by_uamb", 1.0))
            and (ep_p.get("oof_sex_shift_mae", 99.0) < ep_nt.get("oof_sex_shift_mae", 99.0))
            and (fnr_mean <= 0.12)
        )
    gates["gate4_epic_5fold_and_brighter"] = {
        "epic": epic_agg,
        "epic_passed": epic_gate_passed,
        "brighter": brighter_agg,
    }

    hw_path = OUT_DIR / "hardware_latency_vram.json"
    hw_info = json.loads(hw_path.read_text(encoding="utf-8")) if hw_path.exists() else None

    report = {
        "emobank_5fold_oof": summary_rows,
        "emobank_ablation_ladder_M0_to_M5": ablation_ladder,
        "emobank_deberta_v3_5fold": deberta_rows,
        "emobank_logo_7genre": logo_rows,
        "emobank_fold0_deep_ensembles_E5": ensembles,
        "cluster_bootstrap_2000": boot_report,
        "hardware_latency_vram": hw_info,
        "gates": gates,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "all_gates_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report["gates"].items() if k != "gate4_epic_5fold_and_brighter"}, indent=2))
    print("[EPIC Gate4]", json.dumps({"epic_passed": epic_gate_passed, "models": em}, indent=2))
    print(f"[BRIGHTER runs aggregated]: {len(brighter_agg)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobank-dir", type=str, default=None)
    parser.add_argument("--epic-dir", type=str, default=None)
    parser.add_argument("--brighter-dir", type=str, default=None)
    args = parser.parse_args()
    evaluate_all_gates(
        emobank_root=Path(args.emobank_dir) if args.emobank_dir else None,
        epic_root=Path(args.epic_dir) if args.epic_dir else None,
        brighter_root=Path(args.brighter_dir) if args.brighter_dir else None,
    )


if __name__ == "__main__":
    main()
