"""Multi-seed experiment & ablation orchestrator for EmoBank PAT-Net benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import dirac_wasserstein1, discrete_jsd, discrete_wasserstein1
from src.train_emobank import run_single


OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "emobank"


def make_args(**overrides) -> SimpleNamespace:
    defaults = {
        "backbone": "roberta-base",
        "model_type": "patnet",
        "protocol": "official",
        "seed": 42,
        "epochs": 6,
        "batch_size": 32,
        "max_length": 128,
        "latent_dim": 256,
        "dropout": 0.2,
        "encoder_lr": 2e-5,
        "head_lr": 3e-4,
        "weight_decay": 0.01,
        "w_kl": 1.0,
        "w_w1": 1.0,
        "w_mom": 0.5,
        "w_shift_mean": 0.5,
        "w_shift_cdf": 0.5,
        "w_plur_reg": 0.005,
        "w_trans_reg": 0.01,
        "hf_endpoint": "https://hf-mirror.com",
        "out_dir": str(OUT_ROOT),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def load_or_run(tag: str, **kwargs) -> Dict[str, object]:
    out_dir = Path(kwargs.get("out_dir", OUT_ROOT))
    backbone_slug = kwargs.get("backbone", "roberta-base").replace("/", "_")
    run_dir = out_dir / f"{kwargs.get('protocol','official')}_{backbone_slug}_{tag}_seed{kwargs['seed']}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not kwargs.pop("force", False):
        print(f"[CACHE HIT] {run_dir.name}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    kwargs.pop("force", None)
    return run_single(make_args(**kwargs))


def aggregate_runs(summaries: List[Dict[str, object]], group_key: str = "model_type") -> pd.DataFrame:
    metrics = [
        "r_mean_w1",
        "r_bimod_w1",
        "r_mean_jsd",
        "r_mean_pearson",
        "r_mean_ccc",
        "r_mean_std_mae",
        "r_mean_pol_auroc_by_std",
        "r_bimod_vs_diff_auroc_by_uplur",
        "r_bimod_vs_diff_auroc_by_uamb",
        "r_bimod_vs_diff_auroc_by_std",
        "w_mean_w1",
        "w_mean_pearson",
        "shift_mean_mae",
        "shift_mean_high_auroc",
    ]
    by_group: Dict[str, List[Dict[str, float]]] = {}
    for s in summaries:
        key = str(s[group_key])
        by_group.setdefault(key, []).append(s["test_metrics"])

    rows = []
    for name, list_m in by_group.items():
        row: Dict[str, object] = {group_key: name, "n_seeds": len(list_m)}
        for m in metrics:
            vals = [x[m] for x in list_m if m in x and not np.isnan(x[m])]
            if vals:
                row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _sentence_errors(npz_path: Path, model_type: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    p_t, p_q = d["tgt_r_probs"], d["r_probs"]
    w1_arr = (
        dirac_wasserstein1(d["r_disc_mean"], p_t).mean(axis=-1)
        if model_type == "scalar_mse"
        else discrete_wasserstein1(p_t, p_q).mean(axis=-1)
    )
    jsd_arr = discrete_jsd(p_t, p_q).mean(axis=-1)
    shift_true = d["tgt_r_mean"] - d["tgt_w_mean"]
    shift_pred = d["r_disc_mean"] - d["w_disc_mean"]
    shift_ae = np.abs(shift_true - shift_pred).mean(axis=-1)
    return {"w1": w1_arr, "jsd": jsd_arr, "shift_ae": shift_ae}


def paired_tests(
    summaries: List[Dict[str, object]],
    ref_model: str = "patnet",
    protocol: str = "official",
    backbone: str = "roberta-base",
    out_dir: Path = OUT_ROOT,
) -> Dict[str, object]:
    """Compare all baseline models vs ref_model using per-seed paired metrics + item-level Wilcoxon/bootstrap."""
    by_model: Dict[str, Dict[int, Dict[str, float]]] = {}
    for s in summaries:
        by_model.setdefault(str(s["model_type"]), {})[int(s["seed"])] = s["test_metrics"]

    if ref_model not in by_model:
        return {}
    ref_seeds = by_model[ref_model]
    b_slug = backbone.replace("/", "_")
    rng = np.random.default_rng(42)
    results = {}
    for m_name, seed_dict in by_model.items():
        if m_name == ref_model:
            continue
        common = sorted(set(ref_seeds) & set(seed_dict))
        if len(common) < 1:
            continue
        comp: Dict[str, object] = {}
        for metric in [
            "r_mean_w1",
            "r_bimod_w1",
            "r_mean_jsd",
            "r_mean_pearson",
            "r_mean_pol_auroc_by_std",
            "r_bimod_vs_diff_auroc_by_uplur",
            "shift_mean_mae",
        ]:
            a = np.array([ref_seeds[sd].get(metric, np.nan) for sd in common], dtype=float)
            b = np.array([seed_dict[sd].get(metric, np.nan) for sd in common], dtype=float)
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            diff = a - b
            p_val = float(wilcoxon(diff).pvalue) if len(common) >= 5 and np.any(diff != 0) else np.nan
            comp[metric] = {
                "n_seeds": len(common),
                "mean_diff_pat_minus_base": round(float(diff.mean()), 5),
                "seed_wilcoxon_p": None if np.isnan(p_val) else round(p_val, 5),
            }

        # Sentence-level paired significance averaged across matched seeds (N=1,000 test items)
        ref_errs, bas_errs = [], []
        for sd in common:
            rp = out_dir / f"{protocol}_{b_slug}_{ref_model}_seed{sd}" / "test_predictions.npz"
            bp = out_dir / f"{protocol}_{b_slug}_{m_name}_seed{sd}" / "test_predictions.npz"
            if rp.exists() and bp.exists():
                ref_errs.append(_sentence_errors(rp, ref_model))
                bas_errs.append(_sentence_errors(bp, m_name))
        if ref_errs:
            sent_tests = {}
            for k in ("w1", "jsd", "shift_ae"):
                e_ref = np.mean([x[k] for x in ref_errs], axis=0)
                e_bas = np.mean([x[k] for x in bas_errs], axis=0)
                d_vec = e_ref - e_bas
                w_p = float(wilcoxon(d_vec).pvalue) if np.any(d_vec != 0) else 1.0
                n_items = len(d_vec)
                idx = rng.integers(0, n_items, size=(2000, n_items))
                boot_means = d_vec[idx].mean(axis=1)
                boot_p = float(np.mean(boot_means >= 0.0)) if d_vec.mean() < 0 else float(np.mean(boot_means <= 0.0))
                sent_tests[k] = {
                    "n_items": int(n_items),
                    "mean_diff_pat_minus_base": round(float(d_vec.mean()), 5),
                    "item_wilcoxon_p": float(f"{w_p:.3e}"),
                    "bootstrap_one_sided_p": round(boot_p, 4),
                }
            comp["item_level_significance"] = sent_tests
        results[m_name] = comp
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "main", "doc_disjoint", "logo", "ablation"], default="main")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 102, 202, 302, 402])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []

    if args.phase == "smoke":
        s = load_or_run("patnet", model_type="patnet", seed=42, epochs=1, backbone=args.backbone, force=args.force)
        print("SMOKE OK:", json.dumps(s["test_metrics"], indent=2))
        return

    if args.phase == "main":
        models = [
            "scalar_mse",
            "gaussian_nll",
            "nig_evidential",
            "softmax_ldl",
            "patnet_unimodal",
            "patnet_no_transport",
            "patnet",
        ]
        for mt in models:
            for sd in args.seeds:
                summaries.append(
                    load_or_run(
                        mt,
                        model_type=mt,
                        protocol="official",
                        seed=sd,
                        epochs=args.epochs,
                        backbone=args.backbone,
                        force=args.force,
                    )
                )
        b_sfx = "" if args.backbone == "roberta-base" else f"_{args.backbone.replace('/', '_')}"
        agg = aggregate_runs(summaries)
        agg.to_csv(OUT_ROOT / f"main_official{b_sfx}_summary.csv", index=False)
        sig = paired_tests(summaries, protocol="official", backbone=args.backbone)
        (OUT_ROOT / f"main_official{b_sfx}_significance.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
        print("\n=== MAIN OFFICIAL SUMMARY ===")
        print(agg.to_string(index=False))

    elif args.phase == "doc_disjoint":
        for mt in ["scalar_mse", "gaussian_nll", "softmax_ldl", "patnet_no_transport", "patnet"]:
            for sd in args.seeds[:3]:
                summaries.append(
                    load_or_run(
                        mt,
                        model_type=mt,
                        protocol="doc_disjoint",
                        seed=sd,
                        epochs=args.epochs,
                        backbone=args.backbone,
                        force=args.force,
                    )
                )
        agg = aggregate_runs(summaries)
        agg.to_csv(OUT_ROOT / "doc_disjoint_summary.csv", index=False)
        print("\n=== DOC-DISJOINT SUMMARY ===")
        print(agg.to_string(index=False))

    elif args.phase == "logo":
        for genre in ["SemEval", "fiction", "letters"]:
            for mt in ["gaussian_nll", "softmax_ldl", "patnet_no_transport", "patnet"]:
                for sd in args.seeds[:3]:
                    s = load_or_run(
                        mt,
                        model_type=mt,
                        protocol=f"logo_{genre}",
                        seed=sd,
                        epochs=args.epochs,
                        backbone=args.backbone,
                        force=args.force,
                    )
                    s["label"] = f"{genre}__{mt}"
                    summaries.append(s)
        agg = aggregate_runs(summaries, group_key="label")
        agg.to_csv(OUT_ROOT / "logo_summary.csv", index=False)
        print("\n=== LEAVE-ONE-GENRE-OUT SUMMARY ===")
        print(agg.to_string(index=False))

    elif args.phase == "ablation":
        abl_configs = [
            ("full_patnet", {}),
            ("no_w1_loss", {"w_w1": 0.0}),
            ("no_kl_loss", {"w_kl": 0.0}),
            ("no_shift_cdf_loss", {"w_shift_cdf": 0.0, "w_shift_mean": 0.0}),
            ("no_plur_reg", {"w_plur_reg": 0.0}),
        ]
        for label, ov in abl_configs:
            for sd in args.seeds[:3]:
                out_dir = OUT_ROOT / "ablations" / label
                s = run_single(
                    make_args(
                        model_type="patnet",
                        protocol="official",
                        seed=sd,
                        epochs=args.epochs,
                        backbone=args.backbone,
                        out_dir=str(out_dir),
                        **ov,
                    )
                )
                s["ablation"] = label
                summaries.append(s)
        agg = aggregate_runs(summaries, group_key="ablation")
        agg.to_csv(OUT_ROOT / "ablation_summary.csv", index=False)
        print("\n=== ABLATION SUMMARY ===")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
