"""Multi-seed orchestrator with cache-hit support for GoEmotions Raw, EPIC, and BRIGHTER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.brighter_exp import run_brighter_exp
from src.epic_exp import run_epic_experiment
from src.goemotions_exp import run_ge_experiment


OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def cached_json(path: Path, fn, force: bool = False) -> Dict[str, object]:
    if path.exists() and not force:
        print(f"[CACHE HIT] {path.parent.name}")
        return json.loads(path.read_text(encoding="utf-8"))
    return fn()


def agg_table(rows: List[Dict[str, object]], group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for key, grp in df.groupby(group_cols, sort=False):
        rec: Dict[str, object] = {}
        key_tup = key if isinstance(key, tuple) else (key,)
        for c, v in zip(group_cols, key_tup):
            rec[c] = v
        rec["n_seeds"] = len(grp)
        for m in metric_cols:
            if m in grp.columns:
                vals = grp[m].dropna().to_numpy(dtype=float)
                if len(vals):
                    rec[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                    rec[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["epic", "goemotions", "brighter", "all"], default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 102, 202])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.task in ("epic", "all"):
        ep_rows = []
        for mt in ["scalar_bce", "patnet_no_transport", "mixture_beta"]:
            for sd in args.seeds:
                sp = OUT_ROOT / "epic" / f"{mt}_seed{sd}" / "summary.json"
                d = cached_json(sp, lambda mt=mt, sd=sd: run_epic_experiment(model_type=mt, seed=sd, epochs=5), args.force)
                ep_rows.append({"model_type": mt, "seed": sd, **d["test_metrics"]})
        ep_agg = agg_table(
            ep_rows,
            ["model_type"],
            ["soft_bce", "p_iro_mae", "var_mae", "pol_vs_unan_auroc", "male_bce", "female_bce", "sex_shift_mae", "between_sex_div_auroc_by_uplur"],
        )
        ep_agg.to_csv(OUT_ROOT / "epic" / "multi_seed_summary.csv", index=False)
        print("\n=== EPIC MULTI-SEED SUMMARY ===")
        print(ep_agg.to_string(index=False))

    if args.task in ("goemotions", "all"):
        ge_rows = []
        for mode, models in [
            ("total", ["flat_bce", "mc_dropout_bce", "apm_no_cooccur", "apm_net"]),
            ("clear", ["flat_bce", "apm_net"]),
        ]:
            for mt in models:
                for sd in args.seeds:
                    sp = OUT_ROOT / "goemotions" / f"{mode}_{mt}_seed{sd}" / "summary.json"
                    d = cached_json(
                        sp,
                        lambda mt=mt, mode=mode, sd=sd: run_ge_experiment(model_type=mt, target_mode=mode, seed=sd, epochs=3),
                        args.force,
                    )
                    ge_rows.append({"target_mode": mode, "model_type": mt, "seed": sd, **d["test_metrics"]})
        ge_agg = agg_table(
            ge_rows,
            ["target_mode", "model_type"],
            [
                "soft_bce",
                "cooccur_offdiag_rmse",
                "unclear_auroc_by_uamb",
                "unclear_auroc_by_uplur",
                "clear_posneg_auroc_by_uplur",
                "clear_posneg_auroc_by_uamb",
                "dissociation_unclear_gap",
                "dissociation_posneg_gap",
                "abstain10_unclear_recall_uamb",
                "abstain10_clear_pol_damage_uamb",
                "abstain10_clear_pol_damage_entropy",
            ],
        )
        ge_agg.to_csv(OUT_ROOT / "goemotions" / "multi_seed_summary.csv", index=False)
        print("\n=== GOEMOTIONS MULTI-SEED SUMMARY ===")
        print(ge_agg.to_string(index=False))

    if args.task in ("brighter", "all"):
        br_rows = []
        for mt in ["scalar_mse", "gaussian_nll", "softmax_ce", "dml_unimodal", "dml_bimodal"]:
            for sd in args.seeds:
                sp = OUT_ROOT / "brighter" / f"eng_{mt}_seed{sd}" / "summary.json"
                d = cached_json(sp, lambda mt=mt, sd=sd: run_brighter_exp(lang="eng", model_type=mt, seed=sd, epochs=6), args.force)
                br_rows.append({"lang": "eng", "model_type": mt, "seed": sd, **d["test_metrics"]})
        br_agg = agg_table(br_rows, ["lang", "model_type"], ["mean_pearson", "mean_ccc", "mean_mae", "mean_w1"])
        br_agg.to_csv(OUT_ROOT / "brighter" / "multi_seed_summary.csv", index=False)
        print("\n=== BRIGHTER MULTI-SEED SUMMARY ===")
        print(br_agg.to_string(index=False))


if __name__ == "__main__":
    main()
