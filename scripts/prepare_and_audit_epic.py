"""Aggregate and audit EPIC irony corpus with strict parent_text group isolation and demographic counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import OUT_DIR, PROC_DIR, RAW_DIR  # noqa: E402

NAT_MAP = {
    "United Kingdom": "uk",
    "United States": "us",
    "Australia": "au",
    "Ireland": "ie",
    "India": "in",
}


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_DIR / "epic" / "EPICorpus.csv"
    if not raw_csv.exists():
        raw_csv = next((RAW_DIR / "epic").glob("*.csv"))

    df = pd.read_csv(raw_csv)
    df["y"] = (df["label"].astype(str).str.lower() == "iro").astype(int)
    df["parent_text_clean"] = df["parent_text"].fillna("").astype(str).str.strip()
    df["text_clean"] = df["text"].fillna("").astype(str).str.strip()
    df["sex_clean"] = df["Sex"].fillna("Unknown").astype(str).str.strip()
    df["nat_code"] = df["Nationality"].map(NAT_MAP).fillna("other")

    rows = []
    for item_id, grp in df.groupby("id_original", sort=True):
        first = grp.iloc[0]
        p_txt = str(first["parent_text_clean"])
        # Group by parent_text whenever non-empty so shared parents never cross splits
        group_key = f"parent::{p_txt}" if p_txt else f"id::{item_id}"

        n_annot = int(len(grp))
        k_iro = int(grp["y"].sum())
        p_iro = float(k_iro / max(n_annot, 1))

        m_grp = grp[grp["sex_clean"] == "Male"]
        f_grp = grp[grp["sex_clean"] == "Female"]
        n_male, k_male = int(len(m_grp)), int(m_grp["y"].sum())
        n_female, k_female = int(len(f_grp)), int(f_grp["y"].sum())

        p_iro_male = float(k_male / n_male) if n_male > 0 else float("nan")
        p_iro_female = float(k_female / n_female) if n_female > 0 else float("nan")

        row = {
            "id_original": str(item_id),
            "text": str(first["text_clean"]),
            "parent_text": p_txt,
            "parent_id_original": str(first.get("parent_id_original", "")),
            "group_key": group_key,
            "subreddit": str(first.get("subreddit", "")),
            "source": str(first.get("source", "")),
            "language_instance": str(first.get("Language_instance", "")),
            "language_variety": str(first.get("Language_variety", "unknown")),
            "n_annot": n_annot,
            "k_iro": k_iro,
            "p_iro": p_iro,
            "p_not": 1.0 - p_iro,
            "iro_std": float(np.sqrt(max(p_iro * (1.0 - p_iro), 0.0))),
            "n_male": n_male,
            "k_male": k_male,
            "p_iro_male": p_iro_male,
            "n_female": n_female,
            "k_female": k_female,
            "p_iro_female": p_iro_female,
            "both_sex_ge1": bool(n_male >= 1 and n_female >= 1),
            "both_sex_ge2": bool(n_male >= 2 and n_female >= 2),
            "is_polarized": bool((0.25 <= p_iro <= 0.75) and (n_annot >= 4)),
            "is_unanimous": bool((k_iro == 0 or k_iro == n_annot) and (n_annot >= 3)),
        }

        for code in ("uk", "us", "au", "ie", "in"):
            ng = grp[grp["nat_code"] == code]
            row[f"n_nat_{code}"] = int(len(ng))
            row[f"k_nat_{code}"] = int(ng["y"].sum())
            row[f"p_nat_{code}"] = float(ng["y"].mean()) if len(ng) > 0 else float("nan")

        rows.append(row)

    agg = pd.DataFrame(rows).sort_values("id_original").reset_index(drop=True)

    # Strict parent_text-disjoint 5-fold CV stratified by Language_variety
    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_fold = np.zeros(len(agg), dtype=int)
    for f_idx, (_, va_idx) in enumerate(sgkf5.split(agg, agg["language_variety"], agg["group_key"])):
        cv_fold[va_idx] = f_idx
    agg["cv_fold"] = cv_fold

    # Standard split column derived cleanly from fold 0: test=fold 0 (~20%), dev_cal=subsplit (~8%), dev_select (~7%), train (~65%)
    te_mask = agg["cv_fold"] == 0
    trval_df = agg[~te_mask].copy()
    sgkf8 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=100)
    sub_folds = list(sgkf8.split(trval_df, trval_df["language_variety"], trval_df["group_key"]))
    sel_rel_idx = sub_folds[0][1]
    cal_rel_idx = sub_folds[1][1]

    splits = np.full(len(agg), "train", dtype=object)
    trval_indices = trval_df.index.to_numpy()
    splits[te_mask.to_numpy()] = "test"
    splits[trval_indices[sel_rel_idx]] = "dev_select"
    splits[trval_indices[cal_rel_idx]] = "dev_cal"
    agg["split"] = splits

    # Verify zero parent_text leakage across splits
    non_empty = agg[agg["parent_text"] != ""]
    parent_split_n = non_empty.groupby("parent_text")["split"].nunique()
    cross_split_leaks = int((parent_split_n > 1).sum())
    assert cross_split_leaks == 0, f"Detected {cross_split_leaks} parent_text cross-split leaks!"

    out_pq = PROC_DIR / "epic_agg.parquet"
    agg.to_parquet(out_pq, index=False)

    both1 = agg[agg["both_sex_ge1"]]
    both2 = agg[agg["both_sex_ge2"]]
    sex_div1 = np.abs(both1["p_iro_female"] - both1["p_iro_male"])
    sex_div2 = np.abs(both2["p_iro_female"] - both2["p_iro_male"])

    nat_stats = df.groupby("Nationality")["y"].agg(["count", "mean"]).sort_values("count", ascending=False).to_dict("index")

    audit = {
        "n_raw_annotations": int(len(df)),
        "n_items": int(len(agg)),
        "unique_annotators": int(df["user"].nunique()),
        "shared_parent_text_cross_split_leaks": cross_split_leaks,
        "split_counts": agg["split"].value_counts().to_dict(),
        "cv_fold_counts": {int(k): int(v) for k, v in agg["cv_fold"].value_counts().sort_index().items()},
        "mean_p_iro": round(float(agg["p_iro"].mean()), 4),
        "polarized_items_n": int(agg["is_polarized"].sum()),
        "polarized_frac": round(float(agg["is_polarized"].mean()), 4),
        "unanimous_items_n": int(agg["is_unanimous"].sum()),
        "both_sex_ge1_items_n": int(len(both1)),
        "both_sex_ge1_high_div_ge_0_4_frac": round(float((sex_div1 >= 0.4).mean()), 4),
        "both_sex_ge2_items_n": int(len(both2)),
        "both_sex_ge2_high_div_ge_0_35_n": int((sex_div2 >= 0.35).sum()),
        "both_sex_ge2_high_div_ge_0_35_frac": round(float((sex_div2 >= 0.35).mean()), 4),
        "top_nationalities_irony_rate": {
            k: {"count": int(v["count"]), "mean": round(float(v["mean"]), 4)} for k, v in nat_stats.items()
        },
    }
    (OUT_DIR / "audit_epic.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved: {out_pq} ({out_pq.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
