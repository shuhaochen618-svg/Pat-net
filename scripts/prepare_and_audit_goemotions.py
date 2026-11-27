"""Prepare and audit GoEmotions Raw for Ambiguity (example_very_unclear) vs Pluralism (clear rater conflict)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GE_RAW = ROOT / "data" / "raw" / "go_emotions" / "full_dataset"
GE_SIMP = ROOT / "data" / "raw" / "go_emotions" / "simplified"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

POS_EMOS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEG_EMOS = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}


def avg_pairwise_jaccard_disagreement(binary_mat: np.ndarray) -> float:
    """Compute 1 - mean pairwise Jaccard similarity across raters on clear emotion sets."""
    n = len(binary_mat)
    if n < 2:
        return 0.0
    inter = binary_mat @ binary_mat.T
    sums = binary_mat.sum(axis=1, keepdims=True)
    union = sums + sums.T - inter
    iu, ju = np.triu_indices(n, k=1)
    u_pairs = union[iu, ju]
    i_pairs = inter[iu, ju]
    jacc = np.where(u_pairs > 0, i_pairs / np.maximum(u_pairs, 1.0), 1.0)
    return float(1.0 - np.mean(jacc))


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(GE_RAW / f"goemotions_{i}.csv") for i in (1, 2, 3)]
    raw = pd.concat(dfs, ignore_index=True)

    meta_cols = ["text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"]
    emo_cols = [c for c in raw.columns if c not in meta_cols]
    assert len(emo_cols) == 28, f"Expected 28 emotion columns, got {len(emo_cols)}: {emo_cols}"

    # Verify boolean mapping of example_very_unclear
    raw["unclear"] = raw["example_very_unclear"].astype(str).str.lower().isin(["true", "1"])

    # Load simplified official splits to guarantee zero split leakage
    split_map = {}
    for sp, fn in [("train", "train-00000-of-00001.parquet"), ("validation", "validation-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")]:
        sdf = pd.read_parquet(GE_SIMP / fn)
        for eid in sdf["id"].astype(str):
            split_map[eid] = sp

    pos_idx = [i for i, c in enumerate(emo_cols) if c in POS_EMOS]
    neg_idx = [i for i, c in enumerate(emo_cols) if c in NEG_EMOS]

    # Aggregate per example id (saving first-moment proportions + exact second-moment co-occurrence)
    records = []
    cooccur_total_list = []
    cooccur_clear_list = []
    for eid, grp in raw.groupby("id", sort=False):
        n_total = len(grp)
        unclear_n = int(grp["unclear"].sum())
        unclear_frac = unclear_n / float(n_total)
        clear_grp = grp[~grp["unclear"]]
        n_clear = len(clear_grp)

        mat_all = grp[emo_cols].to_numpy(dtype=np.float32)
        probs_total = mat_all.mean(axis=0)
        co_total = (mat_all.T @ mat_all) / float(n_total)

        if n_clear > 0:
            mat_clear = clear_grp[emo_cols].to_numpy(dtype=np.float32)
            probs = mat_clear.mean(axis=0)
            co_clear = (mat_clear.T @ mat_clear) / float(n_clear)
            disagree = avg_pairwise_jaccard_disagreement(mat_clear)
            has_pos = float((mat_clear[:, pos_idx].sum(axis=1) > 0).any())
            has_neg = float((mat_clear[:, neg_idx].sum(axis=1) > 0).any())
            pol_conflict = float(has_pos * has_neg)
        else:
            probs = np.zeros(len(emo_cols), dtype=np.float32)
            co_clear = np.zeros((len(emo_cols), len(emo_cols)), dtype=np.float32)
            disagree = 0.0
            pol_conflict = 0.0

        cooccur_total_list.append(co_total.astype(np.float16))
        cooccur_clear_list.append(co_clear.astype(np.float16))

        # Binary label-wise Shannon entropy
        p_clip = np.clip(probs, 1e-7, 1.0 - 1e-7)
        bin_ent = float(np.sum(-p_clip * np.log2(p_clip) - (1.0 - p_clip) * np.log2(1.0 - p_clip)))

        rec = {
            "id": str(eid),
            "text": str(grp["text"].iloc[0]),
            "split_simplified": split_map.get(str(eid), "raw_only"),
            "n_raters": n_total,
            "n_clear": n_clear,
            "unclear_count": unclear_n,
            "unclear_frac": float(unclear_frac),
            "has_any_unclear": bool(unclear_n > 0),
            "clear_disagree_jaccard": float(disagree),
            "pos_neg_conflict": float(pol_conflict),
            "total_bernoulli_entropy": bin_ent,
        }
        for c, val in zip(emo_cols, probs):
            rec[f"p_clear_{c}"] = float(val)
        for c, val in zip(emo_cols, probs_total):
            rec[f"p_total_{c}"] = float(val)
        records.append(rec)

    out_df = pd.DataFrame(records)
    out_pq = PROC_DIR / "goemotions_raw_agg.parquet"
    out_df.to_parquet(out_pq, index=False)
    np.savez_compressed(
        PROC_DIR / "goemotions_cooccur_f16.npz",
        co_total=np.stack(cooccur_total_list, axis=0),
        co_clear=np.stack(cooccur_clear_list, axis=0),
        emo_cols=np.array(emo_cols),
    )

    # Audit dissociation numbers
    clear_only = out_df[~out_df["has_any_unclear"]]
    unclear_any = out_df[out_df["has_any_unclear"]]
    audit = {
        "raw_annotations": int(len(raw)),
        "unique_examples": int(len(out_df)),
        "unique_raters": int(raw["rater_id"].nunique()),
        "raw_unclear_rows": int(raw["unclear"].sum()),
        "examples_with_any_unclear": int(len(unclear_any)),
        "examples_with_majority_unclear": int((out_df["unclear_frac"] >= 0.5).sum()),
        "examples_all_clear": int(len(clear_only)),
        "clear_only_stats": {
            "disagree_gt_0": int((clear_only["clear_disagree_jaccard"] > 0).sum()),
            "disagree_ge_0p5": int((clear_only["clear_disagree_jaccard"] >= 0.5).sum()),
            "pos_neg_conflict_n": int((clear_only["pos_neg_conflict"] > 0).sum()),
            "pos_neg_conflict_frac": round(float((clear_only["pos_neg_conflict"] > 0).mean()), 4),
            "mean_entropy_when_disagree_ge_0p5": round(float(clear_only.loc[clear_only["clear_disagree_jaccard"] >= 0.5, "total_bernoulli_entropy"].mean()), 4),
        },
        "unclear_any_stats": {
            "mean_entropy": round(float(unclear_any["total_bernoulli_entropy"].mean()), 4),
            "pos_neg_conflict_frac": round(float((unclear_any["pos_neg_conflict"] > 0).mean()), 4),
        },
        "split_counts": out_df["split_simplified"].value_counts().to_dict(),
    }
    (OUT_DIR / "audit_goemotions.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Saved: {out_pq} ({out_pq.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
