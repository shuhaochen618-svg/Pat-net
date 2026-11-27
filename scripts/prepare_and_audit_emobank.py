"""Reconstruct QC-clean EmoBank distributions (0% worker noise leakage) and audit empirical gaps."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_EB = ROOT / "data" / "raw" / "emobank"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"


def reconstruct_qc_clean_histograms(
    agg_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    prefix: str,
) -> tuple[pd.DataFrame, dict]:
    """Match published QC stats (V,A,D,stdV,stdA,stdD,N) to exact subset of raw ratings."""
    grouped = {k: g[["V", "A", "D"]].to_numpy(dtype=float) for k, g in raw_df.groupby("id")}
    dims = ["V", "A", "D"]
    rows: list[dict] = []
    exact_n = 0
    recovered_sub = 0
    unmatched = 0
    max_stat_err = 0.0

    for _, row in agg_df.iterrows():
        sid = row["id"]
        if sid not in grouped:
            continue
        arr = grouped[sid]
        n_target = int(row["N"])
        target = np.array(
            [row["V"], row["A"], row["D"], row["stdV"], row["stdA"], row["stdD"]],
            dtype=float,
        )

        if len(arr) == n_target:
            sub = arr
            exact_n += 1
        elif len(arr) > n_target:
            best_sub = None
            best_err = float("inf")
            for idxs in itertools.combinations(range(len(arr)), n_target):
                cand = arr[list(idxs)]
                stats = np.concatenate([cand.mean(axis=0), cand.std(axis=0, ddof=0)])
                err = float(np.max(np.abs(stats - target)))
                if err < best_err:
                    best_err = err
                    best_sub = cand
                if err <= 0.015:
                    break
            if best_sub is not None and best_err <= 0.02:
                sub = best_sub
                recovered_sub += 1
            else:
                unmatched += 1
                continue
        else:
            unmatched += 1
            continue

        stats = np.concatenate([sub.mean(axis=0), sub.std(axis=0, ddof=0)])
        max_stat_err = max(max_stat_err, float(np.max(np.abs(stats - target))))

        rec: dict[str, object] = {"id": sid, f"{prefix}_n": int(len(sub))}
        for j, d in enumerate(dims):
            vals = np.clip(np.rint(sub[:, j]).astype(int), 1, 5)
            counts = np.bincount(vals, minlength=6)[1:6]
            probs = counts / float(counts.sum())
            for k in range(1, 6):
                rec[f"{prefix}_{d}_c{k}"] = int(counts[k - 1])
                rec[f"{prefix}_{d}_p{k}"] = float(probs[k - 1])
            scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mean = float((probs * scores).sum())
            var = float((probs * (scores - mean) ** 2).sum())
            std = float(np.sqrt(max(var, 0.0)))
            ent = float(-np.sum(np.where(probs > 0, probs * np.log(np.clip(probs, 1e-12, 1.0)), 0.0)))
            p_low = float(probs[0] + probs[1])
            p_high = float(probs[3] + probs[4])
            rec[f"{prefix}_{d}_mean"] = mean
            rec[f"{prefix}_{d}_std"] = std
            rec[f"{prefix}_{d}_var"] = var
            rec[f"{prefix}_{d}_entropy"] = ent
            rec[f"{prefix}_{d}_bipole_min"] = min(p_low, p_high)
            rec[f"{prefix}_{d}_has_hist"] = True
        rows.append(rec)

    meta = {
        "exact_n": exact_n,
        "recovered_subset": recovered_sub,
        "unmatched": unmatched,
        "max_stat_err_vs_published_csv": round(max_stat_err, 6),
    }
    return pd.DataFrame(rows), meta


def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    cdf_p = np.cumsum(p, axis=-1)[..., :-1]
    cdf_q = np.cumsum(q, axis=-1)[..., :-1]
    return np.abs(cdf_p - cdf_q).sum(axis=-1)


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emobank = pd.read_csv(RAW_EB / "emobank.csv")
    meta = pd.read_csv(RAW_EB / "meta.tsv", sep="\t")
    writer_agg = pd.read_csv(RAW_EB / "writer.csv")
    reader_agg = pd.read_csv(RAW_EB / "reader.csv")
    indiv_w = pd.read_csv(RAW_EB / "individual_writer_ratings.csv")
    indiv_r = pd.read_csv(RAW_EB / "individual_reader_ratings.csv")

    r_hist, r_meta = reconstruct_qc_clean_histograms(reader_agg, indiv_r, "r")
    w_hist, w_meta = reconstruct_qc_clean_histograms(writer_agg, indiv_w, "w")

    # Start from all 10,062 benchmark sentences with full published QC Writer & Reader moments
    w_moments = writer_agg.rename(
        columns={
            "V": "w_V_pub_mean",
            "A": "w_A_pub_mean",
            "D": "w_D_pub_mean",
            "stdV": "w_V_pub_std",
            "stdA": "w_A_pub_std",
            "stdD": "w_D_pub_std",
            "N": "w_pub_n",
        }
    )
    r_moments = reader_agg.rename(
        columns={
            "V": "r_V_pub_mean",
            "A": "r_A_pub_mean",
            "D": "r_D_pub_mean",
            "stdV": "r_V_pub_std",
            "stdA": "r_A_pub_std",
            "stdD": "r_D_pub_std",
            "N": "r_pub_n",
        }
    )

    df = emobank.merge(meta, on="id", how="left")
    df = df.merge(w_moments, on="id", how="inner")
    df = df.merge(r_moments, on="id", how="inner")
    df = df.merge(r_hist, on="id", how="left")
    df = df.merge(w_hist, on="id", how="left")

    for d in ["V", "A", "D"]:
        df[f"w_{d}_has_hist"] = df[f"w_{d}_has_hist"].fillna(False).astype(bool)
        df[f"r_{d}_has_hist"] = df[f"r_{d}_has_hist"].fillna(False).astype(bool)
        df[f"shift_signed_{d}"] = df[f"r_{d}_pub_mean"] - df[f"w_{d}_pub_mean"]
        df[f"shift_abs_{d}"] = np.abs(df[f"shift_signed_{d}"])

    # Compute exact W1 on dual-histogram subset (4,199 rows across blog, essays, fiction)
    dual_mask = df["w_V_has_hist"] & df["r_V_has_hist"]
    for d in ["V", "A", "D"]:
        p_w = df.loc[dual_mask, [f"w_{d}_p{k}" for k in range(1, 6)]].to_numpy()
        p_r = df.loc[dual_mask, [f"r_{d}_p{k}" for k in range(1, 6)]].to_numpy()
        df.loc[dual_mask, f"shift_w1_{d}"] = wasserstein_1d(p_w, p_r)

    # Verify emobank.csv V/A/D vs average or reader/writer
    diff_r = float(np.abs(df["V"] - df["r_V_pub_mean"]).max())
    diff_avg = float(np.abs(df["V"] - 0.5 * (df["w_V_pub_mean"] + df["r_V_pub_mean"])).max())

    # Document leakage check across splits + genre-level stats over ALL 10,062
    split_docs = {
        sp: set(df.loc[df["split"] == sp, "document"].dropna().astype(str))
        for sp in ["train", "dev", "test"]
    }
    collapse_reader = {}
    for d in ["V", "A", "D"]:
        m = df[f"r_{d}_mean"]
        s = df[f"r_{d}_std"]
        bp = df[f"r_{d}_bipole_min"]
        mid = (m >= 2.5) & (m <= 3.5)
        unanimous_neutral = mid & (s < 0.05)
        polarized = mid & ((s >= 0.8) | (bp > 0))
        diffuse = mid & (s >= 0.45) & (s < 0.8) & (bp == 0)
        collapse_reader[d] = {
            "total_n": int(len(df)),
            "mean_2p5_3p5_n": int(mid.sum()),
            "unanimous_neutral_n": int(unanimous_neutral.sum()),
            "polarized_n": int(polarized.sum()),
            "polarized_frac_in_neutral_band": round(float(polarized.sum() / max(int(mid.sum()), 1)), 4),
            "diffuse_moderate_n": int(diffuse.sum()),
            "bimodal_both_tails_all_n": int((bp > 0).sum()),
        }

    genre_stats = {}
    for cat, g in df.groupby("category", dropna=False):
        cat_name = str(cat) if pd.notna(cat) else "MISSING"
        genre_stats[cat_name] = {
            "n": int(len(g)),
            "has_writer_hist_n": int(g["w_V_has_hist"].sum()),
            "has_reader_hist_n": int(g["r_V_has_hist"].sum()),
            "V_signed_R_minus_W": round(float(g["shift_signed_V"].mean()), 4),
            "A_signed_R_minus_W": round(float(g["shift_signed_A"].mean()), 4),
            "D_signed_R_minus_W": round(float(g["shift_signed_D"].mean()), 4),
            "frac_abs_V_ge_0p5": round(float((g["shift_abs_V"] >= 0.5).mean()), 4),
            "frac_abs_A_ge_0p5": round(float((g["shift_abs_A"] >= 0.5).mean()), 4),
            "frac_abs_D_ge_0p5": round(float((g["shift_abs_D"] >= 0.5).mean()), 4),
        }
        if g["w_V_has_hist"].sum() > 0:
            sub = g[g["w_V_has_hist"]]
            genre_stats[cat_name]["V_w1_mean"] = round(float(sub["shift_w1_V"].mean()), 4)
            genre_stats[cat_name]["A_w1_mean"] = round(float(sub["shift_w1_A"].mean()), 4)
            genre_stats[cat_name]["D_w1_mean"] = round(float(sub["shift_w1_D"].mean()), 4)

    out_pq = PROC_DIR / "emobank_full.parquet"
    df.to_parquet(out_pq, index=False)

    audit = {
        "total_sentences": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "reader_hist_coverage": int(df["r_V_has_hist"].sum()),
        "writer_hist_coverage": int(df["w_V_has_hist"].sum()),
        "qc_reconstruction": {"reader": r_meta, "writer": w_meta},
        "emobank_V_vs_reader_max_diff": round(diff_r, 4),
        "emobank_V_vs_mean_WR_max_diff": round(diff_avg, 4),
        "document_overlap": {
            "train_test_shared_docs": len(split_docs["train"] & split_docs["test"]),
            "train_dev_shared_docs": len(split_docs["train"] & split_docs["dev"]),
            "total_unique_docs": int(df["document"].nunique()),
        },
        "reader_mean_collapse_audit_10062": collapse_reader,
        "genre_perspective_shift_10062": genre_stats,
    }
    (OUT_DIR / "audit_emobank.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Saved: {out_pq} ({out_pq.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
