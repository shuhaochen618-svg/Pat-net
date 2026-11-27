"""Clean document-disjoint, genre-stratified EmoBank dataset splits and tensor collation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .paths import PROC_DIR
from .struct_features import STRUCT_DIM, extract_struct_matrix

EMOBANK_PARQUET = PROC_DIR / "emobank_full.parquet"
DIMS = ("V", "A", "D")


def load_emobank_dataframe(parquet_path: str | None = None) -> pd.DataFrame:
    path = EMOBANK_PARQUET if parquet_path is None else PROC_DIR / parquet_path
    df = pd.read_parquet(path)
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["document"] = df["document"].fillna(df["id"].astype(str)).astype(str)
    return df


def _stratified_doc_subsplit(
    df_sub: pd.DataFrame,
    n_splits: int = 8,
    fold_idx: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically carve a genre-balanced document-disjoint subset."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(sgkf.split(df_sub, df_sub["category"], df_sub["document"]))
    tr_idx, val_idx = folds[fold_idx % len(folds)]
    return df_sub.iloc[tr_idx].reset_index(drop=True), df_sub.iloc[val_idx].reset_index(drop=True)


def get_emobank_4way_splits(
    df: pd.DataFrame,
    protocol: str = "cv_fold_0",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, dev_select, dev_cal, test) with strict document isolation for CV protocols."""
    if protocol == "official":
        df_tr_raw = df[df["split"] == "train"].reset_index(drop=True)
        df_dv = df[df["split"] == "dev"].reset_index(drop=True)
        df_te = df[df["split"] == "test"].reset_index(drop=True)
        # Keep full dev=1000 untouched as dev_cal for rare-class FNR conformal calibration;
        # carve 10% deterministic group split from train as dev_select (or legacy doc overlap noted).
        gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        tr_i, sel_i = next(gss.split(df_tr_raw, groups=df_tr_raw["document"]))
        return (
            df_tr_raw.iloc[tr_i].reset_index(drop=True),
            df_tr_raw.iloc[sel_i].reset_index(drop=True),
            df_dv,
            df_te,
        )

    if protocol == "doc_disjoint" or protocol.startswith("cv_fold_"):
        fold = 0 if protocol == "doc_disjoint" else int(protocol.split("_")[-1])
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(sgkf.split(df, df["category"], df["document"]))
        trval_idx, te_idx = folds[fold % 5]
        df_trval = df.iloc[trval_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)
        # Split trval cleanly by document: ~1/8 dev_select, ~1/8 dev_cal, ~6/8 train (all doc-disjoint!)
        sgkf_sub = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=100 + fold)
        sub_folds = list(sgkf_sub.split(df_trval, df_trval["category"], df_trval["document"]))
        sel_idx = sub_folds[0][1]
        cal_idx = sub_folds[1][1]
        used = np.zeros(len(df_trval), dtype=bool)
        used[sel_idx] = True
        used[cal_idx] = True
        df_tr = df_trval.iloc[~used].reset_index(drop=True)
        df_sel = df_trval.iloc[sel_idx].reset_index(drop=True)
        df_cal = df_trval.iloc[cal_idx].reset_index(drop=True)
        return df_tr, df_sel, df_cal, df_te

    if protocol.startswith("logo_"):
        held_genre = protocol.split("logo_", 1)[1]
        df_te = df[df["category"] == held_genre].reset_index(drop=True)
        df_trval = df[df["category"] != held_genre].reset_index(drop=True)
        rem, df_cal = _stratified_doc_subsplit(df_trval, n_splits=8, fold_idx=0)
        df_tr, df_sel = _stratified_doc_subsplit(rem, n_splits=7, fold_idx=1)
        return df_tr, df_sel, df_cal, df_te

    raise ValueError(f"Unknown EmoBank split protocol: {protocol}")


def get_emobank_splits(
    df: pd.DataFrame,
    protocol: str = "official",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible 3-split API returning (train, dev_select, test)."""
    tr, sel, _, te = get_emobank_4way_splits(df, protocol=protocol, seed=seed)
    return tr, sel, te


def compute_train_prevalence(df_tr: pd.DataFrame) -> Dict[str, float]:
    """Compute fixed training-set histogram prevalence to stabilize masked KL/W1 gradients."""
    w_has = df_tr["w_V_has_hist"].astype(bool).to_numpy()
    r_has = df_tr["r_V_has_hist"].astype(bool).to_numpy()
    wr_has = w_has & r_has
    return {
        "w_hist_prev": float(max(np.mean(w_has), 0.05)),
        "r_hist_prev": float(max(np.mean(r_has), 0.05)),
        "wr_hist_prev": float(max(np.mean(wr_has), 0.05)),
    }


def fit_struct_scaler_for_splits(
    df_tr: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> Tuple[np.ndarray, ...]:
    """Extract 20d structural cues and standardize strictly using training statistics."""
    tr_raw = extract_struct_matrix(df_tr["text"].astype(str).tolist())
    scaler = StandardScaler().fit(tr_raw)
    outs = [scaler.transform(tr_raw).astype(np.float32)]
    for d in other_dfs:
        raw = extract_struct_matrix(d["text"].astype(str).tolist())
        outs.append(scaler.transform(raw).astype(np.float32))
    return tuple(outs)


class EmoBankDistributionDataset(Dataset):
    """Stores texts, exact Writer/Reader moment/histogram targets, metadata, and structural cues."""

    def __init__(self, df: pd.DataFrame, struct_matrix: np.ndarray | None = None) -> None:
        self.ids = df["id"].astype(str).tolist()
        self.texts = df["text"].astype(str).tolist()
        self.categories = df["category"].fillna("unknown").astype(str).tolist()
        self.documents = df["document"].fillna("").astype(str).tolist()

        if struct_matrix is not None:
            self.struct_feats = np.asarray(struct_matrix, dtype=np.float32)
        else:
            self.struct_feats = np.zeros((len(df), STRUCT_DIM), dtype=np.float32)

        self.w_has_hist = df["w_V_has_hist"].astype(bool).to_numpy()
        self.r_has_hist = df["r_V_has_hist"].astype(bool).to_numpy()

        n = len(df)
        self.w_probs = np.full((n, 3, 5), 0.2, dtype=np.float32)
        self.r_probs = np.full((n, 3, 5), 0.2, dtype=np.float32)
        self.w_mean = np.zeros((n, 3), dtype=np.float32)
        self.w_std = np.zeros((n, 3), dtype=np.float32)
        self.r_mean = np.zeros((n, 3), dtype=np.float32)
        self.r_std = np.zeros((n, 3), dtype=np.float32)

        for j, d in enumerate(DIMS):
            self.r_mean[:, j] = df[f"r_{d}_pub_mean"].to_numpy(dtype=np.float32)
            self.r_std[:, j] = df[f"r_{d}_pub_std"].to_numpy(dtype=np.float32)
            self.w_mean[:, j] = df[f"w_{d}_pub_mean"].to_numpy(dtype=np.float32)
            self.w_std[:, j] = df[f"w_{d}_pub_std"].to_numpy(dtype=np.float32)

            rp = np.stack([df[f"r_{d}_p{k}"].to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1)
            rm = self.r_has_hist
            self.r_probs[rm, j, :] = rp[rm]

            wp = np.stack([df[f"w_{d}_p{k}"].fillna(0.2).to_numpy(dtype=np.float32) for k in range(1, 6)], axis=-1)
            wm = self.w_has_hist
            self.w_probs[wm, j, :] = wp[wm]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "id": self.ids[idx],
            "text": self.texts[idx],
            "category": self.categories[idx],
            "document": self.documents[idx],
            "struct_feats": self.struct_feats[idx],
            "w_has_hist": self.w_has_hist[idx],
            "r_has_hist": self.r_has_hist[idx],
            "w_probs": self.w_probs[idx],
            "r_probs": self.r_probs[idx],
            "w_mean": self.w_mean[idx],
            "w_std": self.w_std[idx],
            "r_mean": self.r_mean[idx],
            "r_std": self.r_std[idx],
        }


def make_collate_fn(tokenizer, max_length: int = 128):
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        enc = tokenizer(
            [x["text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "struct_feats": torch.from_numpy(np.stack([x["struct_feats"] for x in batch])),
            "ids": [x["id"] for x in batch],
            "categories": [x["category"] for x in batch],
            "documents": [x["document"] for x in batch],
            "w_has_hist": torch.tensor([bool(x["w_has_hist"]) for x in batch], dtype=torch.bool),
            "r_has_hist": torch.tensor([bool(x["r_has_hist"]) for x in batch], dtype=torch.bool),
            "w_probs": torch.from_numpy(np.stack([x["w_probs"] for x in batch])),
            "r_probs": torch.from_numpy(np.stack([x["r_probs"] for x in batch])),
            "w_mean": torch.from_numpy(np.stack([x["w_mean"] for x in batch])),
            "w_std": torch.from_numpy(np.stack([x["w_std"] for x in batch])),
            "r_mean": torch.from_numpy(np.stack([x["r_mean"] for x in batch])),
            "r_std": torch.from_numpy(np.stack([x["r_std"] for x in batch])),
        }

    return _collate
