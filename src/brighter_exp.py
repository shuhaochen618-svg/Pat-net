"""BRIGHTER Cross-Lingual Ordinal Intensity Calibration & Extremity Risk Alerting."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal_sets import fit_and_calibrate_alert_crc
from src.feature_reader import BiPolDisentangleReader, ablation_from_name
from src.losses import (
    build_ordinal_grid,
    dml_mixture_to_dist,
    gaussian_discretized_probs,
    kl_divergence_hist,
    unimodal_softmax_probs,
    wasserstein1_from_cdfs,
)
from src.metrics import evaluate_binary_alert_metrics
from src.paths import OUT_DIR, RAW_DIR, resolve_backbone_path
from src.struct_features import STRUCT_DIM, extract_struct_matrix

ALL_EMOS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
ORD_SCORES = (0.0, 1.0, 2.0, 3.0)
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]", re.UNICODE)


def _safe_r(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return 0.0
    return float(pearsonr(y, p)[0])


def _safe_qwk(y_int: np.ndarray, p_int: np.ndarray) -> float:
    if len(y_int) < 5 or len(np.unique(y_int)) < 2:
        return 0.0
    try:
        val = float(cohen_kappa_score(y_int, p_int, weights="quadratic"))
        return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def _compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error on predicted argmax class confidence."""
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(m):
            ece += (np.sum(m) / n) * abs(float(np.mean(acc[m])) - float(np.mean(conf[m])))
    return float(ece)


def _load_brighter_split_dfs(lang: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    ldir = RAW_DIR / "brighter" / lang
    tr = pd.read_parquet(next(ldir.glob("train*.parquet")))
    dv = pd.read_parquet(next(ldir.glob("dev*.parquet")))
    te = pd.read_parquet(next(ldir.glob("test*.parquet")))
    active = [e for e in ALL_EMOS if (e in tr.columns and tr[e].notna().sum() > 0)]
    return tr, dv, te, active


def _get_brighter_4way(
    lang: str,
    cv_fold: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    tr, dv, te, active = _load_brighter_split_dfs(lang)
    if cv_fold < 0:
        # Preserve official test; carve 15% of train into dev_select and keep full official dev for calibration
        kf = KFold(n_splits=7, shuffle=True, random_state=42)
        tr_idx, sel_idx = next(kf.split(tr))
        return (
            tr.iloc[tr_idx].reset_index(drop=True),
            tr.iloc[sel_idx].reset_index(drop=True),
            dv.reset_index(drop=True),
            te.reset_index(drop=True),
            active,
        )
    # 5-fold CV over train+dev for robust validation + still evaluate on official test
    pool = pd.concat([tr, dv], axis=0, ignore_index=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(pool))
    trval_idx, cal_idx = folds[int(cv_fold) % 5]
    sub_tr_idx, sub_sel_idx = next(KFold(n_splits=7, shuffle=True, random_state=100 + int(cv_fold)).split(trval_idx))
    return (
        pool.iloc[trval_idx[sub_tr_idx]].reset_index(drop=True),
        pool.iloc[trval_idx[sub_sel_idx]].reset_index(drop=True),
        pool.iloc[cal_idx].reset_index(drop=True),
        te.reset_index(drop=True),
        active,
    )


def _tokenize_hashed(text: str, vocab_size: int = 16384, max_len: int = 128) -> List[int]:
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
    if has_cjk:
        try:
            import jieba  # type: ignore

            toks = [t.strip() for t in jieba.lcut(text) if t.strip()]
        except Exception:
            toks = list(text.strip())
    else:
        toks = _TOKEN_RE.findall(text.lower())
    ids = [1]  # BOS placeholder at 0
    for t in toks[: max_len - 2]:
        h = (zlib.crc32(t.encode("utf-8")) % (vocab_size - 3)) + 3
        ids.append(int(h))
    ids.append(2)  # EOS placeholder
    return ids


class BiGRUBipolarEncoder(nn.Module):
    """Non-PLM BiGRU backbone emitting pseudo hidden_states tuple compatible with BiPolDisentangleReader."""

    def __init__(self, vocab_size: int = 16384, emb_dim: int = 192, hidden_dim: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_proj = nn.Linear(emb_dim, hidden_dim * 2)
        self.gru1 = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_dim * 2
        self.num_hidden_layers = 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        lengths = torch.clamp(attention_mask.long().sum(dim=1).cpu(), min=1)
        x = self.emb(input_ids)
        h0 = self.emb_proj(x)
        packed1 = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        o1, _ = self.gru1(packed1)
        h1, _ = nn.utils.rnn.pad_packed_sequence(o1, batch_first=True, total_length=input_ids.shape[1])
        packed2 = nn.utils.rnn.pack_padded_sequence(h1, lengths, batch_first=True, enforce_sorted=False)
        o2, _ = self.gru2(packed2)
        h2, _ = nn.utils.rnn.pad_packed_sequence(o2, batch_first=True, total_length=input_ids.shape[1])

        class _Out:
            def __init__(self, hs) -> None:
                self.hidden_states = hs

        return _Out((h0, h1, h2))


class BrighterOrdinalDataset(Dataset):
    """Stores texts, ordinal labels {0,1,2,3}, explicit valid_mask (notna!), and structural features."""

    def __init__(self, df: pd.DataFrame, active_emos: List[str], struct_feats: np.ndarray | None = None) -> None:
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.active_emos = list(active_emos)
        raw = df[self.active_emos]
        self.valid_mask = raw.notna().to_numpy(dtype=bool)
        arr = np.clip(raw.fillna(0.0).to_numpy(dtype=np.float32), 0.0, 3.0)
        self.y_cont = arr
        self.y_int = np.round(arr).astype(np.int64)
        self.struct_feats = (
            np.asarray(struct_feats, dtype=np.float32)
            if struct_feats is not None
            else np.zeros((len(df), STRUCT_DIM), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "text": self.texts[idx],
            "y_cont": self.y_cont[idx],
            "y_int": self.y_int[idx],
            "valid_mask": self.valid_mask[idx],
            "struct_feats": self.struct_feats[idx],
        }


def make_brighter_collate_fn(tokenizer, use_bigru: bool = False, max_length: int = 128):
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        if use_bigru:
            seqs = [_tokenize_hashed(str(x["text"]), max_len=max_length) for x in batch]
            ml = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), ml), dtype=torch.long)
            mask = torch.zeros((len(seqs), ml), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, : len(s)] = 1
        else:
            enc = tokenizer(
                [str(x["text"]) for x in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "struct_feats": torch.from_numpy(np.stack([x["struct_feats"] for x in batch])),
            "y_cont": torch.from_numpy(np.stack([x["y_cont"] for x in batch])),
            "y_int": torch.from_numpy(np.stack([x["y_int"] for x in batch])),
            "valid_mask": torch.from_numpy(np.stack([x["valid_mask"] for x in batch])).bool(),
        }

    return _collate


class BrighterOrdinalModel(nn.Module):
    """Per-emotion query BiPol-Disentangle with honest Unimodal Discretized Ordinal calibration."""

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_emos: int = 5,
        model_type: str = "dml_ordinal",
        ablation_mode: str = "M4_disentangle",
        work_dim: int = 256,
        dropout: float = 0.2,
        special_token_ids: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.num_emos = num_emos
        self.model_type = model_type
        self.is_bigru = (backbone_name == "bigru_jieba") or (model_type == "bigru_bipolar")
        self.ablation_cfg = ablation_from_name(ablation_mode)
        self.special_token_ids = special_token_ids or ()

        if self.is_bigru:
            self.encoder = BiGRUBipolarEncoder(vocab_size=16384, emb_dim=192, hidden_dim=128)
            hidden_size = self.encoder.hidden_size
            n_layers = 3
        else:
            resolved = resolve_backbone_path(backbone_name)
            cfg = AutoConfig.from_pretrained(resolved)
            cfg.output_hidden_states = True
            self.encoder = AutoModel.from_pretrained(resolved, config=cfg)
            hidden_size = int(cfg.hidden_size)
            n_layers = int(getattr(cfg, "num_hidden_layers", 12)) + 1

        self.reader = BiPolDisentangleReader(
            hidden_size=hidden_size,
            n_layers=n_layers,
            work_dim=work_dim,
            ablation=self.ablation_cfg,
            dual_segment=False,
            num_query_heads=num_emos,
            use_prag_disentangle=False,
        )
        in_d = self.reader.work_dim + self.reader.d_disp + self.reader.d_vag + work_dim

        out_per_emo = 4 if model_type in ("softmax_ce", "sord", "dldl_v2") else (1 if model_type == "scalar_mse" else 2)
        self.emo_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_d, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_per_emo),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        struct_feats: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        rf = self.reader(
            enc_out.hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
            struct_feats=struct_feats,
            special_token_ids=self.special_token_ids,
        )
        shared = torch.cat([rf["z_s"], rf["z_disp"], rf["h_vag"]], dim=-1)  # (B, ds+dd+dv)
        shared_exp = shared.unsqueeze(1).expand(-1, self.num_emos, -1)
        emo_feats = rf["emo_feats"]  # (B, E, work_dim)
        h_emo = torch.cat([shared_exp, emo_feats], dim=-1)  # (B, E, in_d)
        raw = self.emo_head(h_emo).float()
        v, _ = build_ordinal_grid(ORD_SCORES, device=raw.device, dtype=torch.float32)

        if self.model_type in ("dml_ordinal", "bigru_bipolar"):
            # Unimodal Discretized Logistic (honest single-rater ordinal, bimodal=False)
            zeros = torch.zeros_like(raw[..., 0])
            out = dml_mixture_to_dist(
                raw_mu_mid=raw[..., 0],
                raw_delta=zeros,
                raw_log_s1=raw[..., 1],
                raw_log_s2=raw[..., 1],
                raw_logit_pi=zeros,
                score_values=ORD_SCORES,
                bimodal=False,
                max_scale_ratio=0.25,
            )
            out["reg_loss"] = rf["reg_loss"]
            return out

        if self.model_type == "uw_ldl":
            out = unimodal_softmax_probs(raw, score_values=ORD_SCORES)
            out["reg_loss"] = rf["reg_loss"]
            return out

        if self.model_type == "gaussian_nll":
            mu = 3.0 * torch.sigmoid(raw[..., 0])
            std = F.softplus(raw[..., 1]) + 0.05
            out = gaussian_discretized_probs(mu, std, score_values=ORD_SCORES)
            out["mu_param"] = mu
            out["std_param"] = std
            out["reg_loss"] = rf["reg_loss"]
            return out

        if self.model_type == "scalar_mse":
            mu = 3.0 * torch.sigmoid(raw[..., 0])
            zeros = torch.zeros_like(mu)
            eye = torch.clamp(torch.round(mu).long(), 0, 3)
            probs = F.one_hot(eye, num_classes=4).float()
            return {
                "probs": probs,
                "cdf_int": torch.cumsum(probs, dim=-1)[..., :-1],
                "disc_mean": mu,
                "disc_var": zeros,
                "disc_std": zeros,
                "u_amb": zeros,
                "u_plur": zeros,
                "reg_loss": rf["reg_loss"],
            }

        # softmax_ce, sord, dldl_v2
        probs = torch.softmax(raw, dim=-1)
        cdf_int = torch.cumsum(probs, dim=-1)[..., :-1]
        mean = (probs * v.view(1, 1, -1)).sum(dim=-1)
        var = torch.clamp((probs * (v.view(1, 1, -1) - mean.unsqueeze(-1)).pow(2)).sum(dim=-1), min=0.0)
        return {
            "probs": probs,
            "cdf_int": cdf_int,
            "disc_mean": mean,
            "disc_var": var,
            "disc_std": torch.sqrt(var + 1e-8),
            "u_amb": -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1),
            "u_plur": var,
            "reg_loss": rf["reg_loss"],
        }


def compute_brighter_loss(
    model_type: str,
    out: Dict[str, torch.Tensor],
    y_int: torch.Tensor,
    y_cont: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    vm = valid_mask.float()
    denom = torch.clamp(vm.sum(), min=1.0)
    y_oh = F.one_hot(torch.clamp(y_int, 0, 3), num_classes=4).float()
    v, _ = build_ordinal_grid(ORD_SCORES, device=y_int.device, dtype=torch.float32)

    if model_type == "scalar_mse":
        mse = (out["disc_mean"].float() - y_cont.float()).pow(2)
        return (mse * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    if model_type == "gaussian_nll":
        mu = out["mu_param"].float()
        std = out["std_param"].float()
        nll = torch.log(std) + 0.5 * ((y_cont.float() - mu) / torch.clamp(std, min=1e-4)).pow(2)
        ce = -torch.log(torch.gather(torch.clamp(out["probs"], min=1e-6), -1, y_int.unsqueeze(-1)).squeeze(-1))
        return ((0.5 * nll + 0.5 * ce) * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    if model_type == "softmax_ce":
        nll = -torch.log(torch.gather(torch.clamp(out["probs"], min=1e-6), -1, y_int.unsqueeze(-1)).squeeze(-1))
        return (nll * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    if model_type == "sord":
        q_sord = torch.softmax(-torch.abs(v.view(1, 1, -1) - y_cont.float().unsqueeze(-1)) / 0.5, dim=-1)
        kl = kl_divergence_hist(q_sord, out["probs"])
        return (kl * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    if model_type == "dldl_v2":
        kl = kl_divergence_hist(y_oh, out["probs"])
        l1 = F.smooth_l1_loss(out["disc_mean"].float(), y_cont.float(), beta=0.15, reduction="none")
        return ((kl + 0.5 * l1) * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    if model_type == "uw_ldl":
        q_sord = torch.softmax(-torch.abs(v.view(1, 1, -1) - y_cont.float().unsqueeze(-1)) / 0.5, dim=-1)
        kl = kl_divergence_hist(q_sord, out["probs"])
        nll = -torch.log(torch.gather(torch.clamp(out["probs"], min=1e-6), -1, y_int.unsqueeze(-1)).squeeze(-1))
        w1 = wasserstein1_from_cdfs(out["cdf_int"], y_oh)
        return ((0.5 * nll + 0.5 * kl + 0.5 * w1) * vm).sum() / denom + 0.25 * out["reg_loss"].float()

    # dml_ordinal / bigru_bipolar: exact ordinal NLL + continuous W1 + expectation L1
    nll = -torch.log(torch.gather(torch.clamp(out["probs"], min=1e-6), -1, y_int.unsqueeze(-1)).squeeze(-1))
    w1 = wasserstein1_from_cdfs(out["cdf_int"], y_oh)
    l1 = F.smooth_l1_loss(out["disc_mean"].float(), y_cont.float(), beta=0.15, reduction="none")
    return ((nll + 0.6 * w1 + 0.4 * l1) * vm).sum() / denom + 0.25 * out["reg_loss"].float()


def _temp_scale_probs(probs: np.ndarray, tau: float) -> np.ndarray:
    p = np.clip(probs, 1e-7, 1.0)
    lp = np.log(p) / max(float(tau), 0.1)
    lp = lp - np.max(lp, axis=-1, keepdims=True)
    ep = np.exp(lp)
    return ep / np.sum(ep, axis=-1, keepdims=True)


def _fit_temperature_on_sel(sel_probs: np.ndarray, sel_y: np.ndarray, sel_mask: np.ndarray) -> float:
    if sel_mask.sum() < 10:
        return 1.0
    p_flat = sel_probs[sel_mask]
    y_flat = sel_y[sel_mask]
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 2.5, 41):
        ps = _temp_scale_probs(p_flat, float(t))
        nll = float(-np.mean(np.log(np.clip(ps[np.arange(len(y_flat)), y_flat], 1e-7, 1.0))))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


@torch.no_grad()
def predict_brighter(
    model: BrighterOrdinalModel,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    probs_l, mean_l, std_l, yc_l, yi_l, vm_l = [], [], [], [], [], []
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["struct_feats"].to(device))
        probs_l.append(out["probs"].cpu().numpy())
        mean_l.append(out["disc_mean"].cpu().numpy())
        std_l.append(out["disc_std"].cpu().numpy())
        yc_l.append(batch["y_cont"].numpy())
        yi_l.append(batch["y_int"].numpy())
        vm_l.append(batch["valid_mask"].numpy())
    return {
        "probs": np.concatenate(probs_l),
        "disc_mean": np.concatenate(mean_l),
        "disc_std": np.concatenate(std_l),
        "y_cont": np.concatenate(yc_l),
        "y_int": np.concatenate(yi_l),
        "valid_mask": np.concatenate(vm_l),
    }


def evaluate_brighter(
    pred: Dict[str, np.ndarray],
    active_emos: Sequence[str],
    tau: float = 1.0,
    prefix: str = "test",
) -> Dict[str, float]:
    probs = _temp_scale_probs(pred["probs"], tau)
    v = np.asarray(ORD_SCORES, dtype=float)
    is_dirac = bool("disc_std" in pred and np.all(pred["disc_std"] < 1e-6))
    mean_pred = pred["disc_mean"] if (is_dirac or (abs(tau - 1.0) < 1e-6 and "disc_mean" in pred)) else (probs * v.reshape(1, 1, -1)).sum(axis=-1)
    y_cont = pred["y_cont"]
    y_int = pred["y_int"]
    vm = pred["valid_mask"].astype(bool)

    r_l, mae_l, qwk_l, nll_l, brier_l, ece_l, ext_auc_l = [], [], [], [], [], [], []
    for j, _ in enumerate(active_emos):
        m = vm[:, j]
        if m.sum() < 5:
            continue
        p_j = probs[m, j, :]
        yt_c = y_cont[m, j]
        yt_i = y_int[m, j]
        ym_hat = mean_pred[m, j]
        yi_hat = np.argmax(p_j, axis=-1)

        r_l.append(_safe_r(yt_c, ym_hat))
        mae_l.append(float(np.mean(np.abs(yt_c - ym_hat))))
        qwk_l.append(_safe_qwk(yt_i, yi_hat))

        y_oh = np.eye(4, dtype=float)[yt_i]
        brier_l.append(float(np.mean(np.sum((p_j - y_oh) ** 2, axis=-1))))
        p_clip = np.clip(p_j[np.arange(len(yt_i)), yt_i], 1e-7, 1.0)
        nll_l.append(float(-np.mean(np.log(p_clip))))
        ece_l.append(_compute_ece(p_j, yt_i))

        ext_lbl = (yt_i >= 2).astype(int)
        p_ext = p_j[:, 2] + p_j[:, 3]
        if len(np.unique(ext_lbl)) == 2:
            ext_auc_l.append(evaluate_binary_alert_metrics(ext_lbl, p_ext)["auroc"])

    return {
        f"{prefix}_macro_pearson": float(np.mean(r_l)) if r_l else 0.0,
        f"{prefix}_macro_mae": float(np.mean(mae_l)) if mae_l else 0.0,
        f"{prefix}_macro_qwk": float(np.mean(qwk_l)) if qwk_l else 0.0,
        f"{prefix}_macro_nll": float(np.mean(nll_l)) if nll_l else 0.0,
        f"{prefix}_macro_brier": float(np.mean(brier_l)) if brier_l else 0.0,
        f"{prefix}_macro_ece": float(np.mean(ece_l)) if ece_l else 0.0,
        f"{prefix}_extremity_auroc": float(np.mean(ext_auc_l)) if ext_auc_l else 0.5,
    }


def run_brighter_exp(
    lang: str = "chn",
    backbone_name: str = "hfl/chinese-roberta-wwm-ext",
    model_type: str = "dml_ordinal",
    ablation_mode: str = "M4_disentangle",
    cv_fold: int = -1,
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 32,
    grad_accum: int = 1,
    max_length: int = 128,
    encoder_lr: float = 2e-5,
    head_lr: float = 5e-4,
    num_workers: int = 0,
    out_dir_override: str | None = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_bigru = (backbone_name == "bigru_jieba") or (model_type == "bigru_bipolar")
    amp_enabled = (device.type == "cuda") and (not is_bigru)

    df_tr, df_sel, df_cal, df_te, active = _get_brighter_4way(lang, cv_fold=cv_fold)
    scaler = StandardScaler().fit(extract_struct_matrix(df_tr["text"].fillna("").astype(str).tolist()))
    sf_tr = scaler.transform(extract_struct_matrix(df_tr["text"].fillna("").astype(str).tolist())).astype(np.float32)
    sf_sel = scaler.transform(extract_struct_matrix(df_sel["text"].fillna("").astype(str).tolist())).astype(np.float32)
    sf_cal = scaler.transform(extract_struct_matrix(df_cal["text"].fillna("").astype(str).tolist())).astype(np.float32)
    sf_te = scaler.transform(extract_struct_matrix(df_te["text"].fillna("").astype(str).tolist())).astype(np.float32)

    tokenizer = None if is_bigru else AutoTokenizer.from_pretrained(resolve_backbone_path(backbone_name))
    special_ids = (
        tuple(
            x
            for x in (
                tokenizer.cls_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )
            if x is not None
        )
        if tokenizer is not None
        else (0, 1, 2)
    )
    collate = make_brighter_collate_fn(tokenizer, use_bigru=is_bigru, max_length=max_length)

    tr_loader = DataLoader(BrighterOrdinalDataset(df_tr, active, sf_tr), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    sel_loader = DataLoader(BrighterOrdinalDataset(df_sel, active, sf_sel), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate)
    cal_loader = DataLoader(BrighterOrdinalDataset(df_cal, active, sf_cal), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate)
    te_loader = DataLoader(BrighterOrdinalDataset(df_te, active, sf_te), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate)

    model = BrighterOrdinalModel(
        backbone_name=backbone_name,
        num_emos=len(active),
        model_type=model_type,
        ablation_mode=ablation_mode,
        special_token_ids=special_ids,
    ).to(device)

    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.") and not is_bigru:
            enc_params.append(p)
        else:
            head_params.append(p)

    param_groups = [{"params": head_params, "lr": 1e-3 if is_bigru else head_lr}]
    if enc_params:
        param_groups.insert(0, {"params": enc_params, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    steps_per_epoch = max(1, (len(tr_loader) + max(grad_accum, 1) - 1) // max(grad_accum, 1))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    fold_tag = "official" if cv_fold < 0 else f"cv{cv_fold}"
    bb_tag = backbone_name.replace("/", "__")
    ab_suf = f"_{ablation_mode}" if ((model_type in ("dml_ordinal", "bigru_bipolar") and ablation_mode != "M4_disentangle") or (model_type not in ("dml_ordinal", "bigru_bipolar") and ablation_mode != "M0_baseline")) else ""
    out_dir = Path(out_dir_override) if out_dir_override else (OUT_DIR / "brighter" / f"{lang}_{fold_tag}_{bb_tag}_{model_type}{ab_suf}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    best_score = -1e9
    best_sel: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tr_loader, start=1):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["struct_feats"].to(device))
                loss = compute_brighter_loss(
                    model_type,
                    out,
                    batch["y_int"].to(device),
                    batch["y_cont"].to(device),
                    batch["valid_mask"].to(device),
                )
                loss_scaled = loss / max(grad_accum, 1)

            grad_scaler.scale(loss_scaled).backward()
            if (step % grad_accum == 0) or (step == len(tr_loader)):
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = grad_scaler.get_scale()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler.get_scale() >= old_scale:
                    scheduler.step()

        sel_pred = predict_brighter(model, sel_loader, device, amp_enabled=amp_enabled)
        sel_m = evaluate_brighter(sel_pred, active, tau=1.0, prefix="sel")
        score = sel_m["sel_macro_pearson"] if model_type == "scalar_mse" else (sel_m["sel_macro_pearson"] + 0.3 * sel_m["sel_macro_qwk"] - 0.1 * sel_m["sel_macro_nll"])
        print(f"[BRIGHTER {lang}][{model_type}] Ep {epoch}/{epochs} sel_r={sel_m['sel_macro_pearson']:.4f} sel_qwk={sel_m['sel_macro_qwk']:.4f} sel_nll={sel_m['sel_macro_nll']:.4f}")
        if score > best_score:
            best_score = score
            best_sel = sel_m
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    sel_pred = predict_brighter(model, sel_loader, device, amp_enabled=amp_enabled)
    cal_pred = predict_brighter(model, cal_loader, device, amp_enabled=amp_enabled)
    te_pred = predict_brighter(model, te_loader, device, amp_enabled=amp_enabled)

    # Fit scalar probability temperature on dev_select (only for probabilistic heads, not Dirac scalar_mse)
    tau = 1.0 if model_type == "scalar_mse" else _fit_temperature_on_sel(sel_pred["probs"], sel_pred["y_int"], sel_pred["valid_mask"])
    cal_metrics = evaluate_brighter(cal_pred, active, tau=tau, prefix="cal")
    te_metrics = evaluate_brighter(te_pred, active, tau=tau, prefix="test")

    # Conformal Risk Control (CRC) on any high-intensity distress extremity (max emotion intensity >= 2)
    def _max_ext_feats(p: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if np.all(p["disc_std"] < 1e-6):
            score_mat = np.where(p["valid_mask"], np.clip(p["disc_mean"] / 3.0, 0.0, 1.0), 0.0)
        else:
            probs_s = _temp_scale_probs(p["probs"], tau)
            score_mat = np.where(p["valid_mask"], probs_s[..., 2] + probs_s[..., 3], 0.0)
        y_vals = np.where(p["valid_mask"], p["y_int"], 0)
        y = (np.max(y_vals, axis=-1) >= 2).astype(int)
        return np.max(score_mat, axis=-1), y

    sel_X, sel_y = _max_ext_feats(sel_pred)
    cal_X, cal_y = _max_ext_feats(cal_pred)
    te_X, te_y = _max_ext_feats(te_pred)
    crc_ext_10 = fit_and_calibrate_alert_crc(sel_X, sel_y, cal_X, cal_y, te_X, te_y, alpha=0.10, use_supervised_combiner=False)

    np.savez_compressed(out_dir / "cal_predictions.npz", **cal_pred)
    np.savez_compressed(out_dir / "test_predictions.npz", **te_pred)
    summary = {
        "task": "brighter_ordinal_calibration",
        "lang": lang,
        "active_emos": active,
        "backbone": backbone_name,
        "model_type": model_type,
        "ablation_mode": ablation_mode,
        "cv_fold": cv_fold,
        "seed": seed,
        "calibrated_temp": round(tau, 4),
        "best_sel": best_sel,
        "cal": cal_metrics,
        "test": te_metrics,
        "crc_extremity_alpha10": crc_ext_10,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="chn")
    parser.add_argument("--backbone", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--model-type", type=str, default="dml_ordinal")
    parser.add_argument("--ablation-mode", type=str, default="M4_disentangle")
    parser.add_argument("--cv-fold", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run_brighter_exp(
        lang=args.lang,
        backbone_name=args.backbone,
        model_type=args.model_type,
        ablation_mode=args.ablation_mode,
        cv_fold=args.cv_fold,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        num_workers=args.num_workers,
        out_dir_override=args.out_dir,
    )


if __name__ == "__main__":
    main()
