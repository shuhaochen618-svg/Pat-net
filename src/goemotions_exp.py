"""GoEmotions Raw experiment: Ambiguity vs Pluralism disentanglement via second-order rater mixture (APM-Net)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from .paths import OUT_DIR, PROC_DIR, resolve_backbone_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GoEmotionsRawDataset(Dataset):
    """GoEmotions Raw aggregated dataset with soft label proportions and 28x28 rater co-occurrence."""

    def __init__(self, df: pd.DataFrame, cooccur: np.ndarray, emo_cols: List[str], target_mode: str = "total") -> None:
        self.texts = df["text"].astype(str).tolist()
        prefix = "p_total_" if target_mode == "total" else "p_clear_"
        self.probs = df[[f"{prefix}{c}" for c in emo_cols]].to_numpy(dtype=np.float32)
        self.cooccur = cooccur.astype(np.float32)
        self.has_unclear = df["has_any_unclear"].to_numpy(dtype=bool)
        self.unclear_frac = df["unclear_frac"].to_numpy(dtype=np.float32)
        self.disagree = df["clear_disagree_jaccard"].to_numpy(dtype=np.float32)
        self.pos_neg = df["pos_neg_conflict"].to_numpy(dtype=np.float32)
        self.entropy = df["total_bernoulli_entropy"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "text": self.texts[idx],
            "probs": torch.from_numpy(self.probs[idx]),
            "cooccur": torch.from_numpy(self.cooccur[idx]),
            "has_unclear": bool(self.has_unclear[idx]),
            "unclear_frac": float(self.unclear_frac[idx]),
            "disagree": float(self.disagree[idx]),
            "pos_neg": float(self.pos_neg[idx]),
            "entropy": float(self.entropy[idx]),
        }


def make_ge_collate(tokenizer, max_length: int = 96):
    def collate(items: List[Dict[str, object]]) -> Dict[str, object]:
        tok = tokenizer(
            [x["text"] for x in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "probs": torch.stack([x["probs"] for x in items], dim=0),
            "cooccur": torch.stack([x["cooccur"] for x in items], dim=0),
            "has_unclear": torch.tensor([bool(x["has_unclear"]) for x in items], dtype=torch.bool),
            "unclear_frac": torch.tensor([x["unclear_frac"] for x in items], dtype=torch.float32),
            "disagree": torch.tensor([x["disagree"] for x in items], dtype=torch.float32),
            "pos_neg": torch.tensor([x["pos_neg"] for x in items], dtype=torch.float32),
            "entropy": torch.tensor([x["entropy"] for x in items], dtype=torch.float32),
        }

    return collate


class AttentionPooling(nn.Module):
    """Single-query masked additive attention pooling over token states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden_states).squeeze(-1)
        mask = attention_mask.bool()
        logits = logits.masked_fill(~mask, -1e4 if logits.dtype == torch.float16 else -1e9)
        attn = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (hidden_states * attn).sum(dim=1)


class AmbiguityPluralismModel(nn.Module):
    """Multi-label perspectivist model supporting flat BCE, MC-Dropout, and APM-Net (rank-2 co-occurrence)."""

    def __init__(
        self,
        backbone: str = "roberta-base",
        model_type: str = "apm_net",
        num_classes: int = 28,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(resolve_backbone_path(backbone))
        hidden = self.encoder.config.hidden_size
        self.pool = AttentionPooling(hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if model_type in ("flat_bce", "mc_dropout_bce"):
            self.head = nn.Linear(latent_dim, num_classes)
        elif model_type in ("apm_net", "apm_no_cooccur"):
            # Two sub-population prototypes constrained to >=1 expected vote when clear + clarity gate
            self.proto1 = nn.Linear(latent_dim, num_classes)
            self.proto2 = nn.Linear(latent_dim, num_classes)
            self.extra1 = nn.Linear(latent_dim, num_classes)
            self.extra2 = nn.Linear(latent_dim, num_classes)
            nn.init.constant_(self.extra1.bias, -4.5)
            nn.init.constant_(self.extra2.bias, -4.5)
            self.pi_head = nn.Linear(latent_dim, 1)
            self.clarity_head = nn.Linear(latent_dim, 1)
            nn.init.constant_(self.clarity_head.bias, 1.5)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_t = enc.last_hidden_state[:, 0]
        att_t = self.pool(enc.last_hidden_state, attention_mask)
        h = self.proj(self.drop(torch.cat([cls_t, att_t], dim=-1)))

        if self.model_type in ("flat_bce", "mc_dropout_bce"):
            probs = torch.sigmoid(self.head(h))
            p_clip = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            ent = (-p_clip * torch.log2(p_clip) - (1.0 - p_clip) * torch.log2(1.0 - p_clip)).sum(dim=-1)
            return {
                "probs": probs,
                "u_amb": ent,
                "u_plur": ent,
                "total_entropy": ent,
            }

        pi = torch.sigmoid(self.pi_head(h)).squeeze(-1)  # (B,)
        clarity = torch.sigmoid(self.clarity_head(h)).squeeze(-1)  # (B,)

        # Every clear rater selects >=1 emotion: q_k = softmax(z_k) + (1 - softmax(z_k)) * sigmoid(extra_k)
        # guarantees q_k in (0, 1)^C and sum_j q_{k,j} >= 1.0 identically, breaking ambiguity/pluralism symmetry.
        s1 = F.softmax(self.proto1(h), dim=-1)
        s2 = F.softmax(self.proto2(h), dim=-1)
        e1 = torch.sigmoid(self.extra1(h))
        e2 = torch.sigmoid(self.extra2(h))
        q1 = s1 + (1.0 - s1) * e1
        q2 = s2 + (1.0 - s2) * e2
        p_clear = (1.0 - pi).unsqueeze(-1) * q1 + pi.unsqueeze(-1) * q2
        probs = clarity.unsqueeze(-1) * p_clear

        # Predicted rank-2 co-occurrence matrix: c_clear * ((1-pi) q1 q1^T + pi q2 q2^T)
        co_pred = clarity.view(-1, 1, 1) * (
            (1.0 - pi).view(-1, 1, 1) * torch.einsum("bi,bj->bij", q1, q1)
            + pi.view(-1, 1, 1) * torch.einsum("bi,bj->bij", q2, q2)
        )

        # Pure ambiguity score = sub-unit unclear/abstention deficit (1 - clarity)
        u_amb = 1.0 - clarity
        # Pure pluralism score = clear-weighted between-faction disagreement variance
        u_plur = clarity * pi * (1.0 - pi) * (q1 - q2).pow(2).sum(dim=-1)

        p_clip = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        ent = (-p_clip * torch.log2(p_clip) - (1.0 - p_clip) * torch.log2(1.0 - p_clip)).sum(dim=-1)
        extra_reg = 0.02 * (e1.pow(2).sum(dim=-1) + e2.pow(2).sum(dim=-1)).mean()
        return {
            "probs": probs,
            "co_pred": co_pred,
            "q1": q1,
            "q2": q2,
            "pi": pi,
            "clarity": clarity,
            "extra_reg": extra_reg,
            "u_amb": u_amb,
            "u_plur": u_plur,
            "total_entropy": ent,
        }


def apm_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], w_co: float = 1.0) -> torch.Tensor:
    probs = torch.clamp(out["probs"], 1e-6, 1.0 - 1e-6)
    tgt = batch["probs"]
    bce = -(tgt * torch.log(probs) + (1.0 - tgt) * torch.log(1.0 - probs)).mean()
    # Anchor clarity gate against sub-unit total vote mass deficit
    mass_loss = F.mse_loss(out["probs"].sum(dim=-1), tgt.sum(dim=-1))
    extra_reg = out.get("extra_reg", 0.0)
    base_loss = bce + 0.5 * mass_loss + extra_reg
    if "co_pred" not in out or w_co <= 0.0:
        return base_loss
    # Second-order off-diagonal co-occurrence + residual covariance losses (O(1) Frobenius sum scale)
    C = probs.shape[-1]
    eye = torch.eye(C, device=probs.device, dtype=torch.bool).unsqueeze(0)
    diff_co = (out["co_pred"] - batch["cooccur"]).masked_fill(eye, 0.0)
    co_loss = diff_co.pow(2).sum(dim=(-1, -2)).mean()
    cov_pred = out["co_pred"] - torch.einsum("bi,bj->bij", probs, probs)
    cov_tgt = batch["cooccur"] - torch.einsum("bi,bj->bij", tgt, tgt)
    diff_cov = (cov_pred - cov_tgt).masked_fill(eye, 0.0)
    cov_loss = diff_cov.pow(2).sum(dim=(-1, -2)).mean()
    return base_loss + w_co * (co_loss + cov_loss)


@torch.no_grad()
def evaluate_ge(
    model: AmbiguityPluralismModel,
    loader: DataLoader,
    device: torch.device,
    mc_passes: int = 8,
    save_npz_path: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    probs_l, co_l, uamb_l, uplur_l, ent_l = [], [], [], [], []
    tgt_l, co_tgt_l, unc_l, dis_l, pn_l = [], [], [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        if model.model_type == "mc_dropout_bce":
            model.train()  # enable dropout for MC passes
            mc_p = []
            for _ in range(mc_passes):
                with autocast(enabled=(device.type == "cuda")):
                    mc_p.append(model(ids, mask)["probs"].float())
            mc_stack = torch.stack(mc_p, dim=0)  # (T, B, C)
            mean_p = mc_stack.mean(dim=0)
            p_clip = torch.clamp(mean_p, 1e-6, 1.0 - 1e-6)
            tot_h = (-p_clip * torch.log2(p_clip) - (1.0 - p_clip) * torch.log2(1.0 - p_clip)).sum(dim=-1)
            mc_clip = torch.clamp(mc_stack, 1e-6, 1.0 - 1e-6)
            exp_h = (-mc_clip * torch.log2(mc_clip) - (1.0 - mc_clip) * torch.log2(1.0 - mc_clip)).sum(dim=-1).mean(dim=0)
            mi = torch.clamp(tot_h - exp_h, min=0.0)
            out = {"probs": mean_p, "u_amb": exp_h, "u_plur": mi, "total_entropy": tot_h}
            model.eval()
        else:
            with autocast(enabled=(device.type == "cuda")):
                out = model(ids, mask)

        probs_l.append(out["probs"].float().cpu().numpy())
        uamb_l.append(out["u_amb"].float().cpu().numpy())
        uplur_l.append(out["u_plur"].float().cpu().numpy())
        ent_l.append(out["total_entropy"].float().cpu().numpy())
        if "co_pred" in out:
            co_l.append(out["co_pred"].float().cpu().numpy())
        else:
            p = out["probs"].float()
            co_l.append(torch.einsum("bi,bj->bij", p, p).cpu().numpy())

        tgt_l.append(batch["probs"].numpy())
        co_tgt_l.append(batch["cooccur"].numpy())
        unc_l.append(batch["has_unclear"].numpy())
        dis_l.append(batch["disagree"].numpy())
        pn_l.append(batch["pos_neg"].numpy())

    probs = np.concatenate(probs_l, axis=0)
    tgt = np.concatenate(tgt_l, axis=0)
    co_pred = np.concatenate(co_l, axis=0)
    co_tgt = np.concatenate(co_tgt_l, axis=0)
    u_amb = np.concatenate(uamb_l, axis=0)
    u_plur = np.concatenate(uplur_l, axis=0)
    ent = np.concatenate(ent_l, axis=0)
    has_unc = np.concatenate(unc_l, axis=0).astype(bool)
    disagree = np.concatenate(dis_l, axis=0)
    pos_neg = np.concatenate(pn_l, axis=0)

    # First-order soft probability error & macro-F1 (at threshold 0.3 / soft)
    bce_val = float(-np.mean(tgt * np.log(np.clip(probs, 1e-6, 1.0)) + (1.0 - tgt) * np.log(np.clip(1.0 - probs, 1e-6, 1.0))))
    probs_mae = float(np.mean(np.abs(probs - tgt)))
    # Off-diagonal second-order co-occurrence RMSE (crucial test of pluralism capture!)
    C = probs.shape[1]
    off_mask = ~np.eye(C, dtype=bool)
    co_rmse = float(np.sqrt(np.mean((co_pred[:, off_mask] - co_tgt[:, off_mask]) ** 2)))

    # Double Dissociation Metrics:
    # Task 1: Identify Linguistic Ambiguity (has_any_unclear) — NEVER seen as direct label!
    unc_y = has_unc.astype(int)
    # Task 2: Identify Pluralistic Polarization on CLEAR texts only (disagree >= 0.5 & ~has_unc OR pos_neg_conflict & ~has_unc)
    clear_mask = ~has_unc
    plur_y = (disagree[clear_mask] >= 0.5).astype(int)
    pol_y = (pos_neg[clear_mask] > 0).astype(int)

    metrics = {
        "soft_bce": bce_val,
        "soft_mae": probs_mae,
        "cooccur_offdiag_rmse": co_rmse,
        "unclear_auroc_by_uamb": float(roc_auc_score(unc_y, u_amb)),
        "unclear_auprc_by_uamb": float(average_precision_score(unc_y, u_amb)),
        "unclear_auroc_by_uplur": float(roc_auc_score(unc_y, u_plur)),
        "unclear_auroc_by_entropy": float(roc_auc_score(unc_y, ent)),
        "clear_disagree_auroc_by_uplur": float(roc_auc_score(plur_y, u_plur[clear_mask])),
        "clear_disagree_auprc_by_uplur": float(average_precision_score(plur_y, u_plur[clear_mask])),
        "clear_disagree_auroc_by_uamb": float(roc_auc_score(plur_y, u_amb[clear_mask])),
        "clear_disagree_auroc_by_entropy": float(roc_auc_score(plur_y, ent[clear_mask])),
        "clear_posneg_auroc_by_uplur": float(roc_auc_score(pol_y, u_plur[clear_mask])),
        "clear_posneg_auroc_by_uamb": float(roc_auc_score(pol_y, u_amb[clear_mask])),
        "clear_posneg_auroc_by_entropy": float(roc_auc_score(pol_y, ent[clear_mask])),
    }
    # Double-dissociation margin: (unclear by u_amb - unclear by u_plur) + (clear_posneg by u_plur - clear_posneg by u_amb)
    metrics["dissociation_unclear_gap"] = metrics["unclear_auroc_by_uamb"] - metrics["unclear_auroc_by_uplur"]
    metrics["dissociation_posneg_gap"] = metrics["clear_posneg_auroc_by_uplur"] - metrics["clear_posneg_auroc_by_uamb"]

    # Downstream Selective Abstention @10% budget: catch unclear items WITHOUT silencing clear pos-neg pluralism!
    k_10 = max(1, int(0.10 * len(unc_y)))
    total_unc = max(1, int(unc_y.sum()))
    clear_pol_full = clear_mask & (pos_neg > 0)
    total_cpol = max(1, int(clear_pol_full.sum()))
    for name, sc in [("uamb", u_amb), ("uplur", u_plur), ("entropy", ent)]:
        top_idx = np.argsort(-sc)[:k_10]
        sel_mask = np.zeros(len(unc_y), dtype=bool)
        sel_mask[top_idx] = True
        metrics[f"abstain10_unclear_recall_{name}"] = float((sel_mask & has_unc).sum() / total_unc)
        metrics[f"abstain10_clear_pol_damage_{name}"] = float((sel_mask & clear_pol_full).sum() / total_cpol)
    if save_npz_path is not None:
        np.savez_compressed(
            save_npz_path,
            probs=probs.astype(np.float32),
            tgt=tgt.astype(np.float32),
            u_amb=u_amb.astype(np.float32),
            u_plur=u_plur.astype(np.float32),
            entropy=ent.astype(np.float32),
            has_unc=has_unc,
            disagree=disagree.astype(np.float32),
            pos_neg=pos_neg.astype(np.float32),
        )
    return metrics


def run_ge_experiment(
    model_type: str = "apm_net",
    target_mode: str = "total",
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 64,
    backbone: str = "roberta-base",
    w_co: float = 1.0,
) -> Dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(PROC_DIR / "goemotions_raw_agg.parquet")
    npz = np.load(PROC_DIR / "goemotions_cooccur_f16.npz")
    co_arr = npz["co_total"] if target_mode == "total" else npz["co_clear"]
    emo_cols = [str(c) for c in npz["emo_cols"]]

    tr_idx = (df["split_simplified"] == "train").to_numpy()
    va_idx = (df["split_simplified"] == "validation").to_numpy()
    te_idx = (df["split_simplified"] == "test").to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(resolve_backbone_path(backbone))
    collate = make_ge_collate(tokenizer, max_length=96)

    tr_loader = DataLoader(GoEmotionsRawDataset(df[tr_idx], co_arr[tr_idx], emo_cols, target_mode), batch_size=batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(GoEmotionsRawDataset(df[va_idx], co_arr[va_idx], emo_cols, target_mode), batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(GoEmotionsRawDataset(df[te_idx], co_arr[te_idx], emo_cols, target_mode), batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    model = AmbiguityPluralismModel(backbone=backbone, model_type=model_type, num_classes=len(emo_cols)).to(device)
    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    hd_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": hd_p, "lr": 3e-4}], weight_decay=0.01)
    tot_steps = max(1, len(tr_loader) * epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * tot_steps), tot_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_loss = 1e9
    best_state = None
    t0 = time.time()
    effective_wco = 0.0 if model_type == "apm_no_cooccur" else w_co

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast(enabled=(device.type == "cuda")):
                out = model(ids, mask)
                loss = apm_loss(out, tgt, w_co=effective_wco)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
            running += float(loss.detach().item())

        va_m = evaluate_ge(model, va_loader, device)
        val_obj = va_m["soft_bce"] + (0.2 * va_m["cooccur_offdiag_rmse"] if model_type == "apm_net" else 0.0)
        if val_obj < best_loss:
            best_loss = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[GoEmo {model_type} s={seed} ep={ep}] tr={running/len(tr_loader):.4f} "
            f"val_bce={va_m['soft_bce']:.4f} co_rmse={va_m['cooccur_offdiag_rmse']:.5f} "
            f"unc_uamb={va_m['unclear_auroc_by_uamb']:.4f} unc_uplur={va_m['unclear_auroc_by_uplur']:.4f} "
            f"pn_uplur={va_m['clear_posneg_auroc_by_uplur']:.4f}"
        )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    out_dir = OUT_DIR / "goemotions" / f"{target_mode}_{model_type}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    te_m = evaluate_ge(model, te_loader, device, save_npz_path=out_dir / "test_predictions.npz")
    res = {
        "model_type": model_type,
        "target_mode": target_mode,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
        "test_metrics": te_m,
    }
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("==> GOEMOTIONS TEST:", json.dumps(te_m, indent=2))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="apm_net")
    p.add_argument("--target-mode", default="total", choices=["total", "clear"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--w-co", type=float, default=1.0)
    a = p.parse_args()
    run_ge_experiment(a.model_type, a.target_mode, a.seed, a.epochs, a.batch_size, a.backbone, a.w_co)
