"""EPIC Perspectivist Irony Experiment: Dual-Segment BiPol-Disentangle + 2-Component Beta-Binomial Transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal_sets import fit_and_calibrate_alert_crc
from src.feature_reader import BiPolDisentangleReader, ablation_from_name
from src.losses import _log_beta_fn, beta_binomial_mixture_nll
from src.metrics import evaluate_binary_alert_metrics
from src.paths import OUT_DIR, PROC_DIR, resolve_backbone_path
from src.struct_features import STRUCT_DIM, extract_struct_matrix

NAT_CODES = ("uk", "us", "au", "ie", "in")


def _safe_r(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return 0.0
    return float(pearsonr(y, p)[0])


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if (len(np.unique(y)) == 2 and np.std(s) > 1e-12) else 0.5


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if (len(np.unique(y)) == 2 and np.std(s) > 1e-12) else float(np.mean(y))


def get_epic_4way_splits(
    df: pd.DataFrame,
    cv_fold: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return strict parent_text-disjoint (train, dev_select, dev_cal, test) DataFrames."""
    if cv_fold < 0 and "split" in df.columns and {"dev_select", "dev_cal"}.issubset(set(df["split"].unique())):
        return (
            df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "dev_select"].reset_index(drop=True),
            df[df["split"] == "dev_cal"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True),
        )
    fold = max(0, int(cv_fold)) % 5
    te_mask = df["cv_fold"] == fold
    df_te = df[te_mask].reset_index(drop=True)
    df_trval = df[~te_mask].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=100 + fold)
    strat_key = df_trval["language_variety"].astype(str) + "_" + df_trval["is_polarized"].astype(int).astype(str)
    if int(strat_key.value_counts().min()) < 8:
        strat_key = df_trval["language_variety"].astype(str)
    sub_folds = list(sgkf.split(df_trval, strat_key, df_trval["group_key"]))
    sel_idx = sub_folds[0][1]
    cal_idx = sub_folds[1][1]
    used = np.zeros(len(df_trval), dtype=bool)
    used[sel_idx] = True
    used[cal_idx] = True
    return (
        df_trval.iloc[~used].reset_index(drop=True),
        df_trval.iloc[sel_idx].reset_index(drop=True),
        df_trval.iloc[cal_idx].reset_index(drop=True),
        df_te,
    )


class EpicDataset(Dataset):
    """Stores parent/reply pairs, full overall + Sex + Nationality Binomial counts, and structural cues."""

    def __init__(self, df: pd.DataFrame, struct_feats: np.ndarray | None = None) -> None:
        self.parents = df["parent_text"].fillna("").astype(str).tolist()
        self.replies = df["text"].fillna("").astype(str).tolist()
        self.n_annot = df["n_annot"].to_numpy(dtype=np.float32)
        self.k_iro = df["k_iro"].to_numpy(dtype=np.float32)
        self.p_iro = df["p_iro"].to_numpy(dtype=np.float32)

        self.n_male = df["n_male"].to_numpy(dtype=np.float32)
        self.k_male = df["k_male"].to_numpy(dtype=np.float32)
        self.p_male = df["p_iro_male"].fillna(-1.0).to_numpy(dtype=np.float32)

        self.n_female = df["n_female"].to_numpy(dtype=np.float32)
        self.k_female = df["k_female"].to_numpy(dtype=np.float32)
        self.p_female = df["p_iro_female"].fillna(-1.0).to_numpy(dtype=np.float32)

        self.n_nat = np.stack([df[f"n_nat_{c}"].to_numpy(dtype=np.float32) for c in NAT_CODES], axis=-1)
        self.k_nat = np.stack([df[f"k_nat_{c}"].to_numpy(dtype=np.float32) for c in NAT_CODES], axis=-1)
        self.p_nat = np.stack([df[f"p_nat_{c}"].fillna(-1.0).to_numpy(dtype=np.float32) for c in NAT_CODES], axis=-1)

        self.is_pol = df["is_polarized"].to_numpy(dtype=bool)
        self.is_una = df["is_unanimous"].to_numpy(dtype=bool)
        self.group_keys = df["group_key"].fillna("").astype(str).tolist()
        self.struct_feats = (
            np.asarray(struct_feats, dtype=np.float32)
            if struct_feats is not None
            else np.zeros((len(df), STRUCT_DIM), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.replies)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "parent": self.parents[idx],
            "reply": self.replies[idx],
            "n_annot": self.n_annot[idx],
            "k_iro": self.k_iro[idx],
            "p_iro": self.p_iro[idx],
            "n_male": self.n_male[idx],
            "k_male": self.k_male[idx],
            "p_male": self.p_male[idx],
            "n_female": self.n_female[idx],
            "k_female": self.k_female[idx],
            "p_female": self.p_female[idx],
            "n_nat": self.n_nat[idx],
            "k_nat": self.k_nat[idx],
            "p_nat": self.p_nat[idx],
            "is_pol": self.is_pol[idx],
            "is_una": self.is_una[idx],
            "group_key": self.group_keys[idx],
            "struct_feats": self.struct_feats[idx],
        }


def make_epic_collate_fn(tokenizer, max_length: int = 160):
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        parents = [x["parent"] if x["parent"] else "(no context)" for x in batch]
        replies = [x["reply"] for x in batch]
        enc = tokenizer(
            parents,
            replies,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        B, L = enc["input_ids"].shape
        seg_ids = torch.full((B, L), -1, dtype=torch.long)
        if hasattr(enc, "sequence_ids"):
            for i in range(B):
                sids = enc.sequence_ids(i)
                if sids is not None:
                    for t, sv in enumerate(sids):
                        if sv in (0, 1):
                            seg_ids[i, t] = int(sv)
        elif sep_id is not None:
            ids = enc["input_ids"]
            for i in range(B):
                seps = (ids[i] == int(sep_id)).nonzero(as_tuple=False).flatten()
                if len(seps) >= 2:
                    seg_ids[i, 1 : int(seps[0].item())] = 0
                    seg_ids[i, int(seps[-2].item()) + 1 : int(seps[-1].item())] = 1

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "seg_ids": seg_ids,
            "struct_feats": torch.from_numpy(np.stack([x["struct_feats"] for x in batch])),
            "n_annot": torch.tensor([float(x["n_annot"]) for x in batch], dtype=torch.float32),
            "k_iro": torch.tensor([float(x["k_iro"]) for x in batch], dtype=torch.float32),
            "p_iro": torch.tensor([float(x["p_iro"]) for x in batch], dtype=torch.float32),
            "n_male": torch.tensor([float(x["n_male"]) for x in batch], dtype=torch.float32),
            "k_male": torch.tensor([float(x["k_male"]) for x in batch], dtype=torch.float32),
            "p_male": torch.tensor([float(x["p_male"]) for x in batch], dtype=torch.float32),
            "n_female": torch.tensor([float(x["n_female"]) for x in batch], dtype=torch.float32),
            "k_female": torch.tensor([float(x["k_female"]) for x in batch], dtype=torch.float32),
            "p_female": torch.tensor([float(x["p_female"]) for x in batch], dtype=torch.float32),
            "n_nat": torch.from_numpy(np.stack([x["n_nat"] for x in batch])),
            "k_nat": torch.from_numpy(np.stack([x["k_nat"] for x in batch])),
            "p_nat": torch.from_numpy(np.stack([x["p_nat"] for x in batch])),
            "is_pol": np.array([x["is_pol"] for x in batch], dtype=bool),
            "is_una": np.array([x["is_una"] for x in batch], dtype=bool),
            "group_keys": [x["group_key"] for x in batch],
        }

    return _collate


def _single_bb_nll(k: torch.Tensor, n: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    kf = torch.clamp(k.float(), min=0.0)
    nf = torch.clamp(n.float(), min=1.0)
    kf = torch.minimum(kf, nf)
    mf = torch.clamp(mu.float(), min=1e-4, max=1.0 - 1e-4)
    kap = torch.clamp(kappa.float(), min=1.0, max=200.0)
    a = mf * kap
    b = (1.0 - mf) * kap
    log_comb = torch.lgamma(nf + 1.0) - torch.lgamma(kf + 1.0) - torch.lgamma(nf - kf + 1.0)
    return -(log_comb + _log_beta_fn(a + kf, b + nf - kf) - _log_beta_fn(a, b))


class EpicPerspectivistModel(nn.Module):
    """Dual-segment BiPol-Disentangle with 2-Component Beta-Binomial Mixture + Conjugacy Transport."""

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        model_type: str = "patnet",
        ablation_mode: str = "M4_disentangle",
        work_dim: int = 256,
        dropout: float = 0.2,
        special_token_ids: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.ablation_cfg = ablation_from_name(ablation_mode)
        self.special_token_ids = special_token_ids or ()
        resolved = resolve_backbone_path(backbone_name)

        cfg = AutoConfig.from_pretrained(resolved)
        cfg.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(resolved, config=cfg)
        self.hidden_size = int(cfg.hidden_size)
        n_layers = int(getattr(cfg, "num_hidden_layers", 12)) + 1

        self.reader = BiPolDisentangleReader(
            hidden_size=self.hidden_size,
            n_layers=n_layers,
            work_dim=work_dim,
            ablation=self.ablation_cfg,
            dual_segment=True,
            num_query_heads=1,
        )
        d_s = self.reader.work_dim
        d_disp = self.reader.d_disp
        d_vag = self.reader.d_vag
        d_p = self.reader.prag_dim

        base_in = d_s + d_disp + d_vag
        full_in = base_in + d_p
        trans_in = d_s + d_p + d_disp

        def _mlp(in_d: int, out_d: int, mid_d: int = 128) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_d, mid_d),
                nn.LayerNorm(mid_d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_d, out_d),
            )

        if model_type == "scalar_bce":
            self.overall_head = _mlp(base_in, 1)
        elif model_type == "beta_binom":
            # Single unimodal Beta-Binomial without subgroup transport
            self.overall_head = _mlp(base_in, 2)
        elif model_type == "patnet_no_transport":
            # Parameter-matched capacity with overall_mu_head isolated on z_s and detached split_head
            self.overall_mu_head = _mlp(d_s, 2)
            self.overall_split_head = _mlp(base_in, 4)
            self.male_head = _mlp(full_in, 2)
            self.female_head = _mlp(full_in, 2)
            self.nat_head = _mlp(full_in, len(NAT_CODES) * 2)
        elif model_type in ("patnet", "sex_nat_transport_beta_binom"):
            # Unimodal overall mean head on z_s + gradient-isolated split head + bounded conjugacy transport
            self.overall_mu_head = _mlp(d_s, 2)
            self.overall_split_head = _mlp(base_in, 4)
            self.prag_gate = nn.Sequential(nn.Linear(d_p, 1 + len(NAT_CODES)), nn.Sigmoid())
            self.sex_trans_head = _mlp(trans_in, 3)  # (delta_logit_sex, raw_kap_m, raw_kap_f)
            self.nat_trans_head = _mlp(trans_in, len(NAT_CODES) * 2)  # (delta_logit_c, raw_kap_c)
            with torch.no_grad():
                self.sex_trans_head[-1].weight.mul_(0.25)
                self.sex_trans_head[-1].bias.zero_()
                self.nat_trans_head[-1].weight.mul_(0.25)
                self.nat_trans_head[-1].bias.zero_()
        else:
            raise ValueError(f"Unknown EPIC model_type: {model_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seg_ids: torch.Tensor,
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
            seg_ids=seg_ids,
            struct_feats=struct_feats,
            special_token_ids=self.special_token_ids,
        )
        z_s, z_disp, h_vag, z_p = rf["z_s"], rf["z_disp"], rf["h_vag"], rf["z_p"]
        base_feat = torch.cat([z_s, z_disp, h_vag], dim=-1)

        if self.model_type == "scalar_bce":
            logit = self.overall_head(base_feat).float().squeeze(-1)
            p_hat = torch.sigmoid(logit)
            u_tot = p_hat * (1.0 - p_hat)
            return {
                "p_overall": p_hat,
                "p_male": p_hat,
                "p_female": p_hat,
                "p_nat": p_hat.unsqueeze(-1).expand(-1, len(NAT_CODES)),
                "kappa_m": torch.full_like(p_hat, 10.0),
                "kappa_f": torch.full_like(p_hat, 10.0),
                "kappa_nat": torch.full((len(p_hat), len(NAT_CODES)), 10.0, device=p_hat.device),
                "mu1": p_hat,
                "mu2": p_hat,
                "pi": torch.full_like(p_hat, 0.5),
                "a1": p_hat * 10.0,
                "b1": (1.0 - p_hat) * 10.0,
                "a2": p_hat * 10.0,
                "b2": (1.0 - p_hat) * 10.0,
                "u_plur": torch.zeros_like(p_hat),
                "u_amb": u_tot,
                "transport_l1": p_hat.new_zeros(()),
                "reg_loss": rf["reg_loss"],
            }

        if self.model_type == "beta_binom":
            raw = self.overall_head(base_feat).float()
            p_hat = torch.sigmoid(raw[:, 0])
            kappa = 4.0 + F.softplus(raw[:, 1])
            return {
                "p_overall": p_hat,
                "p_male": p_hat,
                "p_female": p_hat,
                "p_nat": p_hat.unsqueeze(-1).expand(-1, len(NAT_CODES)),
                "kappa_m": kappa,
                "kappa_f": kappa,
                "kappa_nat": kappa.unsqueeze(-1).expand(-1, len(NAT_CODES)),
                "mu1": p_hat,
                "mu2": p_hat,
                "pi": torch.full_like(p_hat, 0.5),
                "a1": p_hat * kappa,
                "b1": (1.0 - p_hat) * kappa,
                "a2": p_hat * kappa,
                "b2": (1.0 - p_hat) * kappa,
                "u_plur": torch.zeros_like(p_hat),
                "u_amb": p_hat * (1.0 - p_hat) / (kappa + 1.0),
                "transport_l1": p_hat.new_zeros(()),
                "reg_loss": rf["reg_loss"],
            }

        # Overall mean (p_overall, kappa_ov) read purely from z_s; split/dispersion firewall detaches p_overall & z_s
        mu_raw = self.overall_mu_head(z_s).float()
        p_overall = torch.sigmoid(mu_raw[:, 0])
        kappa_ov = 4.0 + F.softplus(mu_raw[:, 1])

        split_in = torch.cat([z_s.detach(), z_disp, h_vag], dim=-1)
        sp = self.overall_split_head(split_in).float()
        p_base = p_overall.detach()
        pi = torch.sigmoid(sp[:, 3])
        max_delta = torch.minimum(
            p_base / torch.clamp(pi, min=1e-3),
            (1.0 - p_base) / torch.clamp(1.0 - pi, min=1e-3),
        )
        delta_p = max_delta * torch.sigmoid(sp[:, 0])
        mu1 = torch.clamp(p_base - pi * delta_p, min=1e-4, max=1.0 - 1e-4)
        mu2 = torch.clamp(p_base + (1.0 - pi) * delta_p, min=1e-4, max=1.0 - 1e-4)
        kappa1 = 4.0 + F.softplus(sp[:, 1])
        kappa2 = 4.0 + F.softplus(sp[:, 2])
        a1, b1 = mu1 * kappa1, (1.0 - mu1) * kappa1
        a2, b2 = mu2 * kappa2, (1.0 - mu2) * kappa2

        u_plur = pi * (1.0 - pi) * (mu2 - mu1).pow(2)
        u_amb = (1.0 - pi) * (mu1 * (1.0 - mu1) / (kappa1 + 1.0)) + pi * (mu2 * (1.0 - mu2) / (kappa2 + 1.0))

        z_s_iso = z_s.detach() if self.ablation_cfg.use_disentangle else z_s
        z_d_iso = z_disp.detach() if self.ablation_cfg.use_disentangle else z_disp
        h_v_iso = h_vag.detach() if self.ablation_cfg.use_disentangle else h_vag
        z_p_tr = (0.15 * z_p + 0.85 * z_p.detach()) if self.ablation_cfg.use_disentangle else z_p

        if self.model_type == "patnet_no_transport":
            full_feat = torch.cat([z_s_iso, z_p_tr, z_d_iso, h_v_iso], dim=-1)
            rm = self.male_head(full_feat).float()
            rf_h = self.female_head(full_feat).float()
            rn = self.nat_head(full_feat).float().view(-1, len(NAT_CODES), 2)
            p_male = torch.sigmoid(rm[:, 0])
            p_female = torch.sigmoid(rf_h[:, 0])
            kappa_m = 4.0 + F.softplus(rm[:, 1])
            kappa_f = 4.0 + F.softplus(rf_h[:, 1])
            p_nat = torch.sigmoid(rn[..., 0])
            kappa_nat = 4.0 + F.softplus(rn[..., 1])
            transport_l1 = p_overall.new_zeros(())
        else:
            # Parameter-space Logistic/Log-Odds Conjugacy Transport conditioned on pragmatic z_p
            p_safe = torch.clamp(p_overall, min=1e-4, max=1.0 - 1e-4)
            logit_base = (torch.log(p_safe) - torch.log(1.0 - p_safe)).detach()
            gates = self.prag_gate(z_p_tr).float()  # (B, 1 + 5)
            t_in = torch.cat([z_s_iso, z_p_tr, z_d_iso], dim=-1)

            st = self.sex_trans_head(t_in).float()
            delta_sex = (0.5 + gates[:, 0]) * (1.8 * torch.tanh(st[:, 0] / 2.0))
            p_male = torch.sigmoid(logit_base - 0.5 * delta_sex)
            p_female = torch.sigmoid(logit_base + 0.5 * delta_sex)
            kappa_m = 4.0 + F.softplus(st[:, 1])
            kappa_f = 4.0 + F.softplus(st[:, 2])

            nt = self.nat_trans_head(t_in).float().view(-1, len(NAT_CODES), 2)
            delta_nat_raw = (0.5 + gates[:, 1:]) * (1.8 * torch.tanh(nt[..., 0] / 2.0))
            delta_nat = delta_nat_raw - delta_nat_raw.mean(dim=-1, keepdim=True)
            p_nat = torch.sigmoid(logit_base.unsqueeze(-1) + delta_nat)
            kappa_nat = 4.0 + F.softplus(nt[..., 1])

            transport_l1 = delta_sex.abs().mean() + delta_nat.abs().mean()

        return {
            "p_overall": p_overall,
            "kappa_ov": kappa_ov,
            "p_male": p_male,
            "p_female": p_female,
            "p_nat": p_nat,
            "kappa_m": kappa_m,
            "kappa_f": kappa_f,
            "kappa_nat": kappa_nat,
            "mu1": mu1,
            "mu2": mu2,
            "pi": pi,
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "u_plur": u_plur,
            "u_amb": u_amb,
            "transport_l1": transport_l1,
            "reg_loss": rf["reg_loss"],
        }


def compute_epic_loss(
    model_type: str,
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    k = batch["k_iro"].float()
    n = batch["n_annot"].float()
    p_obs = k / torch.clamp(n, min=1.0)
    B = max(len(k), 1)

    if model_type == "scalar_bce":
        p = torch.clamp(out["p_overall"].float(), 1e-5, 1.0 - 1e-5)
        bce = -(k * torch.log(p) + (n - k) * torch.log(1.0 - p))
        loss = (bce / torch.clamp(n, min=1.0)).mean() + out["reg_loss"].float()
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "beta_binom":
        bb_ov = _single_bb_nll(k, n, out["p_overall"], out["kappa_m"])
        mom = F.smooth_l1_loss(out["p_overall"].float(), p_obs, beta=0.1)
        loss = (bb_ov / torch.clamp(n, min=1.0)).mean() + mom + out["reg_loss"].float()
        return loss, {"loss": float(loss.detach().item())}

    # Unimodal Beta-Binomial overall likelihood (immune to polarized k=n/2 mixture repulsion)
    bb_ov = _single_bb_nll(k, n, out["p_overall"], out["kappa_ov"])
    # Weight normalized per item to stay balanced across n in [1..8]
    loss_ov = (bb_ov / torch.clamp(n, min=1.0)).mean()
    p_ov_f = torch.clamp(out["p_overall"].float(), 1e-4, 1.0 - 1e-4)
    bce_ov = -(p_obs * torch.log(p_ov_f) + (1.0 - p_obs) * torch.log(1.0 - p_ov_f)).mean()
    loss_mom = 1.5 * F.smooth_l1_loss(p_ov_f, p_obs, beta=0.1) + 0.25 * bce_ov

    # Target-gated conditional parsimony for camp identifiability (strictly isolated from p_overall and z_s)
    disp_obs = p_obs * (1.0 - p_obs)
    min_camp = torch.minimum(k, torch.clamp(n - k, min=0.0))
    w_pol = (torch.sigmoid((disp_obs - 0.19) / 0.03) * (n >= 4).float() * (min_camp >= 2.0).float()).detach()
    w_pol_eff = w_pol / torch.clamp(w_pol.mean(), min=0.05)
    una_mask_f = (n >= 4).float() * (1.0 - w_pol)
    loss_pars = (
        0.25 * una_mask_f * F.smooth_l1_loss(out["u_plur"].float(), torch.zeros_like(out["u_plur"]), beta=0.04, reduction="none")
        + 0.12 * w_pol_eff * torch.clamp(0.40 - (out["mu2"] - out["mu1"]), min=0.0).pow(2)
    ).mean()
    pol_m = w_pol >= 0.5
    una_m = (n >= 4) & (min_camp < 0.5)
    if torch.any(pol_m) and torch.any(una_m):
        lu = torch.log(torch.clamp(out["u_plur"].float(), min=0.0) + 0.005)
        loss_pars = loss_pars + 0.18 * F.softplus(1.0 - (lu[pol_m].unsqueeze(1) - lu[una_m].unsqueeze(0))).mean()

    # Sex-conditioned Beta-Binomial likelihoods (masked by n_male>0, n_female>0)
    nm, km = batch["n_male"].float(), batch["k_male"].float()
    nf, kf = batch["n_female"].float(), batch["k_female"].float()
    mm = (nm > 0).float()
    mf = (nf > 0).float()
    bb_m = _single_bb_nll(km, torch.clamp(nm, min=1.0), out["p_male"], out["kappa_m"]) / torch.clamp(nm, min=1.0)
    bb_f = _single_bb_nll(kf, torch.clamp(nf, min=1.0), out["p_female"], out["kappa_f"]) / torch.clamp(nf, min=1.0)
    loss_sex_bb = (bb_m * mm).sum() / max(float(mm.sum().item()), 1.0) + (bb_f * mf).sum() / max(float(mf.sum().item()), 1.0)

    # Jeffreys-smoothed inverse-variance weighted Sex shift supervision (all pairs, prioritizing ge2)
    both = (nm >= 1.0) & (nf >= 1.0)
    if torch.any(both):
        pm_j = (km + 0.5) / (nm + 1.0)
        pf_j = (kf + 0.5) / (nf + 1.0)
        var_j = (pm_j * (1.0 - pm_j)) / (nm + 1.0) + (pf_j * (1.0 - pf_j)) / (nf + 1.0)
        ge2_boost = torch.where((nm >= 2.0) & (nf >= 2.0), 1.5, 0.35)
        iv_w = torch.clamp(0.08 / torch.clamp(var_j, min=0.02), min=0.15, max=2.5) * both.float() * ge2_boost
        shift_true = pf_j - pm_j
        shift_pred = out["p_female"].float() - out["p_male"].float()
        loss_shift_sex = (F.smooth_l1_loss(shift_pred, shift_true, beta=0.1, reduction="none") * iv_w).sum() / torch.clamp(iv_w.sum(), min=1.0)
        loss_div_mag = (F.smooth_l1_loss(shift_pred.abs(), shift_true.abs(), beta=0.1, reduction="none") * iv_w).sum() / torch.clamp(iv_w.sum(), min=1.0)
    else:
        loss_shift_sex = loss_ov.new_zeros(())
        loss_div_mag = loss_ov.new_zeros(())

    # Nationality-conditioned Binomial / Beta-Binomial supervision across all 5 varieties
    n_nat = batch["n_nat"].float()
    k_nat = batch["k_nat"].float()
    m_nat = (n_nat > 0).float()
    bb_nat = _single_bb_nll(k_nat, torch.clamp(n_nat, min=1.0), out["p_nat"], out["kappa_nat"]) / torch.clamp(n_nat, min=1.0)
    loss_nat = (bb_nat * m_nat).sum() / max(float(m_nat.sum().item()), float(B))

    trans_reg = 0.08 * out["transport_l1"].float()
    loss = (
        loss_ov
        + loss_mom
        + 0.40 * loss_pars
        + 0.12 * loss_sex_bb
        + 0.30 * loss_shift_sex
        + 0.25 * loss_div_mag
        + 0.08 * loss_nat
        + trans_reg
        + 0.03 * out["reg_loss"].float()
    )
    return loss, {"loss": float(loss.detach().item())}


@torch.no_grad()
def predict_epic(
    model: EpicPerspectivistModel,
    loader: DataLoader,
    device: torch.device,
    mc_dropout_T: int = 1,
    amp_enabled: bool = False,
) -> Dict[str, np.ndarray]:
    if mc_dropout_T > 1:
        model.train()
    else:
        model.eval()

    acc: Dict[str, List[np.ndarray]] = {k: [] for k in ("p_ov", "p_m", "p_f", "p_nat", "u_plur", "u_amb")}
    meta_k, meta_n, meta_km, meta_nm, meta_kf, meta_nf, meta_kn, meta_nn, meta_pol, meta_una = [], [], [], [], [], [], [], [], [], []
    groups: List[str] = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        seg = batch["seg_ids"].to(device)
        sf = batch["struct_feats"].to(device)

        runs: Dict[str, List[torch.Tensor]] = {k: [] for k in ("p_ov", "p_m", "p_f", "p_nat", "u_plur", "u_amb")}
        for _ in range(mc_dropout_T):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(ids, mask, seg, struct_feats=sf)
            runs["p_ov"].append(out["p_overall"].float())
            runs["p_m"].append(out["p_male"].float())
            runs["p_f"].append(out["p_female"].float())
            runs["p_nat"].append(out["p_nat"].float())
            runs["u_plur"].append(out["u_plur"].float())
            runs["u_amb"].append(out["u_amb"].float())

        p_ov_stk = torch.stack(runs["p_ov"], dim=0)
        p_ov_mean = p_ov_stk.mean(dim=0)
        # For MC-Dropout baselines, epistemic variance across stochastic passes populates u_plur naturally
        u_plur_out = p_ov_stk.var(dim=0, unbiased=False) if mc_dropout_T > 1 else torch.stack(runs["u_plur"], dim=0).mean(dim=0)

        acc["p_ov"].append(p_ov_mean.cpu().numpy())
        acc["p_m"].append(torch.stack(runs["p_m"], dim=0).mean(dim=0).cpu().numpy())
        acc["p_f"].append(torch.stack(runs["p_f"], dim=0).mean(dim=0).cpu().numpy())
        acc["p_nat"].append(torch.stack(runs["p_nat"], dim=0).mean(dim=0).cpu().numpy())
        acc["u_plur"].append(u_plur_out.cpu().numpy())
        acc["u_amb"].append(torch.stack(runs["u_amb"], dim=0).mean(dim=0).cpu().numpy())

        meta_k.append(batch["k_iro"].numpy())
        meta_n.append(batch["n_annot"].numpy())
        meta_km.append(batch["k_male"].numpy())
        meta_nm.append(batch["n_male"].numpy())
        meta_kf.append(batch["k_female"].numpy())
        meta_nf.append(batch["n_female"].numpy())
        meta_kn.append(batch["k_nat"].numpy())
        meta_nn.append(batch["n_nat"].numpy())
        meta_pol.append(batch["is_pol"])
        meta_una.append(batch["is_una"])
        groups.extend(batch["group_keys"])

    return {
        "p_ov_hat": np.concatenate(acc["p_ov"]),
        "p_m_hat": np.concatenate(acc["p_m"]),
        "p_f_hat": np.concatenate(acc["p_f"]),
        "p_nat_hat": np.concatenate(acc["p_nat"]),
        "u_plur": np.concatenate(acc["u_plur"]),
        "u_amb": np.concatenate(acc["u_amb"]),
        "k_iro": np.concatenate(meta_k),
        "n_annot": np.concatenate(meta_n),
        "k_male": np.concatenate(meta_km),
        "n_male": np.concatenate(meta_nm),
        "k_female": np.concatenate(meta_kf),
        "n_female": np.concatenate(meta_nf),
        "k_nat": np.concatenate(meta_kn),
        "n_nat": np.concatenate(meta_nn),
        "is_pol": np.concatenate(meta_pol),
        "is_una": np.concatenate(meta_una),
        "group_keys": np.array(groups, dtype=object),
    }


def evaluate_epic(pred: Dict[str, np.ndarray], prefix: str = "test") -> Dict[str, float]:
    p_true = pred["k_iro"] / np.maximum(pred["n_annot"], 1.0)
    p_hat = pred["p_ov_hat"]
    u_plur = pred["u_plur"]
    u_amb = pred["u_amb"]

    out: Dict[str, float] = {
        f"{prefix}_overall_mae": float(np.mean(np.abs(p_true - p_hat))),
        f"{prefix}_overall_pearson": _safe_r(p_true, p_hat),
        f"{prefix}_overall_brier": float(np.mean((p_true - p_hat) ** 2)),
    }

    # Unanimous vs Polarized irony detection
    pu_mask = pred["is_pol"] | pred["is_una"]
    if np.sum(pred["is_pol"]) >= 5 and np.sum(pred["is_una"]) >= 5:
        y_pol = pred["is_pol"][pu_mask].astype(int)
        std_hat = np.sqrt(np.clip(p_hat * (1.0 - p_hat), 0.0, None))
        out[f"{prefix}_pol_auroc_by_std"] = _safe_auc(y_pol, std_hat[pu_mask])
        out[f"{prefix}_pol_auroc_by_uplur"] = _safe_auc(y_pol, u_plur[pu_mask])
        out[f"{prefix}_pol_auprc_by_uplur"] = _safe_ap(y_pol, u_plur[pu_mask])
        out[f"{prefix}_pol_auroc_by_uamb"] = _safe_auc(y_pol, u_amb[pu_mask])

    # STRICT Cross-Sex Evaluation on n_male >= 2 & n_female >= 2 (Gate 4!)
    ge2 = (pred["n_male"] >= 2) & (pred["n_female"] >= 2)
    out[f"{prefix}_both_sex_ge2_n"] = int(ge2.sum())
    if ge2.sum() >= 10:
        pm = pred["k_male"][ge2] / pred["n_male"][ge2]
        pf = pred["k_female"][ge2] / pred["n_female"][ge2]
        true_shift = pf - pm
        pred_shift = pred["p_f_hat"][ge2] - pred["p_m_hat"][ge2]
        true_div = np.abs(true_shift)
        pred_div = np.abs(pred_shift)
        y_div = (true_div >= 0.35).astype(int)

        out[f"{prefix}_sex_shift_mae"] = float(np.mean(np.abs(true_shift - pred_shift)))
        out[f"{prefix}_sex_shift_pearson"] = _safe_r(true_shift, pred_shift)
        out[f"{prefix}_between_sex_div_auroc"] = _safe_auc(y_div, pred_div)
        out[f"{prefix}_between_sex_div_auprc"] = _safe_ap(y_div, pred_div)
        out[f"{prefix}_between_sex_div_combined_auroc"] = _safe_auc(y_div, pred_div + 0.5 * np.sqrt(np.clip(u_plur[ge2], 0.0, None)))

    # Cross-Nationality Evaluation (posts with >= 2 nationalities having >= 1 annotator)
    nn_cnt = (pred["n_nat"] >= 1).sum(axis=-1)
    m_nat = nn_cnt >= 2
    if m_nat.sum() >= 20:
        p_n_true = np.where(pred["n_nat"] > 0, pred["k_nat"] / np.maximum(pred["n_nat"], 1.0), np.nan)
        p_n_hat = np.where(pred["n_nat"] > 0, pred["p_nat_hat"], np.nan)
        nat_mae = float(np.nanmean(np.abs(p_n_true[m_nat] - p_n_hat[m_nat])))
        nat_spread_true = np.nanmax(p_n_true[m_nat], axis=-1) - np.nanmin(p_n_true[m_nat], axis=-1)
        nat_spread_hat = np.nanmax(p_n_hat[m_nat], axis=-1) - np.nanmin(p_n_hat[m_nat], axis=-1)
        out[f"{prefix}_nat_shift_mae"] = nat_mae
        out[f"{prefix}_nat_spread_auroc"] = _safe_auc((nat_spread_true >= 0.40).astype(int), nat_spread_hat)

    return out


def run_epic_experiment(
    backbone_name: str = "roberta-base",
    model_type: str = "patnet",
    ablation_mode: str = "M4_disentangle",
    cv_fold: int = -1,
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 16,
    grad_accum: int = 1,
    max_length: int = 160,
    encoder_lr: float = 2e-5,
    head_lr: float = 3e-4,
    mc_dropout_T: int = 1,
    num_workers: int = 0,
    out_dir_override: str | None = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    df = pd.read_parquet(PROC_DIR / "epic_agg.parquet")
    df_tr, df_sel, df_cal, df_te = get_epic_4way_splits(df, cv_fold=cv_fold)

    tr_pairs = [f"{p} {r}".strip() for p, r in zip(df_tr["parent_text"].fillna(""), df_tr["text"].fillna(""))]
    sel_pairs = [f"{p} {r}".strip() for p, r in zip(df_sel["parent_text"].fillna(""), df_sel["text"].fillna(""))]
    cal_pairs = [f"{p} {r}".strip() for p, r in zip(df_cal["parent_text"].fillna(""), df_cal["text"].fillna(""))]
    te_pairs = [f"{p} {r}".strip() for p, r in zip(df_te["parent_text"].fillna(""), df_te["text"].fillna(""))]

    tr_s = extract_struct_matrix(tr_pairs)
    scaler = StandardScaler().fit(tr_s)
    sf_tr = scaler.transform(tr_s).astype(np.float32)
    sf_sel = scaler.transform(extract_struct_matrix(sel_pairs)).astype(np.float32)
    sf_cal = scaler.transform(extract_struct_matrix(cal_pairs)).astype(np.float32)
    sf_te = scaler.transform(extract_struct_matrix(te_pairs)).astype(np.float32)

    resolved = resolve_backbone_path(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    special_ids = tuple(
        x for x in (tokenizer.cls_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id) if x is not None
    )
    collate_fn = make_epic_collate_fn(tokenizer, max_length=max_length)

    tr_loader = DataLoader(EpicDataset(df_tr, sf_tr), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    sel_loader = DataLoader(EpicDataset(df_sel, sf_sel), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    cal_loader = DataLoader(EpicDataset(df_cal, sf_cal), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    te_loader = DataLoader(EpicDataset(df_te, sf_te), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = EpicPerspectivistModel(
        backbone_name=backbone_name,
        model_type=model_type,
        ablation_mode=ablation_mode,
        special_token_ids=special_ids,
    ).to(device)
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, (len(tr_loader) + max(grad_accum, 1) - 1) // max(grad_accum, 1))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    fold_tag = "split0" if cv_fold < 0 else f"cv{cv_fold}"
    bb_tag = backbone_name.replace("/", "__")
    out_dir = Path(out_dir_override) if out_dir_override else (OUT_DIR / "epic" / f"{fold_tag}_{bb_tag}_{model_type}_{ablation_mode}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_sel: Dict[str, float] = {}
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tr_loader, start=1):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["seg_ids"].to(device),
                    struct_feats=batch["struct_feats"].to(device),
                )
                loss, _ = compute_epic_loss(model_type, out, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
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

        sel_pred = predict_epic(model, sel_loader, device, mc_dropout_T=1, amp_enabled=amp_enabled)
        sel_m = evaluate_epic(sel_pred, prefix="sel")
        # Composite selection score dominated by overall Pearson with gentle auxiliary tie-breakers on dev_select
        score = (
            sel_m.get("sel_overall_pearson", 0.0)
            + 0.08 * sel_m.get("sel_between_sex_div_auroc", 0.5)
            + 0.08 * sel_m.get("sel_pol_auroc_by_uplur", sel_m.get("sel_pol_auroc_by_std", 0.5))
            - 0.10 * sel_m.get("sel_sex_shift_mae", 0.3)
        )
        print(f"[EPIC][{fold_tag} {model_type}] Ep {epoch}/{epochs} sel_r={sel_m['sel_overall_pearson']:.4f} sel_sex_div_auc={sel_m.get('sel_between_sex_div_auroc', 0.0):.4f} sel_sex_mae={sel_m.get('sel_sex_shift_mae', 0.0):.4f}")
        if score > best_score:
            best_score = score
            best_sel = sel_m
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    sel_pred = predict_epic(model, sel_loader, device, mc_dropout_T=mc_dropout_T, amp_enabled=amp_enabled)
    cal_pred = predict_epic(model, cal_loader, device, mc_dropout_T=mc_dropout_T, amp_enabled=amp_enabled)
    te_pred = predict_epic(model, te_loader, device, mc_dropout_T=mc_dropout_T, amp_enabled=amp_enabled)

    te_metrics = evaluate_epic(te_pred, prefix="test")

    # Operational Polarization & Cross-Sex Divergence Alerting + CRC (exact leakage-free 4-way flow)
    def _epic_risk_feats(p: Dict[str, np.ndarray]) -> np.ndarray:
        p_hat = p["p_ov_hat"]
        return np.column_stack([
            np.abs(p["p_f_hat"] - p["p_m_hat"]),
            np.sqrt(np.clip(p["u_plur"], 0.0, None)),
            np.sqrt(np.clip(p["u_amb"], 0.0, None)),
            np.sqrt(np.clip(p_hat * (1.0 - p_hat), 0.0, None)),
            np.nanstd(p["p_nat_hat"], axis=-1),
        ]).astype(float)

    # Polarization alert label: is_pol | between-sex divergence >= 0.35 (on ge2)
    def _alert_y(p: Dict[str, np.ndarray]) -> np.ndarray:
        ge2 = (p["n_male"] >= 2) & (p["n_female"] >= 2)
        s_div = np.zeros(len(p["n_male"]), dtype=bool)
        if np.any(ge2):
            pm = p["k_male"][ge2] / p["n_male"][ge2]
            pf = p["k_female"][ge2] / p["n_female"][ge2]
            s_div[ge2] = np.abs(pf - pm) >= 0.35
        return (p["is_pol"] | s_div).astype(int)

    sel_X, cal_X, te_X = _epic_risk_feats(sel_pred), _epic_risk_feats(cal_pred), _epic_risk_feats(te_pred)
    sel_y, cal_y, te_y = _alert_y(sel_pred), _alert_y(cal_pred), _alert_y(te_pred)

    crc_pol_10 = fit_and_calibrate_alert_crc(sel_X, sel_y, cal_X, cal_y, te_X, te_y, alpha=0.10, use_supervised_combiner=True)
    alert_raw = evaluate_binary_alert_metrics(te_y, np.abs(te_pred["p_f_hat"] - te_pred["p_m_hat"]) + np.sqrt(np.clip(te_pred["u_plur"], 0.0, None)))

    np.savez_compressed(out_dir / "test_predictions.npz", **te_pred)
    np.savez_compressed(out_dir / "cal_predictions.npz", **cal_pred)

    summary = {
        "task": "epic_irony_bb_transport",
        "backbone": backbone_name,
        "model_type": model_type,
        "ablation_mode": ablation_mode,
        "cv_fold": cv_fold,
        "seed": seed,
        "mc_dropout_T": mc_dropout_T,
        "best_sel": best_sel,
        "test": te_metrics,
        "alert_raw": alert_raw,
        "crc_pol_alpha10": crc_pol_10,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="roberta-base")
    parser.add_argument("--model-type", type=str, default="patnet")
    parser.add_argument("--ablation-mode", type=str, default="M4_disentangle")
    parser.add_argument("--cv-fold", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--mc-dropout-T", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run_epic_experiment(
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
        mc_dropout_T=args.mc_dropout_T,
        num_workers=args.num_workers,
        out_dir_override=args.out_dir,
    )


if __name__ == "__main__":
    main()
