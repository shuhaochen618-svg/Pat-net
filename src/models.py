"""Dual-Perspective Affect Distribution Models with BiPol-Disentangle Reader and Capacity-Matched Baselines."""

from __future__ import annotations

from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .feature_reader import AblationConfig, BiPolDisentangleReader, ablation_from_name
from .losses import (
    beta_discretized_probs,
    build_ordinal_grid,
    dml_ltv_mixture_to_dist,
    dml_mixture_to_dist,
    gaussian_discretized_probs,
    unimodal_softmax_probs,
)
from .paths import resolve_backbone_path


def _make_mlp_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DualPerspectiveAffectModel(nn.Module):
    """Unified capacity-matched dual-perspective benchmark & BiPol-PATNet architecture."""

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        model_type: str = "patnet",
        num_dims: int = 3,
        num_bins: int = 5,
        latent_dim: int = 256,
        dropout: float = 0.2,
        score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
        ablation_mode: str = "M4_disentangle",
        special_token_ids: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_dims = num_dims
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.score_values = score_values
        self.special_token_ids = special_token_ids

        resolved = resolve_backbone_path(backbone_name)
        self.encoder = AutoModel.from_pretrained(resolved)
        hidden = int(self.encoder.config.hidden_size)
        n_layers = int(getattr(self.encoder.config, "num_hidden_layers", 12)) + 1

        self.ablation_cfg: AblationConfig = ablation_from_name(ablation_mode)
        self.reader = BiPolDisentangleReader(
            hidden_size=hidden,
            n_layers=n_layers,
            work_dim=latent_dim,
            ablation=self.ablation_cfg,
            dual_segment=False,
            num_query_heads=1,
        )
        self.drop = nn.Dropout(dropout)

        d_disp = self.reader.d_disp
        d_vag = self.reader.d_vag
        d_prag = self.reader.prag_dim
        w_in_flat = latent_dim + d_disp + d_vag
        r_in_flat = latent_dim + d_prag + d_disp + d_vag

        if model_type == "scalar_mse":
            out_d = num_dims
        elif model_type in ("gaussian_nll", "uw_ldl", "beta_nll"):
            out_d = num_dims * 2
        elif model_type == "nig_evidential":
            out_d = num_dims * 4
        elif model_type in ("softmax_ldl", "sord", "dldl_v2", "dc_ldl"):
            out_d = num_dims * num_bins
        elif model_type in ("patnet_unimodal", "patnet_no_transport", "patnet"):
            out_d = num_dims * 5
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Factorized LTV heads ONLY for patnet*; all other baselines share equal-depth 2-layer MLPs
        if model_type in ("patnet_unimodal", "patnet_no_transport", "patnet"):
            rho_drop = max(dropout - 0.02, 0.14)
            self.rho_drop = rho_drop
            if self.ablation_cfg.use_multilayer_conv:
                self.top_sem_proj = nn.Sequential(
                    nn.Linear(hidden * 2, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                )
                self.top_sem_gate = nn.Parameter(torch.tensor(0.35))
            else:
                self.top_sem_proj = None

            d_prof = 2 * num_dims
            w_var_in = latent_dim + d_disp + d_vag + d_prof
            r_var_in = latent_dim + d_prag + d_disp + d_vag + d_prof
            self.w_mu_head = _make_mlp_head(latent_dim, latent_dim, num_dims, dropout)
            self.w_split_head = _make_mlp_head(w_var_in, latent_dim, num_dims * 2, rho_drop)
            self.w_scale_head = _make_mlp_head(w_var_in, latent_dim, num_dims * 2, dropout)
            with torch.no_grad():
                self.w_split_head[-1].bias[0::2].fill_(-1.52)
                self.w_split_head[-1].bias[1::2].zero_()
                self.w_scale_head[-1].bias[0::2].fill_(-1.10)
                self.w_scale_head[-1].bias[1::2].zero_()

            if model_type == "patnet_no_transport":
                self.r_mu_head = _make_mlp_head(r_in_flat, latent_dim, num_dims, dropout)
                self.r_split_head = _make_mlp_head(r_var_in, latent_dim, num_dims * 2, rho_drop)
                self.r_scale_head = _make_mlp_head(r_var_in, latent_dim, num_dims * 2, dropout)
                with torch.no_grad():
                    self.r_split_head[-1].bias[0::2].fill_(-1.52)
                    self.r_split_head[-1].bias[1::2].zero_()
                    self.r_scale_head[-1].bias[0::2].fill_(-1.10)
                    self.r_scale_head[-1].bias[1::2].zero_()
            else:
                self.prag_gate = nn.Sequential(nn.Linear(d_prag, num_dims * 5), nn.Sigmoid())
                self.trans_mu = _make_mlp_head(latent_dim + d_prag, latent_dim, num_dims, dropout)
                self.trans_split = _make_mlp_head(r_var_in, latent_dim, num_dims * 2, rho_drop)
                self.trans_scale = _make_mlp_head(r_var_in, latent_dim, num_dims * 2, dropout)
        else:
            self.w_head = _make_mlp_head(w_in_flat, latent_dim, out_d, dropout)
            self.r_head = _make_mlp_head(r_in_flat, latent_dim, out_d, dropout)

    @staticmethod
    def _compute_mu_profile(mu_raw: torch.Tensor) -> torch.Tensor:
        """Centered detached bounded-scale mean profile [rel, |rel| - 0.25] of shape (B, 2*D)."""
        m_norm = torch.sigmoid(mu_raw.detach().float())
        rel = (2.0 * m_norm - 1.0).to(mu_raw.dtype)
        return torch.cat([rel, torch.abs(rel) - 0.25], dim=-1)

    @staticmethod
    def _pack_shared_heads(
        mu: torch.Tensor,
        sp: torch.Tensor,
        sc: torch.Tensor,
        num_dims: int,
    ) -> torch.Tensor:
        """Pack factorized V/A/D LTV head outputs [mu, rho, std_tot, s_asym, logit_pi] into (B, D*5)."""
        return torch.stack([mu, sp[..., 0], sc[..., 0], sc[..., 1], sp[..., 1]], dim=-1).view(-1, num_dims * 5)

    def _decode_dml_params(self, raw_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rw = raw_w.view(-1, self.num_dims, 5)
        return rw[..., 0], rw[..., 1], rw[..., 2], rw[..., 3], rw[..., 4]

    def _forward_patnet_heads(
        self,
        rf: Dict[str, torch.Tensor],
        top_emb: torch.Tensor | None,
        bimodal: bool,
        msr: float,
        msc: float,
        mpd: float | None,
        msa: float,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s = self.drop(rf["z_s"])
        z_disp = self.drop(rf["z_disp"])
        h_vag = self.drop(rf["h_vag"])
        z_p = self.drop(rf["z_p"])

        if self.top_sem_proj is not None and top_emb is not None:
            top_glob = self.top_sem_proj(self.drop(top_emb))
            z_sem = z_s + torch.tanh(self.top_sem_gate) * top_glob
        else:
            z_sem = z_s

        if self.ablation_cfg.use_disentangle:
            z_s_sp = (0.35 * z_sem + 0.65 * z_sem.detach()) if bimodal else z_sem.detach()
            z_p_sp = (0.42 * z_p + 0.58 * z_p.detach()) if bimodal else z_p
            z_disp_sp = (0.42 * z_disp + 0.58 * z_disp.detach()) if bimodal else z_disp
            h_vag_sp = h_vag.detach()
            z_s_var = 0.45 * z_sem + 0.55 * z_sem.detach()
            z_disp_var = z_disp.detach()
            z_p_var = (0.50 * z_p + 0.50 * z_p.detach()) if bimodal else z_p
            z_disp_mu = z_disp.detach()
            h_vag_mu = h_vag.detach()
        else:
            z_s_sp = z_sem
            z_p_sp = z_p
            z_disp_sp = z_disp
            h_vag_sp = h_vag
            z_s_var = z_sem
            z_disp_var = z_disp
            z_p_var = z_p
            z_disp_mu = z_disp
            h_vag_mu = h_vag

        sc_prof_drop = 0.12 if self.drop.p > 0 else 0.0
        mu_w = self.w_mu_head(z_sem)
        prof_w = self._compute_mu_profile(mu_w)
        prof_w_sc = F.dropout(prof_w, p=sc_prof_drop, training=self.training)
        sc_w = self.w_scale_head(torch.cat([z_s_var, z_disp_var, h_vag, prof_w_sc], dim=-1)).view(-1, self.num_dims, 2)
        sp_w = self.w_split_head(torch.cat([z_s_sp, z_disp_sp, h_vag_sp, prof_w], dim=-1)).view(-1, self.num_dims, 2)
        raw_w = self._pack_shared_heads(mu_w, sp_w, sc_w, self.num_dims)

        out_w = dml_ltv_mixture_to_dist(
            *self._decode_dml_params(raw_w),
            score_values=self.score_values,
            bimodal=bimodal,
            max_scale_ratio=msr,
            min_scale=msc,
            max_pi_dev=mpd,
            max_scale_asym=msa,
        )

        if self.model_type == "patnet_no_transport":
            zs_p_mu = torch.cat([z_sem, z_p], dim=-1)
            zs_p_sp = torch.cat([z_s_sp, z_p_sp], dim=-1)
            mu_r = self.r_mu_head(torch.cat([zs_p_mu, z_disp_mu, h_vag_mu], dim=-1))
            prof_r = self._compute_mu_profile(mu_r)
            prof_r_sc = F.dropout(prof_r, p=sc_prof_drop, training=self.training)
            sc_r = self.r_scale_head(torch.cat([z_s_var, z_p_var, z_disp_var, h_vag, prof_r_sc], dim=-1)).view(-1, self.num_dims, 2)
            sp_r = self.r_split_head(torch.cat([zs_p_sp, z_disp_sp, h_vag_sp, prof_r], dim=-1)).view(-1, self.num_dims, 2)
            raw_r = self._pack_shared_heads(mu_r, sp_r, sc_r, self.num_dims)
            transport_l1 = raw_w.new_zeros(())
        else:
            pg_flat = self.prag_gate(z_p)
            pg_mu = pg_flat.view(-1, self.num_dims, 5)[..., 0]
            t_mu = self.trans_mu(torch.cat([z_sem, z_p], dim=-1))
            mu_r_est = mu_w + (0.5 + pg_mu) * t_mu
            prof_r = self._compute_mu_profile(mu_r_est)
            prof_r_sc = F.dropout(prof_r, p=sc_prof_drop, training=self.training)
            t_sp = self.trans_split(torch.cat([z_s_sp, z_p_sp, z_disp_sp, h_vag_sp, prof_r], dim=-1)).view(-1, self.num_dims, 2)
            t_sc = self.trans_scale(torch.cat([z_s_var, z_p_var, z_disp_var, h_vag, prof_r_sc], dim=-1)).view(-1, self.num_dims, 2)
            t_raw = self._pack_shared_heads(t_mu, t_sp, t_sc, self.num_dims)
            delta_raw = (0.5 + pg_flat) * t_raw
            raw_r = raw_w + delta_raw
            transport_l1 = delta_raw.abs().mean()

        out_r = dml_ltv_mixture_to_dist(
            *self._decode_dml_params(raw_r),
            score_values=self.score_values,
            bimodal=bimodal,
            max_scale_ratio=msr,
            min_scale=msc,
            max_pi_dev=mpd,
            max_scale_asym=msa,
        )
        return out_w, out_r, transport_l1, raw_w, raw_r

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        struct_feats: torch.Tensor | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor]:
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        rf = self.reader(
            enc.hidden_states,
            attention_mask,
            input_ids=input_ids,
            special_token_ids=self.special_token_ids,
            struct_feats=struct_feats,
        )

        v_grid, _ = build_ordinal_grid(self.score_values, device=rf["z_s"].device, dtype=torch.float32)

        if self.model_type in ("patnet_unimodal", "patnet_no_transport", "patnet"):
            bimodal = self.model_type != "patnet_unimodal"
            if self.top_sem_proj is not None:
                top = enc.hidden_states[-1]
                cm_f = rf["content_mask"].unsqueeze(-1).to(top.dtype)
                top_mean = (top * cm_f).sum(dim=1) / torch.clamp(cm_f.sum(dim=1), min=1.0)
                top_emb = torch.cat([top[:, 0, :], top_mean], dim=-1)
            else:
                top_emb = None

            msr = 0.114 if bimodal else 0.22
            msc = 0.055 if bimodal else 0.055
            mpd = 0.05 if bimodal else None
            msa = 0.06 if bimodal else 0.0

            out_w, out_r, transport_l1, raw_w1, raw_r1 = self._forward_patnet_heads(rf, top_emb, bimodal, msr, msc, mpd, msa)
            if self.training:
                out_w2, out_r2, _, raw_w2, raw_r2 = self._forward_patnet_heads(rf, top_emb, bimodal, msr, msc, mpd, msa)

                def _mean_detached_dml_probs(raw_params: torch.Tensor, raw_ref: torch.Tensor) -> torch.Tensor:
                    mu_ref, _, _, _, _ = self._decode_dml_params(raw_ref)
                    _, rho_p, std_p, asym_p, pi_p = self._decode_dml_params(raw_params)
                    d = dml_ltv_mixture_to_dist(
                        mu_ref.detach(),
                        rho_p,
                        std_p,
                        asym_p,
                        pi_p,
                        score_values=self.score_values,
                        bimodal=bimodal,
                        max_scale_ratio=msr,
                        min_scale=msc,
                        max_pi_dev=mpd,
                        max_scale_asym=msa,
                    )
                    return torch.clamp(d["probs"].float(), min=1e-6)

                q_r1 = _mean_detached_dml_probs(raw_r1, raw_r1)
                q_r2 = _mean_detached_dml_probs(raw_r2, raw_r1)
                p_r1 = torch.clamp(out_r["probs"].float(), min=1e-6)
                p_r2 = torch.clamp(out_r2["probs"].float(), min=1e-6)
                rdrop_shape_kl = 0.5 * (
                    (q_r1 * (q_r1.log() - q_r2.log())).sum(dim=-1).mean()
                    + (q_r2 * (q_r2.log() - q_r1.log())).sum(dim=-1).mean()
                )
                rdrop_full_kl = 0.5 * (
                    (p_r1 * (p_r1.log() - p_r2.log())).sum(dim=-1).mean()
                    + (p_r2 * (p_r2.log() - p_r1.log())).sum(dim=-1).mean()
                )
                rdrop_up = F.smooth_l1_loss(
                    out_r["u_plur"].float(), out_r2["u_plur"].float(), beta=0.016
                )
                if bimodal:
                    q_w1 = _mean_detached_dml_probs(raw_w1, raw_w1)
                    q_w2 = _mean_detached_dml_probs(raw_w2, raw_w1)
                    p_w1 = torch.clamp(out_w["probs"].float(), min=1e-6)
                    p_w2 = torch.clamp(out_w2["probs"].float(), min=1e-6)
                    rdrop_shape_kl = 0.85 * rdrop_shape_kl + 0.20 * (
                        (q_w1 * (q_w1.log() - q_w2.log())).sum(dim=-1).mean()
                        + (q_w2 * (q_w2.log() - q_w1.log())).sum(dim=-1).mean()
                    )
                    rdrop_full_kl = 0.85 * rdrop_full_kl + 0.20 * (
                        (p_w1 * (p_w1.log() - p_w2.log())).sum(dim=-1).mean()
                        + (p_w2 * (p_w2.log() - p_w1.log())).sum(dim=-1).mean()
                    )
                    rdrop_up = 0.5 * rdrop_up + 0.25 * F.smooth_l1_loss(
                        out_w["u_plur"].float(), out_w2["u_plur"].float(), beta=0.02
                    )
                rf["reg_loss"] = rf["reg_loss"] + 0.340 * rdrop_shape_kl + 0.130 * rdrop_full_kl + 0.090 * rdrop_up
            return {
                "w": out_w,
                "r": out_r,
                "transport_l1": transport_l1,
                "reg_loss": rf["reg_loss"],
                "z_s": rf["z_s"],
                "z_p": rf["z_p"],
            }

        z_s = self.drop(rf["z_s"])
        z_disp = self.drop(rf["z_disp"])
        h_vag = self.drop(rf["h_vag"])
        z_p = self.drop(rf["z_p"])

        w_feat = torch.cat([z_s, z_disp, h_vag], dim=-1)
        z_s_r = z_s.detach() if self.ablation_cfg.use_disentangle else z_s
        r_feat = torch.cat([z_s_r, z_p, z_disp, h_vag], dim=-1)

        if self.model_type in ("softmax_ldl", "sord", "dldl_v2", "dc_ldl"):
            out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
            for persp, feat in (("w", w_feat), ("r", r_feat)):
                head = self.w_head if persp == "w" else self.r_head
                logits = head(feat).float().view(-1, self.num_dims, self.num_bins)
                probs = torch.softmax(logits, dim=-1)
                cdf_int = torch.cumsum(probs, dim=-1)[..., :-1]
                mean = (probs * v_grid.view(1, 1, -1)).sum(dim=-1)
                var = torch.clamp((probs * (v_grid.view(1, 1, -1) - mean.unsqueeze(-1)).pow(2)).sum(dim=-1), min=0.0)
                std = torch.sqrt(var + 1e-8)
                ent = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1)
                out[persp] = {
                    "probs": probs,
                    "cdf_int": cdf_int,
                    "disc_mean": mean,
                    "disc_var": var,
                    "disc_std": std,
                    "u_amb": ent,
                    "u_plur": var,
                }
            return out

        if self.model_type == "uw_ldl":
            out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
            for persp, feat in (("w", w_feat), ("r", r_feat)):
                head = self.w_head if persp == "w" else self.r_head
                out[persp] = unimodal_softmax_probs(head(feat).view(-1, self.num_dims, 2), self.score_values)
            return out

        if self.model_type == "beta_nll":
            out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
            for persp, feat in (("w", w_feat), ("r", r_feat)):
                head = self.w_head if persp == "w" else self.r_head
                raw = head(feat).float().view(-1, self.num_dims, 2)
                alpha = F.softplus(raw[..., 0]) + 0.20
                beta = F.softplus(raw[..., 1]) + 0.20
                out[persp] = beta_discretized_probs(alpha, beta, self.score_values)
            return out

        if self.model_type == "scalar_mse":
            v_min = float(self.score_values[0])
            span = float(self.score_values[-1] - v_min)
            out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
            for persp, feat in (("w", w_feat), ("r", r_feat)):
                head = self.w_head if persp == "w" else self.r_head
                mu = v_min + span * torch.sigmoid(head(feat).float())
                zeros = torch.zeros_like(mu)
                eye_idx = torch.clamp(torch.round(mu - v_min).long(), 0, self.num_bins - 1)
                one_hot = F.one_hot(eye_idx, num_classes=self.num_bins).float()
                out[persp] = {
                    "probs": one_hot,
                    "cdf_int": torch.cumsum(one_hot, dim=-1)[..., :-1],
                    "disc_mean": mu,
                    "disc_var": zeros,
                    "disc_std": zeros,
                    "u_amb": zeros,
                    "u_plur": zeros,
                }
            return out

        if self.model_type == "gaussian_nll":
            v_min = float(self.score_values[0])
            span = float(self.score_values[-1] - v_min)
            out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
            for persp, feat in (("w", w_feat), ("r", r_feat)):
                head = self.w_head if persp == "w" else self.r_head
                raw = head(feat).float().view(-1, self.num_dims, 2)
                mu = v_min + span * torch.sigmoid(raw[..., 0])
                std = F.softplus(raw[..., 1]) + 0.05
                d = gaussian_discretized_probs(mu, std, self.score_values)
                d["mu_param"] = mu
                d["std_param"] = std
                out[persp] = d
            return out

        # nig_evidential
        v_min = float(self.score_values[0])
        span = float(self.score_values[-1] - v_min)
        out = {"reg_loss": rf["reg_loss"], "transport_l1": z_s.new_zeros(())}
        for persp, feat in (("w", w_feat), ("r", r_feat)):
            head = self.w_head if persp == "w" else self.r_head
            raw = head(feat).float().view(-1, self.num_dims, 4)
            gamma = v_min + span * torch.sigmoid(raw[..., 0])
            nu = F.softplus(raw[..., 1]) + 1e-2
            alpha = F.softplus(raw[..., 2]) + 1.05
            beta = F.softplus(raw[..., 3]) + 1e-2
            aleatoric_var = beta / torch.clamp(alpha - 1.0, min=0.05)
            epistemic_var = aleatoric_var / nu
            std = torch.sqrt(torch.clamp(aleatoric_var, min=1e-4, max=9.0))
            d = gaussian_discretized_probs(gamma, std, self.score_values)
            d["gamma"] = gamma
            d["nu"] = nu
            d["alpha"] = alpha
            d["beta"] = beta
            d["u_amb"] = aleatoric_var
            d["u_plur"] = epistemic_var
            out[persp] = d
        return out
