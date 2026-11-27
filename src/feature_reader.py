"""BiPol-Disentangle: Anti-Cancellation Multi-Scale Bipolar Affect Reader & Orthogonal Perspective Disentangler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .struct_features import STRUCT_DIM


@dataclass
class AblationConfig:
    """Modular feature-reader ablation flags (M0 through M5)."""

    name: str = "M4_disentangle"
    use_multilayer_conv: bool = True
    use_bipolar_pool: bool = True
    use_tension_route: bool = True
    use_disentangle: bool = True
    use_struct: bool = False
    w_indep: float = 0.05
    w_ib: float = 1e-3
    w_orth: float = 0.02


def ablation_from_name(name: str) -> AblationConfig:
    presets = {
        "M0_baseline": AblationConfig(name="M0_baseline", use_multilayer_conv=False, use_bipolar_pool=False, use_tension_route=False, use_disentangle=False, use_struct=False),
        "M1_multilayer_conv": AblationConfig(name="M1_multilayer_conv", use_multilayer_conv=True, use_bipolar_pool=False, use_tension_route=False, use_disentangle=False, use_struct=False),
        "M2_bipolar_pool": AblationConfig(name="M2_bipolar_pool", use_multilayer_conv=True, use_bipolar_pool=True, use_tension_route=False, use_disentangle=False, use_struct=False),
        "M3_tension_route": AblationConfig(name="M3_tension_route", use_multilayer_conv=True, use_bipolar_pool=True, use_tension_route=True, use_disentangle=False, use_struct=False),
        "M4_disentangle": AblationConfig(name="M4_disentangle", use_multilayer_conv=True, use_bipolar_pool=True, use_tension_route=True, use_disentangle=True, use_struct=False),
        "M5_full_struct": AblationConfig(name="M5_full_struct", use_multilayer_conv=True, use_bipolar_pool=True, use_tension_route=True, use_disentangle=True, use_struct=True),
    }
    if name not in presets:
        raise ValueError(f"Unknown ablation preset: {name}. Available: {list(presets)}")
    return presets[name]


def build_interior_content_mask(
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    special_token_ids: Tuple[int, ...] | None = (),
) -> torch.Tensor:
    """Exclude PAD, position-0 CLS/BOS, and trailing EOS/SEP tokens whenever >= 2 interior tokens exist."""
    base = attention_mask.bool()
    B, L = base.shape
    interior = base.clone()
    interior[:, 0] = False
    lengths = base.long().sum(dim=1)
    for b in range(B):
        last_idx = int(lengths[b].item()) - 1
        if last_idx > 0:
            interior[b, last_idx] = False
    if input_ids is not None and special_token_ids:
        for sid in special_token_ids:
            interior = interior & (input_ids != int(sid))
    has_enough = interior.long().sum(dim=1, keepdim=True) >= 2
    return torch.where(has_enough, interior, base)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    m = mask.unsqueeze(-1) if logits.ndim == 3 and mask.ndim == 2 else mask
    masked = logits.masked_fill(~m, -1e4 if logits.dtype == torch.float16 else -1e9)
    return torch.softmax(masked, dim=dim)


def compute_nhsic(z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
    """Normalized RBF Hilbert-Schmidt Independence Criterion (nHSIC) for nonlinear z_s perp z_p."""
    B = z_s.shape[0]
    if B < 4:
        return z_s.new_zeros((), dtype=torch.float32)
    x = F.normalize(z_s.float(), dim=-1)
    y = F.normalize(z_p.float(), dim=-1)

    d2_x = torch.cdist(x, x, p=2.0).pow(2)
    d2_y = torch.cdist(y, y, p=2.0).pow(2)
    with torch.no_grad():
        triu = torch.triu(torch.ones((B, B), device=x.device, dtype=torch.bool), diagonal=1)
        sig_x = torch.clamp(torch.median(d2_x[triu]), min=1e-4)
        sig_y = torch.clamp(torch.median(d2_y[triu]), min=1e-4)

    K = torch.exp(-d2_x / (2.0 * sig_x))
    L = torch.exp(-d2_y / (2.0 * sig_y))
    Kc = K - K.mean(dim=1, keepdim=True) - K.mean(dim=0, keepdim=True) + K.mean()
    Lc = L - L.mean(dim=1, keepdim=True) - L.mean(dim=0, keepdim=True) + L.mean()
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum()
    hsic_yy = (Lc * Lc).sum()
    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy + 1e-8)


class SingleAttentionPool(nn.Module):
    """Legacy single-head attention pooling kept cleanly for M0/M1 ablation checks."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = _masked_softmax(self.proj(hidden).squeeze(-1), mask, dim=1)
        return (hidden * w.unsqueeze(-1)).sum(dim=1)


class MultiLayerScalarMix(nn.Module):
    """Dual ELMo-style learnable layer mixing for content (lex) and pragmatics (prag)."""

    def __init__(self, n_layers: int, hidden_size: int, work_dim: int) -> None:
        super().__init__()
        # Initialize toward upper semantic layers so small-N fine-tuning avoids early Layer-0 dilution while staying learnable
        self.w_lex = nn.Parameter(torch.linspace(-1.5, 1.5, n_layers))
        prag_init = torch.zeros(n_layers)
        if n_layers >= 3:
            prag_init[-3:] = 1.0
        self.w_prag = nn.Parameter(prag_init)
        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_p = nn.Parameter(torch.ones(1))

        self.proj_s = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))
        self.proj_p = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))
        self.proj_cls = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H = torch.stack(hidden_states, dim=2)  # (B, L, n_layers, D)
        wl = torch.softmax(self.w_lex, dim=0).view(1, 1, -1, 1)
        wp = torch.softmax(self.w_prag, dim=0).view(1, 1, -1, 1)
        h_lex = self.gamma_s * (H * wl).sum(dim=2)
        h_prag = self.gamma_p * (H * wp).sum(dim=2)
        return (
            self.proj_s(h_lex),
            self.proj_p(h_prag),
            self.proj_cls(hidden_states[-1][:, 0, :]),
        )


class LocalModifierContrastReader(nn.Module):
    """Multi-scale Conv1d negation/modifier gate (U_s) + causal-anticausal clause pivot reader (U_p)."""

    def __init__(self, work_dim: int) -> None:
        super().__init__()
        c_branch = max(16, work_dim // 2)
        groups = 4 if work_dim % 4 == 0 else 1
        self.conv1 = nn.Conv1d(work_dim, c_branch, kernel_size=1)
        self.conv3_dw = nn.Conv1d(work_dim, work_dim, kernel_size=3, padding=1, groups=groups)
        self.conv3_pw = nn.Conv1d(work_dim, c_branch, kernel_size=1)
        self.conv5_dw = nn.Conv1d(work_dim, work_dim, kernel_size=5, padding=2, groups=groups)
        self.conv5_pw = nn.Conv1d(work_dim, c_branch, kernel_size=1)

        self.fuse_m = nn.Sequential(
            nn.Linear(c_branch * 3, work_dim),
            nn.LayerNorm(work_dim),
            nn.GELU(),
        )
        self.gate_mod = nn.Linear(work_dim * 2, work_dim)
        nn.init.constant_(self.gate_mod.bias, -1.5)
        self.cand_mod = nn.Linear(work_dim * 2, work_dim)

        self.prag_fuse = nn.Sequential(
            nn.Linear(work_dim * 3, work_dim),
            nn.GELU(),
            nn.LayerNorm(work_dim),
        )

    @staticmethod
    def _left_right_contrast(h: torch.Tensor, mask: torch.Tensor, w: int = 3) -> torch.Tensor:
        """Compute Avg_{[t-w, t-1]}(h) - Avg_{[t+1, t+w]}(h) strictly respecting sequence masks."""
        B, L, D = h.shape
        m_f = mask.unsqueeze(-1).to(h.dtype)
        hm = h * m_f
        left_sum = torch.zeros_like(hm)
        left_cnt = torch.zeros((B, L, 1), device=h.device, dtype=h.dtype)
        right_sum = torch.zeros_like(hm)
        right_cnt = torch.zeros((B, L, 1), device=h.device, dtype=h.dtype)
        for d in range(1, w + 1):
            if L > d:
                left_sum[:, d:, :] += hm[:, :-d, :]
                left_cnt[:, d:, :] += m_f[:, :-d, :]
                right_sum[:, :-d, :] += hm[:, d:, :]
                right_cnt[:, :-d, :] += m_f[:, d:, :]
        left_avg = left_sum / torch.clamp(left_cnt, min=1.0)
        right_avg = right_sum / torch.clamp(right_cnt, min=1.0)
        return (left_avg - right_avg) * m_f

    def forward(
        self,
        h_s: torch.Tensor,
        h_p: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m_f = attention_mask.unsqueeze(-1).to(h_s.dtype)
        x = (h_s * m_f).transpose(1, 2)  # (B, D, L)
        c1 = self.conv1(x)
        c3 = self.conv3_pw(self.conv3_dw(x))
        c5 = self.conv5_pw(self.conv5_dw(x))
        m_prime = self.fuse_m(torch.cat([c1, c3, c5], dim=1).transpose(1, 2)) * m_f

        cat_s = torch.cat([h_s, m_prime], dim=-1)
        g_mod = torch.sigmoid(self.gate_mod(cat_s)) * m_f
        u_s = h_s + g_mod * torch.tanh(self.cand_mod(cat_s))

        delta_p = self._left_right_contrast(h_p, attention_mask, w=3)
        cat_p = torch.cat([h_p, delta_p.abs(), h_p * delta_p], dim=-1)
        u_p = h_p + self.prag_fuse(cat_p) * m_f
        return u_s, u_p, g_mod.norm(dim=-1), delta_p.norm(dim=-1)


class BipolarTensionPooling(nn.Module):
    """Antisymmetric bipolar pole pooling + orthogonal context slots + explicit 21d dispersion readout."""

    def __init__(self, work_dim: int, k_axis: int = 4, m_ctx: int = 2, tau_pool: float = 0.25) -> None:
        super().__init__()
        self.k_axis = k_axis
        self.m_ctx = m_ctx
        self.tau_pool = tau_pool
        self.d_pole = max(4, work_dim // k_axis)
        self.d_bipolar = self.k_axis * self.d_pole

        self.tok_ln = nn.LayerNorm(work_dim)
        self.w_axis = nn.Parameter(torch.randn(k_axis, work_dim) * (1.0 / (work_dim ** 0.5)))
        self.val_proj = nn.Linear(work_dim, self.d_pole)
        self.diff_proj = nn.Sequential(
            nn.Linear(self.d_bipolar * 2, max(16, work_dim // 2)),
            nn.GELU(),
            nn.LayerNorm(max(16, work_dim // 2)),
        )

        self.q_ctx = nn.Parameter(torch.randn(m_ctx, work_dim) * (1.0 / (work_dim ** 0.5)))
        self.ctx_proj = nn.Sequential(nn.Linear(m_ctx * work_dim, work_dim), nn.LayerNorm(work_dim))
        self.sal_proj = nn.Linear(work_dim, 1)

        disp_in_dim = k_axis * 5 + 1  # 21 scalars
        self.d_tension = max(8, work_dim // 4)
        self.tension_proj = nn.Sequential(
            nn.Linear(disp_in_dim, self.d_tension),
            nn.LayerNorm(self.d_tension),
            nn.GELU(),
        )
        self.d_disp = max(16, work_dim // 2) + self.d_tension
        self.disp_ln = nn.LayerNorm(self.d_disp)
        self.d_vag = max(8, work_dim // 8)
        self.vag_proj = nn.Sequential(
            nn.Linear(k_axis + 1, self.d_vag),
            nn.LayerNorm(self.d_vag),
            nn.GELU(),
        )

    def forward(
        self,
        u_s: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, L, D = u_s.shape
        u_norm = self.tok_ln(u_s)
        w_unit = F.normalize(self.w_axis, dim=-1)
        A = torch.einsum("bld,kd->blk", u_norm, w_unit)  # (B, L, K_axis)

        mask_k = content_mask.unsqueeze(-1).expand(-1, -1, self.k_axis)
        alpha_pos = _masked_softmax(A / self.tau_pool, mask_k, dim=1)    # (B, L, K)
        alpha_neg = _masked_softmax(-A / self.tau_pool, mask_k, dim=1)   # (B, L, K)

        v_tok = self.val_proj(u_s)  # (B, L, d_pole)
        h_pos = torch.einsum("blk,bld->bkd", alpha_pos, v_tok)  # (B, K, d_pole)
        h_neg = torch.einsum("blk,bld->bkd", alpha_neg, v_tok)  # (B, K, d_pole)

        h_pos_flat = h_pos.reshape(B, self.d_bipolar)
        h_neg_flat = h_neg.reshape(B, self.d_bipolar)
        z_center = 0.5 * (h_pos_flat + h_neg_flat)
        diff_vec = h_pos_flat - h_neg_flat
        z_diff = self.diff_proj(torch.cat([diff_vec.abs(), diff_vec], dim=-1))

        # Orthogonal context slots
        ctx_logits = torch.einsum("bld,md->blm", u_s, self.q_ctx)
        alpha_ctx = _masked_softmax(ctx_logits, content_mask.unsqueeze(-1).expand(-1, -1, self.m_ctx), dim=1)
        z_ctx_slots = torch.einsum("blm,bld->bmd", alpha_ctx, u_s)
        z_ctx = self.ctx_proj(z_ctx_slots.reshape(B, self.m_ctx * D))

        # Orthogonality losses (axis + context slots)
        gram_axis = w_unit @ w_unit.t()
        loss_axis = (gram_axis - torch.eye(self.k_axis, device=u_s.device, dtype=gram_axis.dtype)).pow(2).sum()
        a_ctx_t = alpha_ctx.transpose(1, 2)  # (B, M, L)
        gram_ctx = torch.bmm(a_ctx_t, alpha_ctx)
        eye_ctx = torch.eye(self.m_ctx, device=u_s.device, dtype=gram_ctx.dtype).unsqueeze(0)
        loss_slot = (gram_ctx - eye_ctx).pow(2).mean()

        # Explicit float32 token-level second-order dispersion & bimodal co-activation (anti-collapse!)
        A_f = A.float()
        w_sal = _masked_softmax(self.sal_proj(u_s).squeeze(-1).float(), content_mask, dim=1)  # (B, L)
        mu_A = (w_sal.unsqueeze(-1) * A_f).sum(dim=1, keepdim=True)
        sigma_tok = torch.sqrt((w_sal.unsqueeze(-1) * (A_f - mu_A).pow(2)).sum(dim=1) + 1e-6)  # (B, K)
        span_tok = (alpha_pos.float() * A_f).sum(dim=1) - (alpha_neg.float() * A_f).sum(dim=1)  # (B, K)
        pole_gap = (h_pos.float() - h_neg.float()).norm(dim=-1) / math.sqrt(float(self.d_pole))
        cos_opp = 1.0 - F.cosine_similarity(h_pos.float(), h_neg.float(), dim=-1)

        sig_pos = torch.sigmoid(A_f).masked_fill(~mask_k, 0.0)
        sig_neg = torch.sigmoid(-A_f).masked_fill(~mask_k, 0.0)
        b_coact = sig_pos.max(dim=1).values * sig_neg.max(dim=1).values  # (B, K)

        u_f = u_s.float()
        mu_u = (w_sal.unsqueeze(-1) * u_f).sum(dim=1, keepdim=True)
        sigma_emb = torch.sqrt((w_sal.unsqueeze(-1) * (u_f - mu_u).pow(2)).sum(dim=1).mean(dim=-1, keepdim=True) + 1e-6)

        tau_disp = torch.cat([sigma_tok, span_tok, pole_gap, cos_opp, b_coact, sigma_emb], dim=-1).to(u_s.dtype)
        h_tension = self.tension_proj(tau_disp)
        z_disp = self.disp_ln(torch.cat([z_diff, h_tension], dim=-1))

        vag_mask = 1.0 - b_coact.max(dim=-1, keepdim=True).values
        vag_raw = torch.cat([sigma_emb * vag_mask, sigma_tok * (1.0 - b_coact)], dim=-1).to(u_s.dtype)
        h_vag = self.vag_proj(vag_raw)

        return {
            "z_center": z_center,
            "z_ctx": z_ctx,
            "z_disp": z_disp,
            "h_vag": h_vag,
            "loss_orth": loss_axis + loss_slot,
            "alpha_pos": alpha_pos.detach(),
            "alpha_neg": alpha_neg.detach(),
            "b_coact": b_coact.detach(),
        }


class BiPolDisentangleReader(nn.Module):
    """Unified drop-in affective feature & perspective reader supporting M0..M5 + task adapters."""

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        work_dim: int = 256,
        ablation: AblationConfig | None = None,
        dual_segment: bool = False,
        num_query_heads: int = 1,
        use_prag_disentangle: bool = True,
    ) -> None:
        super().__init__()
        self.ab = ablation or AblationConfig()
        self.work_dim = work_dim
        self.dual_segment = dual_segment
        self.num_query_heads = num_query_heads
        self.use_prag_disentangle = use_prag_disentangle
        self.prag_dim = max(8, work_dim // 2)

        # Stage 1
        if self.ab.use_multilayer_conv:
            self.layer_mix = MultiLayerScalarMix(n_layers, hidden_size, work_dim)
            self.local_reader = LocalModifierContrastReader(work_dim)
        else:
            self.legacy_proj_s = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))
            self.legacy_proj_p = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))
            self.legacy_proj_cls = nn.Sequential(nn.Linear(hidden_size, work_dim), nn.LayerNorm(work_dim))

        # Stage 3
        self.legacy_attn_s = SingleAttentionPool(work_dim)
        self.bipolar_pool = BipolarTensionPooling(work_dim) if self.ab.use_bipolar_pool else None

        self.d_disp = self.bipolar_pool.d_disp if self.bipolar_pool is not None else max(16, work_dim // 2)
        self.d_vag = self.bipolar_pool.d_vag if self.bipolar_pool is not None else max(8, work_dim // 8)

        # Optional zero-lexicon structural projection (M5)
        self.struct_proj = nn.Sequential(nn.Linear(STRUCT_DIM, max(16, work_dim // 4)), nn.LayerNorm(max(16, work_dim // 4)), nn.GELU()) if self.ab.use_struct else None
        struct_out = max(16, work_dim // 4) if self.ab.use_struct else 0

        # Content bottleneck z_s anchored on clean semantic pooling + bounded bipolar/context residual
        s_in_dim = work_dim * 2 + struct_out
        self.content_mlp = nn.Sequential(
            nn.Linear(s_in_dim, work_dim),
            nn.LayerNorm(work_dim),
            nn.GELU(),
        )
        self.bipolar_s_proj = (
            nn.Sequential(nn.Linear(work_dim * 2, work_dim), nn.LayerNorm(work_dim))
            if self.ab.use_bipolar_pool
            else None
        )
        self.bipolar_res_gate = nn.Parameter(torch.full((1,), 0.10)) if self.ab.use_bipolar_pool else None

        # Pragmatic pooling & VIB z_p
        self.prag_attn = nn.Linear(work_dim * 2 if self.ab.use_multilayer_conv else work_dim, 1)
        p_in_dim = work_dim * (3 if self.dual_segment else 1) + struct_out
        self.prag_mu = nn.Linear(p_in_dim, self.prag_dim)
        self.prag_logvar = nn.Linear(p_in_dim, self.prag_dim)

        # Optional per-emotion / per-label query pooling for BRIGHTER multi-emotion disentanglement
        if num_query_heads > 1:
            self.q_heads = nn.Parameter(torch.randn(num_query_heads, work_dim) * (1.0 / math.sqrt(work_dim)))

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        special_token_ids: Tuple[int, ...] = (),
        struct_feats: torch.Tensor | None = None,
        seg_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = hidden_states[-1].shape
        content_mask = build_interior_content_mask(attention_mask, input_ids=input_ids, special_token_ids=special_token_ids)

        if self.ab.use_multilayer_conv:
            h_s, h_p, h_cls = self.layer_mix(hidden_states)
            u_s, u_p, g_mod_norm, delta_p_norm = self.local_reader(h_s, h_p, attention_mask.bool())
        else:
            top = hidden_states[-1]
            u_s = self.legacy_proj_s(top)
            u_p = self.legacy_proj_p(top)
            h_cls = self.legacy_proj_cls(top[:, 0, :])
            g_mod_norm = torch.zeros((B, L), device=top.device, dtype=top.dtype)
            delta_p_norm = torch.zeros((B, L), device=top.device, dtype=top.dtype)

        reg_loss = u_s.new_zeros((), dtype=torch.float32)

        attn_s = self.legacy_attn_s(u_s, content_mask)
        s_parts = [h_cls, attn_s]
        if self.ab.use_bipolar_pool and self.bipolar_pool is not None:
            bp = self.bipolar_pool(u_s, content_mask)
            z_disp = bp["z_disp"]
            h_vag = bp["h_vag"]
            reg_loss = reg_loss + self.ab.w_orth * bp["loss_orth"].float()
            alpha_pos, alpha_neg, b_coact = bp["alpha_pos"], bp["alpha_neg"], bp["b_coact"]
        else:
            bp = None
            z_disp = torch.zeros((B, self.d_disp), device=u_s.device, dtype=u_s.dtype)
            h_vag = torch.zeros((B, self.d_vag), device=u_s.device, dtype=u_s.dtype)
            alpha_pos = torch.zeros((B, L, 1), device=u_s.device, dtype=u_s.dtype)
            alpha_neg = alpha_pos
            b_coact = torch.zeros((B, 1), device=u_s.device, dtype=u_s.dtype)

        st_emb = self.struct_proj(struct_feats.to(u_s.dtype)) if (self.ab.use_struct and self.struct_proj is not None and struct_feats is not None) else None
        if st_emb is not None:
            s_parts.append(st_emb)
        z_s = self.content_mlp(torch.cat(s_parts, dim=-1))
        if bp is not None and self.bipolar_s_proj is not None and self.bipolar_res_gate is not None:
            bp_res = torch.tanh(self.bipolar_s_proj(torch.cat([bp["z_center"], bp["z_ctx"]], dim=-1)))
            z_s = z_s + self.bipolar_res_gate * bp_res

        # Pragmatic branch
        if self.ab.use_multilayer_conv:
            prag_in = torch.cat([u_p, u_p * (1.0 + torch.tanh(delta_p_norm).unsqueeze(-1))], dim=-1)
        else:
            prag_in = u_p
        prag_logits = self.prag_attn(prag_in).squeeze(-1)

        if self.dual_segment and seg_ids is not None:
            m_ctx = attention_mask.bool() & (seg_ids == 0)
            m_rep = attention_mask.bool() & (seg_ids == 1)
            # Fall back cleanly if context missing
            m_ctx = torch.where(m_ctx.any(dim=1, keepdim=True), m_ctx, attention_mask.bool())
            m_rep = torch.where(m_rep.any(dim=1, keepdim=True), m_rep, attention_mask.bool())
            a_ctx = _masked_softmax(prag_logits, m_ctx, dim=1)
            a_rep = _masked_softmax(prag_logits, m_rep, dim=1)
            c_ctx = (u_p * a_ctx.unsqueeze(-1)).sum(dim=1)
            c_rep = (u_p * a_rep.unsqueeze(-1)).sum(dim=1)
            c_p_list = [c_rep, (c_ctx - c_rep).abs(), c_ctx * c_rep]
            alpha_prag = a_rep.detach()
        else:
            a_prag = _masked_softmax(prag_logits, attention_mask.bool(), dim=1)
            c_p = (u_p * a_prag.unsqueeze(-1)).sum(dim=1)
            c_p_list = [c_p]
            alpha_prag = a_prag.detach()

        if st_emb is not None:
            c_p_list.append(st_emb)
        c_p_cat = torch.cat(c_p_list, dim=-1)

        mu_p = self.prag_mu(c_p_cat)
        if self.ab.use_disentangle and self.use_prag_disentangle:
            logvar_p = torch.clamp(self.prag_logvar(c_p_cat), min=-8.0, max=4.0)
            if self.training:
                std_p = torch.exp(0.5 * logvar_p)
                z_p = mu_p + std_p * torch.randn_like(std_p)
            else:
                z_p = mu_p
            kl_ib = 0.5 * (logvar_p.exp().float() + mu_p.float().pow(2) - 1.0 - logvar_p.float()).mean()
            nhsic = compute_nhsic(z_s, z_p)
            reg_loss = reg_loss + self.ab.w_ib * kl_ib + self.ab.w_indep * nhsic
        else:
            z_p = torch.tanh(mu_p)

        out = {
            "z_s": z_s,
            "z_disp": z_disp if self.ab.use_tension_route else torch.zeros_like(z_disp),
            "h_vag": h_vag if self.ab.use_tension_route else torch.zeros_like(h_vag),
            "z_p": z_p,
            "u_s": u_s,
            "content_mask": content_mask,
            "reg_loss": reg_loss,
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "b_coact": b_coact,
            "g_mod_norm": g_mod_norm.detach(),
            "prag_pivot_score": (alpha_prag * delta_p_norm.detach()),
        }

        if self.num_query_heads > 1:
            # Per-emotion attention query pooling over u_s -> (B, E, work_dim)
            q_logits = torch.einsum("bld,ed->ble", u_s, self.q_heads)
            q_w = _masked_softmax(q_logits, content_mask.unsqueeze(-1).expand(-1, -1, self.num_query_heads), dim=1)
            out["emo_feats"] = torch.einsum("ble,bld->bed", q_w, u_s)

        return out
