"""Mathematically identifiable distributional, polarization-aware, and transport losses."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ordinal_grid(
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return grid values (K,) and interior thresholds (K-1,) for ordered bins."""
    v = torch.tensor(score_values, device=device, dtype=dtype)
    tau_int = 0.5 * (v[:-1] + v[1:])
    return v, tau_int


def _probs_to_moments(
    probs: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute interior CDF, discrete mean, discrete variance, and discrete std."""
    probs_f = probs.float()
    v_f = v.float()
    cdf_int = torch.cumsum(probs_f, dim=-1)[..., :-1]
    view_shape = (1,) * (probs_f.ndim - 1) + (-1,)
    v_b = v_f.view(view_shape)
    disc_mean = (probs_f * v_b).sum(dim=-1)
    disc_var = torch.clamp((probs_f * (v_b - disc_mean.unsqueeze(-1)).pow(2)).sum(dim=-1), min=0.0)
    disc_std = torch.sqrt(disc_var + 1e-8)
    return cdf_int, disc_mean, disc_var, disc_std


def _logistic_discretized_probs(
    loc: torch.Tensor,
    scale: torch.Tensor,
    tau_int: torch.Tensor,
) -> torch.Tensor:
    z = (tau_int.view(*([1] * loc.ndim), -1) - loc.unsqueeze(-1)) / torch.clamp(scale.unsqueeze(-1), min=1e-4)
    cdf_int = torch.sigmoid(z)
    ones = torch.ones_like(cdf_int[..., :1])
    zeros = torch.zeros_like(cdf_int[..., :1])
    cdf_full = torch.cat([zeros, cdf_int, ones], dim=-1)
    return torch.clamp(cdf_full[..., 1:] - cdf_full[..., :-1], min=1e-7)


def gaussian_discretized_probs(
    mu: torch.Tensor,
    std: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Dict[str, torch.Tensor]:
    """Discretized Gaussian ordinal baseline with exact numerical stability."""
    mu_f, std_f = mu.float(), std.float()
    v, tau_int = build_ordinal_grid(score_values, device=mu_f.device, dtype=torch.float32)
    z = (tau_int.view(*([1] * mu_f.ndim), -1) - mu_f.unsqueeze(-1)) / (
        torch.clamp(std_f.unsqueeze(-1), min=1e-4) * math.sqrt(2.0)
    )
    cdf_int = 0.5 * (1.0 + torch.erf(z))
    ones = torch.ones_like(cdf_int[..., :1])
    zeros = torch.zeros_like(cdf_int[..., :1])
    cdf_full = torch.cat([zeros, cdf_int, ones], dim=-1)
    probs = torch.clamp(cdf_full[..., 1:] - cdf_full[..., :-1], min=1e-7)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    cdf_d, d_mean, d_var, d_std = _probs_to_moments(probs, v)
    return {
        "probs": probs,
        "cdf_int": cdf_d,
        "disc_mean": d_mean,
        "disc_var": d_var,
        "disc_std": d_std,
        "u_amb": d_var,
        "u_plur": torch.zeros_like(d_var),
    }


def beta_discretized_probs(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    pts_per_bin: int = 16,
) -> Dict[str, torch.Tensor]:
    """Differentiable discretized Beta(alpha, beta) over bounded ordinal bins (supports U-shapes)."""
    a = torch.clamp(alpha.float(), min=0.15, max=80.0)
    b = torch.clamp(beta.float(), min=0.15, max=80.0)
    v, tau_int = build_ordinal_grid(score_values, device=a.device, dtype=torch.float32)
    v_min = float(score_values[0])
    v_max = float(score_values[-1])
    span = max(v_max - v_min, 1e-4)
    # Map ordinal bin boundaries symmetrically onto [0, 1] with half-step margin
    ext_lo = v_min - 0.5 * (v[1] - v[0]).item()
    ext_hi = v_max + 0.5 * (v[-1] - v[-2]).item()
    ext_span = max(ext_hi - ext_lo, span)
    edges = torch.cat([v.new_tensor([ext_lo]), tau_int, v.new_tensor([ext_hi])])
    u_edges = torch.clamp((edges - ext_lo) / ext_span, min=1e-4, max=1.0 - 1e-4)  # (K+1,)

    k_bins = len(score_values)
    # Midpoint quadrature inside each bin interval [u_k, u_{k+1}]
    t_rel = (torch.arange(pts_per_bin, device=a.device, dtype=torch.float32) + 0.5) / float(pts_per_bin)
    u_lo = u_edges[:-1].unsqueeze(-1)  # (K, 1)
    u_hi = u_edges[1:].unsqueeze(-1)   # (K, 1)
    widths = (u_hi - u_lo)             # (K, 1)
    grid = u_lo + widths * t_rel.unsqueeze(0)  # (K, M)

    log_x = torch.log(grid).view(*([1] * a.ndim), k_bins, pts_per_bin)
    log_1mx = torch.log(1.0 - grid).view(*([1] * a.ndim), k_bins, pts_per_bin)
    log_w = torch.log(widths / float(pts_per_bin)).view(*([1] * a.ndim), k_bins, 1)
    log_dens = (a.unsqueeze(-1).unsqueeze(-1) - 1.0) * log_x + (b.unsqueeze(-1).unsqueeze(-1) - 1.0) * log_1mx + log_w
    bin_log_mass = torch.logsumexp(log_dens, dim=-1)  # exact up to B(a,b) normalizing constant
    probs = torch.softmax(bin_log_mass, dim=-1)
    cdf_d, d_mean, d_var, d_std = _probs_to_moments(probs, v)

    mu_unit = a / (a + b)
    cont_var = ((a * b) / ((a + b).pow(2) * (a + b + 1.0))) * (ext_span ** 2)
    is_u = ((a < 1.0) & (b < 1.0)).float()
    return {
        "probs": probs,
        "cdf_int": cdf_d,
        "disc_mean": d_mean,
        "disc_var": d_var,
        "disc_std": d_std,
        "alpha": a,
        "beta": b,
        "u_amb": cont_var * (1.0 - 0.5 * is_u),
        "u_plur": torch.clamp(torch.maximum(1.0 - a, 1.0 - b), min=0.0) * cont_var,
    }


def unimodal_softmax_probs(
    logits_raw: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Dict[str, torch.Tensor]:
    """Unimodal-Constrained Softmax (UW-LDL style negative-quadratic parameterization)."""
    v, _ = build_ordinal_grid(score_values, device=logits_raw.device, dtype=torch.float32)
    v_min, v_max = float(score_values[0]), float(score_values[-1])
    span = max(v_max - v_min, 1e-4)
    raw = logits_raw.float()
    mu_u = v_min + span * torch.sigmoid(raw[..., 0])
    conc = F.softplus(raw[..., 1]) + 0.05
    logits = -conc.unsqueeze(-1) * (v.view(*([1] * mu_u.ndim), -1) - mu_u.unsqueeze(-1)).pow(2)
    probs = torch.softmax(logits, dim=-1)
    cdf_d, d_mean, d_var, d_std = _probs_to_moments(probs, v)
    return {
        "probs": probs,
        "cdf_int": cdf_d,
        "disc_mean": d_mean,
        "disc_var": d_var,
        "disc_std": d_std,
        "u_amb": d_var,
        "u_plur": torch.zeros_like(d_var),
    }


def dml_mixture_to_dist(
    raw_mu_mid: torch.Tensor,
    raw_delta: torch.Tensor,
    raw_log_s1: torch.Tensor,
    raw_log_s2: torch.Tensor,
    raw_logit_pi: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    bimodal: bool = True,
    max_scale_ratio: float = 0.125,
    min_scale: float = 0.06,
    max_pi_dev: float | None = None,
    scale_couple: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Map unconstrained parameters to identifiable 2-component Discretized Logistic Mixture.

    Theorem 1 & 2 fixes applied:
    - Upper bound s_hi <= 0.15 * span (default 0.125 -> 0.50 on [1..5], strictly below s_crit^dip = 1.039)
      prevents single-component U-inversion while allowing unimodal dispersion up to disc_std ~ 0.91
      without forcing spurious delta/u_plur on non-polarized high-variance texts.
    - Optional max_pi_dev & scale_couple enforce balanced pluralistic camp shares (preventing one component
      from parking inside neutral Bin 3 as a unimodal skew core) while leaving default None behavior
      bit-for-bit identical for cross-domain callers.
    - Tracks dead-zone saturation penalty below raw_delta = -3.5 without inflating unimodal floor.
    """
    v, tau_int = build_ordinal_grid(score_values, device=raw_mu_mid.device, dtype=torch.float32)
    v_min, v_max = float(score_values[0]), float(score_values[-1])
    span = v_max - v_min

    # Give 0.5-bin headroom past [v_min, v_max] so extreme unanimous ratings stay sharp without s -> 0.
    mu_mid = (v_min - 0.5) + (span + 1.0) * torch.sigmoid(raw_mu_mid.float())
    s_min = float(min_scale)
    s_hi = max(s_min + 0.05, max_scale_ratio * span)
    if scale_couple is not None:
        s_bar = 0.5 * (raw_log_s1.float() + raw_log_s2.float())
        s_diff = float(scale_couple) * torch.tanh(0.5 * (raw_log_s1.float() - raw_log_s2.float()))
        ls1, ls2 = s_bar + s_diff, s_bar - s_diff
    else:
        ls1, ls2 = raw_log_s1.float(), raw_log_s2.float()
    s1 = s_min + (s_hi - s_min) * torch.sigmoid(ls1 - 0.5)
    s2 = s_min + (s_hi - s_min) * torch.sigmoid(ls2 - 0.5)

    if bimodal:
        delta = span * torch.sigmoid(raw_delta.float())
        if max_pi_dev is not None and float(max_pi_dev) < 0.5:
            d_pi = max(0.0, float(max_pi_dev))
            pi = (0.5 - d_pi) + (2.0 * d_pi) * torch.sigmoid(raw_logit_pi.float())
        else:
            pi = torch.sigmoid(raw_logit_pi.float())
    else:
        delta = torch.zeros_like(mu_mid)
        pi = torch.full_like(mu_mid, 0.5)

    mu1 = torch.clamp(mu_mid - pi * delta, min=v_min - 0.5, max=v_max + 0.5)
    mu2 = torch.clamp(mu_mid + (1.0 - pi) * delta, min=v_min - 0.5, max=v_max + 0.5)

    p1 = _logistic_discretized_probs(mu1, s1, tau_int)
    p2 = _logistic_discretized_probs(mu2, s2, tau_int)
    probs = (1.0 - pi.unsqueeze(-1)) * p1 + pi.unsqueeze(-1) * p2
    probs = probs / probs.sum(dim=-1, keepdim=True)

    cdf_int, disc_mean, disc_var, disc_std = _probs_to_moments(probs, v)

    logi_var1 = (math.pi ** 2 / 3.0) * s1.pow(2)
    logi_var2 = (math.pi ** 2 / 3.0) * s2.pow(2)
    u_amb = (1.0 - pi) * logi_var1 + pi * logi_var2
    u_plur = pi * (1.0 - pi) * (mu2 - mu1).pow(2)

    return {
        "probs": probs,
        "cdf_int": cdf_int,
        "mu_mid": mu_mid,
        "mu1": mu1,
        "mu2": mu2,
        "delta": delta,
        "raw_delta_pen": F.relu(-3.5 - raw_delta.float()).pow(2) if bimodal else torch.zeros_like(mu_mid),
        "s1": s1,
        "s2": s2,
        "pi": pi,
        "u_amb": u_amb,
        "u_plur": u_plur,
        "disc_mean": disc_mean,
        "disc_var": disc_var,
        "disc_std": disc_std,
    }


def dml_ltv_mixture_to_dist(
    raw_mu_mid: torch.Tensor,
    raw_rho: torch.Tensor,
    raw_std_tot: torch.Tensor,
    raw_s_asym: torch.Tensor,
    raw_logit_pi: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    bimodal: bool = True,
    max_scale_ratio: float = 0.114,
    min_scale: float = 0.055,
    max_pi_dev: float | None = 0.05,
    max_scale_asym: float = 0.06,
    rho_min: float = 0.033,
    rho_max: float = 0.82,
    amb_cap: float = 0.51,
    amb_exp_beta: float = 2.66,
) -> Dict[str, torch.Tensor]:
    """Exponential-Ceiling LTV 2-component DML with Exact Directed Non-Adjacent Variance Decomposition.

    Preserves Theorem 1 & 2 identifiability bounds (s_hi <= 0.46 < s_crit^dip = 1.039) and decomposes
    observed ordinal discrete variance disc_var(P) = u_amb + u_plur into directed non-adjacent cross-camp
    opposition u_plur (identically zero for unimodal & adjacent-bin consensus) and within-camp/adjacent u_amb.
    """
    v, tau_int = build_ordinal_grid(score_values, device=raw_mu_mid.device, dtype=torch.float32)
    v_min, v_max = float(score_values[0]), float(score_values[-1])
    span = v_max - v_min
    c_logi = (math.pi ** 2) / 3.0

    mu_mid = (v_min - 0.5) + (span + 1.0) * torch.sigmoid(raw_mu_mid.float())
    s_min = float(min_scale)
    s_hi = max(s_min + 0.05, float(max_scale_ratio) * span)

    sigma_tot = 0.125 + (1.55 - 0.125) * torch.sigmoid(raw_std_tot.float())
    v_tot = sigma_tot.pow(2)

    if bimodal:
        rho = float(rho_min) + float(rho_max - rho_min) * torch.sigmoid(raw_rho.float())
        if max_pi_dev is not None and float(max_pi_dev) < 0.5:
            d_pi = max(0.0, float(max_pi_dev))
            pi = (0.5 - d_pi) + (2.0 * d_pi) * torch.sigmoid(raw_logit_pi.float())
        else:
            pi = torch.sigmoid(raw_logit_pi.float())
        kappa = torch.exp(float(max_scale_asym) * torch.tanh(raw_s_asym.float()))
        c_eff = torch.clamp(float(amb_cap) * torch.exp(-float(amb_exp_beta) * rho), min=0.130)
        raw_amb = (1.0 - rho) * v_tot
        u_amb_star = torch.clamp(
            raw_amb - (F.softplus(12.0 * (raw_amb - c_eff)) - F.softplus(-12.0 * c_eff)) / 12.0,
            min=1e-5,
        )
        u_plur_star = torch.clamp(v_tot - u_amb_star, min=1e-5)
        delta = torch.clamp(torch.sqrt(u_plur_star / (pi * (1.0 - pi) + 1e-8)), min=0.0, max=span)
    else:
        rho = torch.zeros_like(mu_mid)
        pi = torch.full_like(mu_mid, 0.5)
        kappa = torch.ones_like(mu_mid)
        u_plur_star = torch.zeros_like(v_tot)
        u_amb_star = v_tot
        delta = torch.zeros_like(mu_mid)

    mu1 = torch.clamp(mu_mid - pi * delta, min=v_min - 0.5, max=v_max + 0.5)
    mu2 = torch.clamp(mu_mid + (1.0 - pi) * delta, min=v_min - 0.5, max=v_max + 0.5)

    asym_denom = c_logi * ((1.0 - pi) * kappa.pow(2) + pi / torch.clamp(kappa.pow(2), min=1e-6)) + 1e-8
    s_base = torch.sqrt(u_amb_star / asym_denom)
    s1 = torch.clamp(kappa * s_base, min=s_min, max=s_hi)
    s2 = torch.clamp(s_base / torch.clamp(kappa, min=1e-6), min=s_min, max=s_hi)

    p1_raw = _logistic_discretized_probs(mu1, s1, tau_int).float()
    p2_raw = _logistic_discretized_probs(mu2, s2, tau_int).float()
    p1 = p1_raw / p1_raw.sum(dim=-1, keepdim=True)
    p2 = p2_raw / p2_raw.sum(dim=-1, keepdim=True)
    pi_f = pi.float()
    probs = (1.0 - pi_f.unsqueeze(-1)) * p1 + pi_f.unsqueeze(-1) * p2
    probs = probs / probs.sum(dim=-1, keepdim=True)

    cdf_int, disc_mean, disc_var, disc_std = _probs_to_moments(probs, v)
    u_amb_cont = (1.0 - pi_f) * (c_logi * s1.float().pow(2)) + pi_f * (c_logi * s2.float().pow(2))

    if bimodal:
        u_plur_cont = pi_f * (1.0 - pi_f) * (mu2.float() - mu1.float()).pow(2)
        k_bins = len(score_values)
        idx = torch.arange(k_bins, device=probs.device)
        mask_nonadj = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= 2
        diff_v = v.float().unsqueeze(0) - v.float().unsqueeze(1)
        weight_ij = torch.where(mask_nonadj, diff_v.pow(2), torch.zeros_like(diff_v))
        cross_pos = F.relu(p1.unsqueeze(-1) * p2.unsqueeze(-2) - p1.unsqueeze(-2) * p2.unsqueeze(-1))
        u_plur = torch.minimum(pi_f * (1.0 - pi_f) * (cross_pos * weight_ij).sum(dim=(-2, -1)), disc_var)
        u_amb = torch.clamp(disc_var - u_plur, min=0.0)

        # Orthogonal rho-only discrete cross-camp variance (d/d(mu_mid)==0, d/d(v_tot)==0, d/d(s1,s2)==0)
        v_tot_d = v_tot.detach()
        pi_d = pi_f.detach()
        raw_amb_r = (1.0 - rho) * v_tot_d
        u_amb_star_r = torch.clamp(
            raw_amb_r - (F.softplus(12.0 * (raw_amb_r - c_eff)) - F.softplus(-12.0 * c_eff)) / 12.0,
            min=1e-5,
        )
        delta_r = torch.clamp(
            torch.sqrt(torch.clamp(v_tot_d - u_amb_star_r, min=1e-5) / (pi_d * (1.0 - pi_d) + 1e-8)),
            min=0.0,
            max=span,
        )
        mu1_r = torch.clamp(mu_mid.detach() - pi_d * delta_r, min=v_min - 0.5, max=v_max + 0.5)
        mu2_r = torch.clamp(mu_mid.detach() + (1.0 - pi_d) * delta_r, min=v_min - 0.5, max=v_max + 0.5)
        p1_r_raw = _logistic_discretized_probs(mu1_r, s1.detach(), tau_int).float()
        p2_r_raw = _logistic_discretized_probs(mu2_r, s2.detach(), tau_int).float()
        p1_r = p1_r_raw / p1_r_raw.sum(dim=-1, keepdim=True)
        p2_r = p2_r_raw / p2_r_raw.sum(dim=-1, keepdim=True)
        cross_pos_r = F.relu(p1_r.unsqueeze(-1) * p2_r.unsqueeze(-2) - p1_r.unsqueeze(-2) * p2_r.unsqueeze(-1))
        u_plur_rho = torch.minimum(pi_d * (1.0 - pi_d) * (cross_pos_r * weight_ij).sum(dim=(-2, -1)), disc_var.detach())
    else:
        u_plur_cont = torch.zeros_like(disc_var)
        u_plur = torch.zeros_like(disc_var)
        u_plur_rho = torch.zeros_like(disc_var)
        u_amb = disc_var

    return {
        "probs": probs,
        "cdf_int": cdf_int,
        "mu_mid": mu_mid,
        "mu1": mu1,
        "mu2": mu2,
        "delta": delta,
        "rho_plur": rho,
        "v_tot": v_tot,
        "raw_delta_pen": F.relu(-3.5 - raw_rho.float()).pow(2) if bimodal else torch.zeros_like(mu_mid),
        "s1": s1,
        "s2": s2,
        "pi": pi,
        "u_amb": u_amb,
        "u_amb_cont": u_amb_cont,
        "u_plur": u_plur,
        "u_plur_cont": u_plur_cont,
        "u_plur_rho": u_plur_rho,
        "disc_mean": disc_mean,
        "disc_var": disc_var,
        "disc_std": disc_std,
    }


def wasserstein1_from_cdfs(
    pred_cdf_int: torch.Tensor,
    target_probs: torch.Tensor,
    bin_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact differentiable 1D Wasserstein-1 distance between predicted interior CDF and target probs."""
    tgt_cdf_int = torch.cumsum(target_probs.float(), dim=-1)[..., :-1]
    diff = torch.abs(pred_cdf_int.float() - tgt_cdf_int)
    if bin_steps is not None:
        diff = diff * bin_steps.float()
    return diff.sum(dim=-1)


def kl_divergence_hist(
    target_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Forward KL(target || pred) = sum p log(p / q) safely handling zero bins."""
    t_f = target_probs.float()
    q_f = pred_probs.float()
    p = torch.clamp(t_f, min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = torch.clamp(q_f, min=eps)
    q = q / q.sum(dim=-1, keepdim=True)
    return (t_f * (torch.log(p) - torch.log(q))).sum(dim=-1)


def pairwise_divisiveness(
    probs: torch.Tensor,
    v: torch.Tensor,
    gap_gated: bool = False,
) -> torch.Tensor:
    """Pairwise divisiveness 0.5 * sum_{i,j} p_i p_j psi(|v_i - v_j|).

    When gap_gated=True, psi(d) = max(0, d - 1)^2 zeroes out adjacent-bin jitter (d=1).
    When gap_gated=False, psi(d) = d matches ICML 2026 DC-LDL linear divisiveness.
    """
    p = probs.float()
    dist = torch.abs(v.float().unsqueeze(0) - v.float().unsqueeze(1))
    psi = torch.clamp(dist - 1.0, min=0.0).pow(2) if gap_gated else dist
    return 0.5 * torch.einsum("...i,ij,...j->...", p, psi, p)


def compute_target_pol_gates(
    tgt_probs: torch.Tensor,
    tgt_std: torch.Tensor,
    has_hist: torch.Tensor,
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute target-only (w_pol, w_uni) identifiability gates (Theorem 3 & Lemma)."""
    p = tgt_probs.float()
    s = tgt_std.float()
    v, _ = build_ordinal_grid(score_values, device=p.device, dtype=torch.float32)

    if len(score_values) == 5:
        p_low = p[..., 0] + p[..., 1]
        p_mid = p[..., 2]
        p_high = p[..., 3] + p[..., 4]
    else:
        mid = len(score_values) // 2
        p_low = p[..., :mid].sum(dim=-1)
        p_mid = p[..., mid] if len(score_values) % 2 == 1 else torch.zeros_like(p_low)
        p_high = p[..., mid + (len(score_values) % 2) :].sum(dim=-1)

    d_psi2 = pairwise_divisiveness(p, v, gap_gated=True)
    tail_min = torch.minimum(p_low, p_high)
    dip_or_spread = torch.maximum(tail_min - p_mid, s - 0.72)

    # Center d_psi2 at 0.048 (above bell-trap [0,0.2,0.6,0.2,0]=0.040, capturing moderate 2-4 splits >= 0.0625)
    gamma_hist = (
        torch.sigmoid((tail_min - 0.18) / 0.025)
        * torch.sigmoid((dip_or_spread + 0.01) / 0.03)
        * torch.sigmoid((d_psi2 - 0.048) / 0.02)
    )
    gamma_uni_mom = torch.sigmoid((0.69 - s) / 0.04)
    gamma_pol_mom = torch.sigmoid((s - 1.05) / 0.06)

    mask = has_hist.bool().unsqueeze(-1)
    w_pol = torch.where(mask, gamma_hist, gamma_pol_mom)
    w_uni = torch.where(mask, 1.0 - gamma_hist, gamma_uni_mom)
    return w_pol.detach(), w_uni.detach()


def patnet_emobank_loss(
    out: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite dual-perspective loss with prevalence-normalized masks and Theorem 3 identifiability."""
    v, _ = build_ordinal_grid(score_values, device=batch["r_probs"].device, dtype=torch.float32)
    bin_steps = v[1:] - v[:-1]
    B = max(int(batch["r_probs"].shape[0]), 1)
    C = max(int(batch["r_probs"].shape[1]), 1)

    rho_map = {
        "w": max(float(weights.get("w_hist_prev", 0.417)), 0.05),
        "r": max(float(weights.get("r_hist_prev", 1.0)), 0.05),
    }
    rho_wr = max(float(weights.get("wr_hist_prev", 0.417)), 0.05)
    alpha_pol = float(weights.get("pol_focus", 1.0))

    total_dist = batch["r_probs"].new_zeros((), dtype=torch.float32)
    total_shape = batch["r_probs"].new_zeros((), dtype=torch.float32)
    total_mom = batch["r_probs"].new_zeros((), dtype=torch.float32)
    total_pars = batch["r_probs"].new_zeros((), dtype=torch.float32)

    # Check whether this head actually optimizes bimodal separation parameters
    has_dml_params = ("delta" in out["r"]) and bool(weights.get("use_parsimony", True))

    for persp in ("w", "r"):
        pred = out[persp]
        tgt_p = batch[f"{persp}_probs"].float()
        tgt_m = batch[f"{persp}_mean"].float()
        tgt_s = batch[f"{persp}_std"].float()
        has_h = batch[f"{persp}_has_hist"].bool()
        m_mask = has_h.unsqueeze(-1).float()
        denom_hist = float(B * C) * rho_map[persp]

        w_pol, w_uni = compute_target_pol_gates(tgt_p, tgt_s, has_h, score_values=score_values)
        kl_imp = 0.955 + 0.90 * alpha_pol * w_pol
        w1_imp = 0.574 * torch.clamp(1.0 - 0.55 * alpha_pol * w_pol, min=0.28)

        kl = kl_divergence_hist(tgt_p, pred["probs"])
        w1 = wasserstein1_from_cdfs(pred["cdf_int"], tgt_p, bin_steps=bin_steps)
        dist_per = float(weights.get("kl", 1.0)) * kl * kl_imp + float(weights.get("w1", 1.0)) * w1 * w1_imp
        total_dist = total_dist + (dist_per * m_mask).sum() / denom_hist

        # Divisiveness consistency (linear D1 + gap-gated D_psi2) + 3-region anti-mode mass loss
        if len(score_values) == 5 and float(weights.get("pol_shape", 0.8)) > 0:
            d_hat = pairwise_divisiveness(pred["probs"], v, gap_gated=True)
            d_tgt = pairwise_divisiveness(tgt_p, v, gap_gated=True)
            d1_hat = pairwise_divisiveness(pred["probs"], v, gap_gated=False)
            d1_tgt = pairwise_divisiveness(tgt_p, v, gap_gated=False)
            loss_div = 0.85 * (d_hat - d_tgt).pow(2) + 1.15 * (d1_hat - d1_tgt).pow(2)

            p_hat = pred["probs"].float()
            pl_h, pm_h, pr_h = p_hat[..., 0] + p_hat[..., 1], p_hat[..., 2], p_hat[..., 3] + p_hat[..., 4]
            pl_t, pm_t, pr_t = tgt_p[..., 0] + tgt_p[..., 1], tgt_p[..., 2], tgt_p[..., 3] + tgt_p[..., 4]
            reg_l2 = (pl_h - pl_t).pow(2) + (pm_h - pm_t).pow(2) + (pr_h - pr_t).pow(2)
            # Softmin over opposite tails directly matches bipolar anti-mode tension
            sm_h = -0.05 * torch.logsumexp(torch.stack([-20.0 * pl_h, -20.0 * pr_h], dim=-1), dim=-1)
            sm_t = -0.05 * torch.logsumexp(torch.stack([-20.0 * pl_t, -20.0 * pr_t], dim=-1), dim=-1)
            shape_per = loss_div + 0.75 * reg_l2 + (sm_h - sm_t).pow(2)
            shape_imp = 1.30 + 0.35 * alpha_pol * w_pol
            total_shape = total_shape + float(weights.get("pol_shape", 0.8)) * (shape_per * shape_imp * m_mask).sum() / denom_hist

        # Full unweighted conditional moment supervision across ALL sentences (eliminates cross-fold calibration drift)
        m_err = 0.35 * F.smooth_l1_loss(pred["disc_mean"].float(), tgt_m, beta=0.18) + 0.65 * F.mse_loss(pred["disc_mean"].float(), tgt_m)
        s_err = (pred["disc_std"].float() - tgt_s).pow(2).mean()
        v_err = (pred["disc_var"].float() - tgt_s.pow(2)).pow(2).mean()
        cal_m = (pred["disc_mean"].float().mean(dim=0) - tgt_m.mean(dim=0)).pow(2).mean()
        cal_s = (pred["disc_std"].float().mean(dim=0) - tgt_s.mean(dim=0)).pow(2).mean()
        total_mom = total_mom + (6.90 * m_err + 0.88 * s_err + 0.45 * v_err + 0.88 * cal_m + 0.125 * cal_s)

        # Theorem 3: Orthogonal Variance Allocation (v54 V/D/pear + crisp NVH<=0.33 + sharp orthogonal split_head pull)
        if has_dml_params and "delta" in pred and pred["delta"].requires_grad:
            la = float(weights.get("pars_a", 0.45))
            lrep = float(weights.get("pars_rep", 0.60))
            dmin = float(weights.get("delta_min", 1.5))

            u_plur_cont = pred.get("u_plur_cont", pred["u_plur"]).float()
            u_plur_rho = pred.get("u_plur_rho", u_plur_cont).float()
            u_amb_c = pred.get("u_amb_cont", pred["u_amb"]).float()
            delta = pred["delta"].float()
            pi = pred["pi"].float()
            rho = pred["rho_plur"].float() if "rho_plur" in pred else None
            tgt_var = tgt_s.pow(2)

            # Target gate separating genuine polarized rows (g_pol -> 1) from unimodal / diffuse dispersion (g_pol -> 0)
            g_pol = torch.clamp((w_pol - 0.18) / 0.28, min=0.0, max=1.0).pow(1.15)
            # For missing-histogram Writer rows with high std (>=0.40), shape is unobserved -> mask negative identifiability loss
            obs_neg = (has_h.unsqueeze(-1) | (tgt_s < 0.40)).float()
            low_gate = torch.sigmoid((0.37 - tgt_s) / 0.06)
            high_gate = torch.sigmoid((tgt_s - 0.56) / 0.04)

            w_una = (1.0 - g_pol) * obs_neg * low_gate
            w_rlow = (1.0 - g_pol) * obs_neg * (1.0 - low_gate) * (1.0 - high_gate)
            w_nvh = (1.0 - g_pol) * obs_neg * (1.0 - low_gate) * high_gate
            w_vod = g_pol

            def _regime_mean(weights: torch.Tensor, per_item: torch.Tensor, active_thr: float, min_denom: float):
                dim_sum = weights.sum(dim=0)
                active = dim_sum > active_thr
                if bool(active.any()):
                    term = ((weights * per_item).sum(dim=0) / torch.clamp(dim_sum, min=min_denom))[active].mean()
                else:
                    term = u_plur_cont.new_zeros(())
                return dim_sum, term

            # 1) Unanimous regime (tgt_std < 0.37): strong shift-invariant u_plur + tight rho floor eliminate A/D una bloat.
            u_amb_tgt_una = torch.clamp(0.40 * tgt_var, min=0.008, max=0.042)
            up_pen_una_c = F.smooth_l1_loss(F.relu(u_plur_cont - 0.004), torch.zeros_like(u_plur_cont), beta=0.011, reduction="none")
            up_pen_una_r = F.smooth_l1_loss(F.relu(u_plur_rho - 0.002), torch.zeros_like(u_plur_rho), beta=0.012, reduction="none") + 0.30 * F.relu(u_plur_rho - 0.010)
            rho_pen_una = (
                F.smooth_l1_loss(F.relu(rho - 0.038), torch.zeros_like(rho), beta=0.022, reduction="none")
                if rho is not None
                else torch.zeros_like(u_plur_cont)
            )
            una_per_item = (
                1.97 * up_pen_una_c
                + 1.77 * up_pen_una_r
                + 1.62 * rho_pen_una
                + 1.42 * F.smooth_l1_loss(u_amb_c, u_amb_tgt_una, beta=0.016, reduction="none")
                + 1.15 * F.relu(u_amb_c - 0.042)
                + 0.21 * (pi - 0.5).pow(2)
            )
            una_dim_sum, una_term = _regime_mean(w_una, una_per_item, active_thr=0.50, min_denom=4.0)

            # 1b) Low-variance unimodal skew regime (0.37 <= tgt_std < 0.56): crisp rho/up control cures D r_low residual bloat.
            u_amb_tgt_rlow = torch.clamp(0.57 * tgt_var, min=0.055, max=0.178)
            up_pen_rlow_c = F.smooth_l1_loss(F.relu(u_plur_cont - 0.004), torch.zeros_like(u_plur_cont), beta=0.011, reduction="none")
            up_pen_rlow_r = F.smooth_l1_loss(F.relu(u_plur_rho - 0.002), torch.zeros_like(u_plur_rho), beta=0.012, reduction="none") + 0.35 * F.relu(u_plur_rho - 0.009)
            rho_pen_rlow = (
                F.smooth_l1_loss(F.relu(rho - 0.038), torch.zeros_like(rho), beta=0.020, reduction="none")
                if rho is not None
                else torch.zeros_like(u_plur_cont)
            )
            rlow_per_item = (
                1.76 * up_pen_rlow_c
                + 1.76 * up_pen_rlow_r
                + 1.46 * rho_pen_rlow
                + 0.72 * F.smooth_l1_loss(u_amb_c, u_amb_tgt_rlow, beta=0.024, reduction="none")
                + 0.70 * F.relu(u_plur_rho - 0.046 * u_amb_c.detach())
                + 0.22 * (pi - 0.5).pow(2)
            )
            rlow_dim_sum, rlow_term = _regime_mean(w_rlow, rlow_per_item, active_thr=0.50, min_denom=4.0)

            # 2) High-variance unimodal spread regime (tgt_std >= 0.56): crisp NVH u_plur suppression keeps V/A/D up cleanly above std.
            u_amb_tgt_nvh = torch.clamp(0.66 * tgt_var, min=0.068, max=0.305)
            up_pen_nvh_c = F.smooth_l1_loss(F.relu(u_plur_cont - 0.005), torch.zeros_like(u_plur_cont), beta=0.014, reduction="none")
            up_pen_nvh_r = F.smooth_l1_loss(F.relu(u_plur_rho - 0.002), torch.zeros_like(u_plur_rho), beta=0.013, reduction="none")
            rho_pen_nvh = (
                F.smooth_l1_loss(F.relu(rho - 0.041), torch.zeros_like(rho), beta=0.022, reduction="none")
                if rho is not None
                else torch.zeros_like(u_plur_cont)
            )
            nvh_per_item = (
                1.10 * up_pen_nvh_c
                + 1.62 * up_pen_nvh_r
                + 0.98 * rho_pen_nvh
                + 0.76 * F.smooth_l1_loss(u_amb_c, u_amb_tgt_nvh, beta=0.028, reduction="none")
                + 0.72 * F.relu(u_plur_rho - 0.044 * u_amb_c.detach())
                + 0.22 * (pi - 0.5).pow(2)
            )
            nvh_dim_sum, nvh_term = _regime_mean(w_nvh, nvh_per_item, active_thr=0.25, min_denom=6.0)

            # 3) Polarized VoD regime: balanced orthogonal pull with superlinear inverse-prevalence preserves ER0.8 without D spillover.
            u_amb_tgt_pos = torch.clamp(0.202 * tgt_var, min=0.120, max=0.160)
            u_plur_floor_cont = torch.clamp(tgt_var - 0.16, min=0.38, max=1.40)
            u_plur_floor_rho = torch.clamp(0.50 * tgt_var, min=0.230, max=0.75)
            u_plur_ceil_pos = torch.clamp(tgt_var + 0.05, min=0.58, max=1.80)
            up_under_cont = F.smooth_l1_loss(F.relu(u_plur_floor_cont - u_plur_cont), torch.zeros_like(u_plur_cont), beta=0.06, reduction="none")
            up_under_rho = F.smooth_l1_loss(F.relu(u_plur_floor_rho - u_plur_rho), torch.zeros_like(u_plur_rho), beta=0.05, reduction="none")
            up_over_cont = F.smooth_l1_loss(F.relu(u_plur_cont - u_plur_ceil_pos), torch.zeros_like(u_plur_cont), beta=0.06, reduction="none")
            rho_floor_pos = (
                F.smooth_l1_loss(F.relu(0.654 - rho), torch.zeros_like(rho), beta=0.06, reduction="none")
                if rho is not None
                else torch.zeros_like(u_plur_cont)
            )
            dmin_eff = min(dmin, 1.38)
            pi_skew_excess = F.relu(torch.abs(pi - 0.5) - 0.20)
            pos_per_item = (
                0.78 * up_under_cont
                + 1.26 * up_under_rho
                + 0.40 * up_over_cont
                + 1.38 * rho_floor_pos
                + (0.86 + 0.35 * la) * F.smooth_l1_loss(u_amb_c, u_amb_tgt_pos, beta=0.028, reduction="none")
                + 1.14 * F.relu(u_amb_c - 0.174)
                + 0.96 * F.relu(u_amb_c.detach() + 0.075 - u_plur_rho)
                + (0.78 * lrep) * torch.clamp(dmin_eff - delta, min=0.0).pow(2)
                + 0.30 * (pi - 0.5).pow(2)
                + 2.50 * pi_skew_excess.pow(2)
            )
            pos_dim_sum = w_vod.sum(dim=0)
            active_dims = pos_dim_sum > 0.05
            if bool(active_dims.any()):
                dim_obs = torch.clamp(una_dim_sum + rlow_dim_sum + nvh_dim_sum + pos_dim_sum, min=1.0)
                r_pol = pos_dim_sum / dim_obs
                # Superlinear inverse batch-prevalence scaling focuses on genuinely rare-VoD dimensions while preventing D spillover.
                inv_prev = torch.clamp((0.052 / torch.clamp(r_pol, min=0.018)).pow(1.65), min=1.00, max=2.75)
                pos_dim_err = inv_prev * (w_vod * pos_per_item).sum(dim=0) / torch.clamp(pos_dim_sum, min=0.30)
                pos_term = pos_dim_err[active_dims].mean()
            else:
                pos_term = u_plur_cont.new_zeros(())

            raw_pen = pred.get("raw_delta_pen", torch.zeros_like(delta))
            persp_scale = 0.24 if persp == "r" else 0.06
            total_pars = total_pars + persp_scale * (1.64 * una_term + 1.16 * rlow_term + 0.84 * nvh_term + 1.32 * pos_term) + 0.004 * raw_pen.mean()

    # Cross-perspective W->R transport loss
    pred_w_mean = out["w"]["disc_mean"].float()
    pred_r_mean = out["r"]["disc_mean"].float()
    true_shift_mean = batch["r_mean"].float() - batch["w_mean"].float()
    pred_shift_mean = pred_r_mean - pred_w_mean
    loss_shift_mean = F.smooth_l1_loss(pred_shift_mean, true_shift_mean, beta=0.15)

    dual_mask = (batch["w_has_hist"].bool() & batch["r_has_hist"].bool()).unsqueeze(-1).float()
    denom_dual = float(B * C) * rho_wr
    tgt_cdf_w = torch.cumsum(batch["w_probs"].float(), dim=-1)[..., :-1]
    tgt_cdf_r = torch.cumsum(batch["r_probs"].float(), dim=-1)[..., :-1]
    pred_shift_cdf = out["r"]["cdf_int"].float() - out["w"]["cdf_int"].float()
    true_shift_cdf = tgt_cdf_r - tgt_cdf_w
    cdf_diff_l1 = torch.abs(pred_shift_cdf - true_shift_cdf).sum(dim=-1)
    # Also match total W1 shift magnitude between predicted W and R histograms
    pred_wr_w1 = torch.abs(out["r"]["cdf_int"].float() - out["w"]["cdf_int"].float()).sum(dim=-1)
    true_wr_w1 = torch.abs(tgt_cdf_r - tgt_cdf_w).sum(dim=-1)
    wr_mag_l1 = torch.abs(pred_wr_w1 - true_wr_w1)
    loss_shift_cdf = ((cdf_diff_l1 + 0.5 * wr_mag_l1) * dual_mask).sum() / denom_dual

    trans_reg = out.get("transport_l1", torch.zeros_like(loss_shift_mean)).float()
    extra_reg = out.get("reg_loss", torch.zeros_like(loss_shift_mean)).float()

    w_mom = float(weights.get("mom", 0.6))
    w_sm = float(weights.get("shift_mean", 1.5))
    w_sc = float(weights.get("shift_cdf", 0.8))
    w_tr = float(weights.get("trans_reg", 0.002))
    w_ex = float(weights.get("extra_reg", 0.30))

    loss = (
        total_dist
        + total_shape
        + w_mom * total_mom
        + total_pars
        + w_sm * loss_shift_mean
        + w_sc * loss_shift_cdf
        + w_tr * trans_reg
        + w_ex * extra_reg
    )
    stats = {
        "loss": float(loss.detach().item()),
        "dist": float(total_dist.detach().item()),
        "shape": float(total_shape.detach().item()),
        "mom": float(total_mom.detach().item()),
        "pars": float(total_pars.detach().item()),
        "shift_mean": float(loss_shift_mean.detach().item()),
        "shift_cdf": float(loss_shift_cdf.detach().item()),
        "extra_reg": float(extra_reg.detach().item()),
    }
    return loss, stats


def _nig_nll(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    y, gamma, nu, alpha, beta = y.float(), gamma.float(), nu.float(), alpha.float(), beta.float()
    two_b_nu = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / torch.clamp(nu, min=1e-6))
        - alpha * torch.log(torch.clamp(two_b_nu, min=1e-6))
        + (alpha + 0.5) * torch.log(torch.clamp(nu * (y - gamma).pow(2) + two_b_nu, min=1e-6))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def _log_beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def beta_binomial_mixture_nll(
    k: torch.Tensor,
    n: torch.Tensor,
    pi: torch.Tensor,
    alpha1: torch.Tensor,
    beta1: torch.Tensor,
    alpha2: torch.Tensor,
    beta2: torch.Tensor,
) -> torch.Tensor:
    """Exact marginal negative log-likelihood of a 2-component Beta-Binomial mixture."""
    kf = torch.clamp(k.float(), min=0.0)
    nf = torch.clamp(n.float(), min=1.0)
    kf = torch.minimum(kf, nf)
    pi_f = torch.clamp(pi.float(), min=1e-5, max=1.0 - 1e-5)
    a1, b1 = torch.clamp(alpha1.float(), min=0.1), torch.clamp(beta1.float(), min=0.1)
    a2, b2 = torch.clamp(alpha2.float(), min=0.1), torch.clamp(beta2.float(), min=0.1)

    log_comb = torch.lgamma(nf + 1.0) - torch.lgamma(kf + 1.0) - torch.lgamma(nf - kf + 1.0)
    lp1 = torch.log(1.0 - pi_f) + log_comb + _log_beta_fn(a1 + kf, b1 + nf - kf) - _log_beta_fn(a1, b1)
    lp2 = torch.log(pi_f) + log_comb + _log_beta_fn(a2 + kf, b2 + nf - kf) - _log_beta_fn(a2, b2)
    return -torch.logsumexp(torch.stack([lp1, lp2], dim=-1), dim=-1)


def compute_model_loss(
    model_type: str,
    out: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    score_values: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unified fair training loss dispatcher for PAT-Net and all SOTA baselines."""
    if model_type in ("patnet", "patnet_unimodal", "patnet_no_transport", "beta_nll", "uw_ldl"):
        w_adj = dict(weights)
        if model_type in ("patnet_unimodal", "beta_nll", "uw_ldl"):
            w_adj["use_parsimony"] = False
        return patnet_emobank_loss(out, batch, w_adj, score_values=score_values)

    if model_type == "softmax_ldl":
        w_adj = dict(weights)
        w_adj["use_parsimony"] = False
        return patnet_emobank_loss(out, batch, w_adj, score_values=score_values)

    if model_type == "sord":
        # Metric-aware ordinal soft targets (Diaz & Marathe, CVPR 2019) + exact hist KL when present
        v, _ = build_ordinal_grid(score_values, device=batch["r_probs"].device, dtype=torch.float32)
        T = float(weights.get("sord_temp", 0.5))
        loss = batch["r_mean"].new_zeros((), dtype=torch.float32)
        B = max(int(batch["r_probs"].shape[0]), 1)
        C = max(int(batch["r_probs"].shape[1]), 1)
        rho_map = {"w": max(float(weights.get("w_hist_prev", 0.417)), 0.05), "r": 1.0}
        for persp in ("w", "r"):
            tgt_m = batch[f"{persp}_mean"].float()
            tgt_p = batch[f"{persp}_probs"].float()
            has_h = batch[f"{persp}_has_hist"].bool().unsqueeze(-1).float()
            q_sord = torch.softmax(-torch.abs(v.view(1, 1, -1) - tgt_m.unsqueeze(-1)) / T, dim=-1)
            kl_sord = kl_divergence_hist(q_sord, out[persp]["probs"]).mean()
            kl_hist = (kl_divergence_hist(tgt_p, out[persp]["probs"]) * has_h).sum() / (float(B * C) * rho_map[persp])
            loss = loss + 0.5 * kl_sord + kl_hist
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "dldl_v2":
        # Gao et al., IJCAI 2018: KL + expectation L1 (on all rows via disc_mean)
        B = max(int(batch["r_probs"].shape[0]), 1)
        C = max(int(batch["r_probs"].shape[1]), 1)
        rho_map = {"w": max(float(weights.get("w_hist_prev", 0.417)), 0.05), "r": 1.0}
        loss = batch["r_mean"].new_zeros((), dtype=torch.float32)
        for persp in ("w", "r"):
            has_h = batch[f"{persp}_has_hist"].bool().unsqueeze(-1).float()
            kl = (kl_divergence_hist(batch[f"{persp}_probs"], out[persp]["probs"]) * has_h).sum() / (float(B * C) * rho_map[persp])
            l1 = F.smooth_l1_loss(out[persp]["disc_mean"].float(), batch[f"{persp}_mean"].float(), beta=0.1)
            loss = loss + kl + 1.0 * l1
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "dc_ldl":
        # Lu et al., ICML 2026: Divisiveness-Consistent LDL (KL + mean + pairwise divisiveness D1 + var)
        v, _ = build_ordinal_grid(score_values, device=batch["r_probs"].device, dtype=torch.float32)
        B = max(int(batch["r_probs"].shape[0]), 1)
        C = max(int(batch["r_probs"].shape[1]), 1)
        rho_map = {"w": max(float(weights.get("w_hist_prev", 0.417)), 0.05), "r": 1.0}
        loss = batch["r_mean"].new_zeros((), dtype=torch.float32)
        for persp in ("w", "r"):
            pred = out[persp]
            tgt_p = batch[f"{persp}_probs"].float()
            tgt_m = batch[f"{persp}_mean"].float()
            tgt_s = batch[f"{persp}_std"].float()
            has_h = batch[f"{persp}_has_hist"].bool().unsqueeze(-1).float()
            denom = float(B * C) * rho_map[persp]
            kl = (kl_divergence_hist(tgt_p, pred["probs"]) * has_h).sum() / denom
            div_hat = pairwise_divisiveness(pred["probs"], v, gap_gated=False)
            div_tgt = pairwise_divisiveness(tgt_p, v, gap_gated=False)
            div_loss = ((div_hat - div_tgt).pow(2) * has_h).sum() / denom
            m_err = F.smooth_l1_loss(pred["disc_mean"].float(), tgt_m, beta=0.15)
            s_err = (pred["disc_std"].float() - tgt_s).pow(2).mean()
            loss = loss + kl + 1.0 * div_loss + 0.5 * (m_err + s_err)
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "scalar_mse":
        loss_w = F.mse_loss(out["w"]["disc_mean"].float(), batch["w_mean"].float())
        loss_r = F.mse_loss(out["r"]["disc_mean"].float(), batch["r_mean"].float())
        loss = loss_w + loss_r
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "gaussian_nll":
        loss = batch["r_mean"].new_zeros((), dtype=torch.float32)
        for persp in ("w", "r"):
            mu = out[persp]["mu_param"].float()
            std = out[persp]["std_param"].float()
            tgt_m = batch[f"{persp}_mean"].float()
            tgt_v = batch[f"{persp}_std"].float().pow(2)
            nll = torch.log(std) + 0.5 * (tgt_v + (mu - tgt_m).pow(2)) / torch.clamp(std.pow(2), min=1e-4)
            loss = loss + nll.mean()
        return loss, {"loss": float(loss.detach().item())}

    if model_type == "nig_evidential":
        # Fair variance-supervised NIG baseline (two-point moment-matched expected NLL + variance target)
        loss = batch["r_mean"].new_zeros((), dtype=torch.float32)
        for persp in ("w", "r"):
            p = out[persp]
            tgt_m = batch[f"{persp}_mean"].float()
            tgt_s = batch[f"{persp}_std"].float()
            nll_sym = 0.5 * (
                _nig_nll(tgt_m - tgt_s, p["gamma"], p["nu"], p["alpha"], p["beta"])
                + _nig_nll(tgt_m + tgt_s, p["gamma"], p["nu"], p["alpha"], p["beta"])
            )
            e_var = p["beta"].float() / torch.clamp(p["alpha"].float() - 1.0, min=0.05)
            var_match = F.smooth_l1_loss(e_var, tgt_s.pow(2), beta=0.25)
            reg = torch.abs(tgt_m - p["gamma"].float()) * (2.0 * p["nu"].float() + p["alpha"].float())
            loss = loss + (nll_sym + 0.02 * reg).mean() + 0.5 * var_match
        return loss, {"loss": float(loss.detach().item())}

    raise ValueError(f"Unsupported model_type: {model_type}")
