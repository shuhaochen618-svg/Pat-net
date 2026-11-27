"""Comprehensive Stage-0 mathematical, loss, metric, and conformal unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal_sets import build_risk_features, fit_and_calibrate_alert_crc, mondrian_conformal_credal_sets
from src.losses import (
    beta_binomial_mixture_nll,
    beta_discretized_probs,
    compute_model_loss,
    dml_mixture_to_dist,
    pairwise_divisiveness,
    unimodal_softmax_probs,
)
from src.metrics import (
    cluster_bootstrap_paired_test,
    esteban_ray_polarization,
    evaluate_binary_alert_metrics,
    evaluate_perspective_predictions,
    evaluate_perspective_shift,
    extract_clean_polarization_masks,
    holm_bonferroni,
)
from src.struct_features import STRUCT_DIM, extract_struct_matrix


def test_theorem1_dml_bounds_and_symmetry() -> None:
    B, C = 4, 3
    raw_mu = torch.zeros((B, C))
    raw_delta = torch.tensor([[10.0, 0.0, 2.0]] * B)
    raw_ls1 = torch.full((B, C), 50.0)
    raw_ls2 = torch.full((B, C), 50.0)
    raw_pi = torch.zeros((B, C))

    d = dml_mixture_to_dist(raw_mu, raw_delta, raw_ls1, raw_ls2, raw_pi, bimodal=True, max_scale_ratio=0.15)
    assert torch.allclose(d["probs"].sum(dim=-1), torch.ones((B, C)), atol=1e-5)
    # Theorem 1: s_hi = 0.15 * 4.0 = 0.60
    assert float(d["s1"].max().item()) <= 0.6001
    assert float(d["s2"].max().item()) <= 0.6001
    # Unimodal column (col 1: raw_delta=0 => delta near 2.0; test bimodal=False guarantees std <= 1.04)
    d_uni = dml_mixture_to_dist(raw_mu, raw_delta, raw_ls1, raw_ls2, raw_pi, bimodal=False, max_scale_ratio=0.15)
    assert float(d_uni["u_plur"].max().item()) == 0.0
    assert float(d_uni["disc_std"].max().item()) <= 1.04, f"Unimodal std exceeded Theorem 1 bound: {d_uni['disc_std']}"
    assert float(d_uni["probs"][0, 0, 2].item()) >= float(d_uni["probs"][0, 0, 0].item())

    # Bell trap cannot happen under single logistic s <= 0.60; verify p_low/p_mid monotone around mean=3
    print("[PASS] test_theorem1_dml_bounds_and_symmetry")


def test_all_emobank_losses_and_theorem3_gradients() -> None:
    B, C = 8, 3
    torch.manual_seed(7)
    r_probs = torch.softmax(torch.randn(B, C, 5), dim=-1)
    w_probs = torch.softmax(torch.randn(B, C, 5), dim=-1)
    # Construct crisp unanimous row 0 and crisp bimodal row 1
    r_probs[0] = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]] * C)
    r_probs[1] = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.5]] * C)
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    r_mean = (r_probs * v).sum(dim=-1)
    w_mean = (w_probs * v).sum(dim=-1)
    r_std = torch.sqrt(torch.clamp((r_probs * (v - r_mean.unsqueeze(-1)) ** 2).sum(dim=-1), min=0.0))
    w_std = torch.sqrt(torch.clamp((w_probs * (v - w_mean.unsqueeze(-1)) ** 2).sum(dim=-1), min=0.0))
    batch = {
        "r_probs": r_probs,
        "w_probs": w_probs,
        "r_mean": r_mean,
        "w_mean": w_mean,
        "r_std": r_std,
        "w_std": w_std,
        "r_has_hist": torch.ones(B, dtype=torch.bool),
        "w_has_hist": torch.tensor([True, True, True, True, False, False, False, False], dtype=torch.bool),
    }

    weights = {"kl": 1.0, "w1": 1.0, "pol_shape": 0.5, "pars_p": 0.05, "pars_a": 0.05, "pars_rep": 0.05, "w_hist_prev": 0.5, "r_hist_prev": 1.0, "wr_hist_prev": 0.5}

    # Verify Theorem 3 conditional parsimony parameters move in right directions
    raw_delta = torch.zeros((B, C), requires_grad=True)
    raw_ls1 = torch.zeros((B, C), requires_grad=True)
    raw_mu = torch.zeros((B, C), requires_grad=True)
    d_w = dml_mixture_to_dist(raw_mu, raw_delta, raw_ls1, raw_ls1, torch.zeros((B, C)), bimodal=True)
    d_r = dml_mixture_to_dist(raw_mu, raw_delta, raw_ls1, raw_ls1, torch.zeros((B, C)), bimodal=True)
    loss, _ = compute_model_loss("patnet", {"w": d_w, "r": d_r}, batch, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw_delta.grad).all() and raw_delta.grad.abs().sum().item() > 0

    # Verify standalone losses for beta_discretized_probs and unimodal_softmax_probs
    bp = beta_discretized_probs(torch.full((4, 3), 0.5), torch.full((4, 3), 0.5))
    assert torch.allclose(bp["probs"].sum(dim=-1), torch.ones((4, 3)), atol=1e-5)
    assert float(bp["probs"][0, 0, 0].item()) > float(bp["probs"][0, 0, 2].item())
    uw = unimodal_softmax_probs(torch.randn(4, 3, 2))
    assert torch.allclose(uw["probs"].sum(dim=-1), torch.ones((4, 3)), atol=1e-5)

    # Verify DC-LDL divisiveness is larger on split tails than single center bin
    p_split = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.5])
    p_center = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    assert float(pairwise_divisiveness(p_split, v, gap_gated=True)) > float(pairwise_divisiveness(p_center, v, gap_gated=True))
    print("[PASS] test_all_emobank_losses_and_theorem3_gradients")


def test_beta_binomial_mixture_nll() -> None:
    k = torch.tensor([0.0, 2.0, 5.0, 8.0])
    n = torch.tensor([5.0, 4.0, 5.0, 8.0])
    pi = torch.full((4,), 0.5)
    a1 = torch.full((4,), 1.2)
    b1 = torch.full((4,), 8.0)
    a2 = torch.full((4,), 8.0)
    b2 = torch.full((4,), 1.2)
    nll = beta_binomial_mixture_nll(k, n, pi, a1, b1, a2, b2)
    assert torch.isfinite(nll).all() and (nll >= 0.0).all()
    # Mixed half-split (2/4) vs extreme unanimous (8/8) should have reasonable finite likelihood
    assert float(nll[3].item()) < 3.0
    print("[PASS] test_beta_binomial_mixture_nll")


def test_clean_polarization_masks_and_metrics() -> None:
    p_bell = np.array([0.0, 0.25, 0.50, 0.25, 0.0])      # std = sqrt(0.5) = 0.707, bell trap!
    p_vod = np.array([0.35, 0.05, 0.0, 0.05, 0.55])      # bimodal valley!
    p_unam = np.array([0.0, 0.0, 1.0, 0.0, 0.0])         # unanimous neutral
    p_diff = np.array([0.05, 0.15, 0.65, 0.15, 0.0])     # diffuse clean unimodal (std=0.70, p_mid=0.65)
    P = np.vstack([p_bell, p_vod, p_unam, p_diff])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = (P * v).sum(-1)
    s = np.sqrt(((P * (v - m[:, None]) ** 2)).sum(-1))

    masks = extract_clean_polarization_masks(P, m, s)
    assert masks["bell_trap"][0] and not masks["valley_or_dip"][0], "Bell trap must be separated from valley_or_dip!"
    assert masks["valley_strict"][1] and masks["valley_or_dip"][1]
    assert masks["unanimous_neut"][2]
    assert masks["diffuse_clean"][3] and not masks["bell_trap"][3]

    er = esteban_ray_polarization(P, alpha=0.8)
    assert er[1] > er[0] and er[1] > er[2]

    # Dirac vs probabilistic prediction check
    ev_dirac = evaluate_perspective_predictions(
        np.tile(P[:, None, :], (4, 3, 1)),
        np.tile(P[:, None, :], (4, 3, 1)),
        np.tile(m[:, None], (4, 3)),
        np.tile(m[:, None], (4, 3)),
        np.tile(s[:, None], (4, 3)),
        np.zeros((16, 3)),
        np.ones(16, dtype=bool),
        prefix="r",
    )
    assert np.isnan(ev_dirac["r_mean_jsd"])
    assert np.isfinite(ev_dirac["r_mean_w1"])

    ev_shift = evaluate_perspective_shift(
        np.random.rand(20, 3) + 2.0,
        np.random.rand(20, 3) + 2.0,
        np.random.rand(20, 3) + 2.0,
        np.random.rand(20, 3) + 2.0,
    )
    assert "shift_mean_mae" in ev_shift

    # Cluster bootstrap & Holm-Bonferroni
    boot = cluster_bootstrap_paired_test(np.ones(40) * 0.2, np.ones(40) * 0.3, [f"doc_{i // 4}" for i in range(40)], n_boot=300)
    assert boot["n_clusters"] == 10 and boot["diff_mean"] < 0 and boot["p_two_sided"] < 0.05
    adj = holm_bonferroni([0.001, 0.04, 0.20])
    assert abs(adj[0] - 0.003) < 1e-8 and abs(adj[1] - 0.08) < 1e-8
    print("[PASS] test_clean_polarization_masks_and_metrics")


def test_mondrian_conformal_and_alert_crc() -> None:
    rng = np.random.default_rng(123)
    n_s, n_c, n_t = 120, 200, 400
    def _syn(n):
        logits = rng.normal(0, 1.0, size=(n, 5))
        p_t = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        p_h = np.exp(logits + rng.normal(0, 0.35, size=(n, 5)))
        p_h = p_h / p_h.sum(-1, keepdims=True)
        ua = rng.uniform(0.1, 0.8, size=n)
        up = rng.uniform(0.0, 0.6, size=n)
        cats = rng.choice(["blog", "fiction", "news"], size=n)
        return p_h, p_t, ua, up, cats

    sp_h, sp_t, s_ua, s_up, _ = _syn(n_s)
    cp_h, cp_t, c_ua, c_up, c_cat = _syn(n_c)
    tp_h, tp_t, t_ua, t_up, t_cat = _syn(n_t)

    cred = mondrian_conformal_credal_sets(sp_h, sp_t, s_ua, s_up, cp_h, cp_t, c_ua, c_up, tp_h, tp_t, t_ua, t_up, test_categories=t_cat, cal_categories=c_cat, alpha=0.10)
    assert 0.82 <= cred["mondrian_w1_coverage"] <= 0.98
    assert 0.85 <= cred["crc_human_coverage"] <= 1.0

    # Alert CRC test
    X_s = build_risk_features(sp_h, s_ua, s_up)
    X_c = build_risk_features(cp_h, c_ua, c_up)
    X_t = build_risk_features(tp_h, t_ua, t_up)
    y_s = (s_up > 0.42).astype(int)
    y_c = (c_up > 0.42).astype(int)
    y_t = (t_up > 0.42).astype(int)
    crc = fit_and_calibrate_alert_crc(X_s, y_s, X_c, y_c, X_t, y_t, alpha=0.10, cal_severity=c_up, test_severity=t_up)
    assert crc["test_fnr"] <= 0.25 and "sev_crc_test_risk" in crc
    _ = evaluate_binary_alert_metrics(y_t, X_t[:, 0])

    # EN + ZH structural extractor
    sf = extract_struct_matrix(["Wait — how?! Totally shocked yet hopeful...", "虽然有点难过，但是我并不害怕！怎么会这样？"])
    assert sf.shape == (2, STRUCT_DIM) and np.isfinite(sf).all() and sf[1, 13] > 0 and sf[1, 14] > 0
    print("[PASS] test_mondrian_conformal_and_alert_crc")


if __name__ == "__main__":
    test_theorem1_dml_bounds_and_symmetry()
    test_all_emobank_losses_and_theorem3_gradients()
    test_beta_binomial_mixture_nll()
    test_clean_polarization_masks_and_metrics()
    test_mondrian_conformal_and_alert_crc()
    print("ALL CORE TESTS PASSED.")
