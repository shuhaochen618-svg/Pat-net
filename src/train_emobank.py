"""Train and evaluate Dual-Perspective Affect models on EmoBank with clean 4-way splits & Conformal Risk Control."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
def _amp_autocast(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .conformal_sets import (
    build_risk_features,
    fit_and_calibrate_alert_crc,
    mondrian_conformal_credal_sets,
)
from .datasets import (
    EmoBankDistributionDataset,
    compute_train_prevalence,
    fit_struct_scaler_for_splits,
    get_emobank_4way_splits,
    load_emobank_dataframe,
    make_collate_fn,
)
from .losses import compute_model_loss
from .metrics import (
    esteban_ray_polarization,
    evaluate_binary_alert_metrics,
    evaluate_perspective_predictions,
    evaluate_perspective_shift,
    extract_clean_polarization_masks,
)
from .models import DualPerspectiveAffectModel
from .paths import OUT_DIR, resolve_backbone_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(
    model: DualPerspectiveAffectModel,
    loader: DataLoader,
    device: torch.device,
    mc_dropout_T: int = 0,
) -> Dict[str, np.ndarray | List[str]]:
    use_mc = mc_dropout_T > 1
    model.train(mode=use_mc)
    keys_arr = ["probs", "disc_mean", "disc_std", "u_amb", "u_plur"]
    buf: Dict[str, List[np.ndarray]] = {f"{p}_{k}": [] for p in ("w", "r") for k in keys_arr}
    if use_mc:
        buf["r_mc_mi"] = []
        buf["w_mc_mi"] = []
    buf.update(
        {
            "tgt_w_probs": [],
            "tgt_r_probs": [],
            "tgt_w_mean": [],
            "tgt_r_mean": [],
            "tgt_w_std": [],
            "tgt_r_std": [],
            "w_has_hist": [],
            "r_has_hist": [],
        }
    )
    ids: List[str] = []
    cats: List[str] = []
    docs: List[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        st = batch["struct_feats"].to(device, non_blocking=True)
        if not use_mc:
            with _amp_autocast(enabled=(device.type == "cuda")):
                out = model(input_ids, attn, struct_feats=st)
            for p in ("w", "r"):
                for k in keys_arr:
                    buf[f"{p}_{k}"].append(out[p][k].detach().float().cpu().numpy())
        else:
            mc_probs = {p: [] for p in ("w", "r")}
            mc_means = {p: [] for p in ("w", "r")}
            mc_stds = {p: [] for p in ("w", "r")}
            mc_uamb = {p: [] for p in ("w", "r")}
            mc_uplur = {p: [] for p in ("w", "r")}
            for _ in range(mc_dropout_T):
                with _amp_autocast(enabled=(device.type == "cuda")):
                    out = model(input_ids, attn, struct_feats=st)
                for p in ("w", "r"):
                    mc_probs[p].append(out[p]["probs"].detach().float())
                    mc_means[p].append(out[p]["disc_mean"].detach().float())
                    mc_stds[p].append(out[p]["disc_std"].detach().float())
                    mc_uamb[p].append(out[p]["u_amb"].detach().float())
                    mc_uplur[p].append(out[p]["u_plur"].detach().float())
            for p in ("w", "r"):
                p_stack = torch.stack(mc_probs[p], dim=0)  # (T, B, C, K)
                p_bar = p_stack.mean(dim=0)
                ent_bar = -(torch.clamp(p_bar, 1e-8) * torch.log2(torch.clamp(p_bar, 1e-8))).sum(dim=-1)
                bar_ent = -(torch.clamp(p_stack, 1e-8) * torch.log2(torch.clamp(p_stack, 1e-8))).sum(dim=-1).mean(dim=0)
                mi = torch.clamp(ent_bar - bar_ent, min=0.0)
                buf[f"{p}_probs"].append(p_bar.cpu().numpy())
                buf[f"{p}_disc_mean"].append(torch.stack(mc_means[p], 0).mean(0).cpu().numpy())
                buf[f"{p}_disc_std"].append(torch.stack(mc_stds[p], 0).mean(0).cpu().numpy())
                buf[f"{p}_u_amb"].append(bar_ent.cpu().numpy())
                buf[f"{p}_u_plur"].append(mi.cpu().numpy())
                buf[f"{p}_mc_mi"].append(mi.cpu().numpy())

        for k in ("w_probs", "r_probs", "w_mean", "r_mean", "w_std", "r_std", "w_has_hist", "r_has_hist"):
            buf[f"tgt_{k}" if not k.endswith("hist") else k].append(batch[k].numpy())
        ids.extend(batch["ids"])
        cats.extend(batch["categories"])
        docs.extend(batch["documents"])

    model.eval()
    res: Dict[str, np.ndarray | List[str]] = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
    res["ids"] = ids
    res["categories"] = cats
    res["documents"] = docs
    return res


def score_predictions(pred: Dict[str, np.ndarray | List[str]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(
        evaluate_perspective_predictions(
            tgt_probs=pred["tgt_r_probs"],
            pred_probs=pred["r_probs"],
            tgt_mean=pred["tgt_r_mean"],
            pred_mean=pred["r_disc_mean"],
            tgt_std=pred["tgt_r_std"],
            pred_std=pred["r_disc_std"],
            has_hist=pred["r_has_hist"],
            u_plur=pred["r_u_plur"],
            u_amb=pred["r_u_amb"],
            prefix="r",
        )
    )
    metrics.update(
        evaluate_perspective_predictions(
            tgt_probs=pred["tgt_w_probs"],
            pred_probs=pred["w_probs"],
            tgt_mean=pred["tgt_w_mean"],
            pred_mean=pred["w_disc_mean"],
            tgt_std=pred["tgt_w_std"],
            pred_std=pred["w_disc_std"],
            has_hist=pred["w_has_hist"],
            u_plur=pred["w_u_plur"],
            u_amb=pred["w_u_amb"],
            prefix="w",
        )
    )
    metrics.update(
        evaluate_perspective_shift(
            w_mean_true=pred["tgt_w_mean"],
            r_mean_true=pred["tgt_r_mean"],
            w_mean_pred=pred["w_disc_mean"],
            r_mean_pred=pred["r_disc_mean"],
        )
    )
    return metrics


def selection_score(model_type: str, dev_metrics: Dict[str, float]) -> float:
    """Fair checkpoint selection criteria matched to each model family's objective."""
    if model_type == "scalar_mse":
        return 0.5 * (dev_metrics["r_mean_pearson"] + dev_metrics["w_mean_pearson"])
    if model_type in ("gaussian_nll", "nig_evidential"):
        return 0.5 * (dev_metrics["r_mean_pearson"] + dev_metrics["w_mean_pearson"]) - 0.5 * dev_metrics["r_mean_std_mae"]
    base = (
        0.5 * (dev_metrics["r_mean_pearson"] + dev_metrics["w_mean_pearson"])
        - dev_metrics.get("r_mean_w1", 0.0)
        - 0.25 * dev_metrics.get("r_mean_std_mae", 0.0)
    )
    return float(base)


def evaluate_operational_and_conformal(
    sel_pred: Dict[str, np.ndarray | List[str]],
    cal_pred: Dict[str, np.ndarray | List[str]],
    te_pred: Dict[str, np.ndarray | List[str]],
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Run leakage-free Mondrian credal calibration and Conditional-FNR CRC alerting."""
    def _filter_has_hist(d: Dict[str, np.ndarray | List[str]]) -> Dict[str, np.ndarray | List[str]]:
        if "r_has_hist" not in d:
            return d
        m = np.asarray(d["r_has_hist"], dtype=bool)
        if np.all(m):
            return d
        n = len(m)
        out_d: Dict[str, np.ndarray | List[str]] = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and len(v) == n:
                out_d[k] = v[m]
            elif isinstance(v, list) and len(v) == n:
                out_d[k] = [v[i] for i in range(n) if m[i]]
            else:
                out_d[k] = v
        return out_d

    sel_pred = _filter_has_hist(sel_pred)
    cal_pred = _filter_has_hist(cal_pred)
    te_pred = _filter_has_hist(te_pred)
    out: Dict[str, float] = {}
    # Evaluate across V, A, D jointly or per axis; aggregate clean operational alerts on V (distress) + any-axis (vod)
    def _any_vod_label(pred_d: Dict[str, np.ndarray | List[str]]) -> np.ndarray:
        flags = np.zeros(len(pred_d["tgt_r_mean"]), dtype=bool)
        for j in range(3):
            m = extract_clean_polarization_masks(pred_d["tgt_r_probs"][:, j, :], pred_d["tgt_r_mean"][:, j], pred_d["tgt_r_std"][:, j])
            flags |= m["neut"] & m["valley_or_dip"]
        return flags.astype(int)

    def _v_distress_label(pred_d: Dict[str, np.ndarray | List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        m = extract_clean_polarization_masks(pred_d["tgt_r_probs"][:, 0, :], pred_d["tgt_r_mean"][:, 0], pred_d["tgt_r_std"][:, 0])
        y = (m["distress_neut"]).astype(int)
        p_low = pred_d["tgt_r_probs"][:, 0, :2].sum(-1)
        sev = p_low * m["neut"].astype(float)
        return y, sev

    def _combined_pol_features(pred_d: Dict[str, np.ndarray | List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        feats = [build_risk_features(pred_d["r_probs"][:, j, :], pred_d["r_u_amb"][:, j], pred_d["r_u_plur"][:, j]) for j in range(3)]
        X = np.concatenate(feats, axis=1)
        # Unsupervised single-score ranking index: max across axes of (u_plur or ER_0.8)
        raw_er = np.max(np.stack([esteban_ray_polarization(pred_d["r_probs"][:, j, :]) for j in range(3)], axis=-1), axis=-1)
        raw_uplur = np.max(pred_d["r_u_plur"], axis=-1)
        return X, np.log1p(raw_uplur * 100.0) + raw_er

    sel_X_pol, _ = _combined_pol_features(sel_pred)
    cal_X_pol, cal_sc_unsup = _combined_pol_features(cal_pred)
    te_X_pol, te_sc_unsup = _combined_pol_features(te_pred)

    sel_y_pol = _any_vod_label(sel_pred)
    cal_y_pol = _any_vod_label(cal_pred)
    te_y_pol = _any_vod_label(te_pred)

    # 1) Supervised combiner CRC (strictly fitted on dev_select, calibrated on dev_cal)
    crc_pol_sup = fit_and_calibrate_alert_crc(sel_X_pol, sel_y_pol, cal_X_pol, cal_y_pol, te_X_pol, te_y_pol, alpha=alpha, use_supervised_combiner=True)
    for k, v in crc_pol_sup.items():
        out[f"pol_crc_sup_{k}"] = v

    # 2) Pure unsupervised CRC directly on raw u_plur (isolates zero-supervision alerting power!)
    crc_pol_uplur = fit_and_calibrate_alert_crc(
        np.max(sel_pred["r_u_plur"], axis=-1), sel_y_pol,
        np.max(cal_pred["r_u_plur"], axis=-1), cal_y_pol,
        np.max(te_pred["r_u_plur"], axis=-1), te_y_pol,
        alpha=alpha, use_supervised_combiner=False,
    )
    for k, v in crc_pol_uplur.items():
        out[f"pol_crc_uplur_{k}"] = v

    # Operational budget metrics on raw u_plur and unsupervised ER+u_plur
    for k, v in evaluate_binary_alert_metrics(te_y_pol, np.max(te_pred["r_u_plur"], axis=-1)).items():
        out[f"op_pol_uplur_{k}"] = v
    for k, v in evaluate_binary_alert_metrics(te_y_pol, te_sc_unsup).items():
        out[f"op_pol_unsup_{k}"] = v

    # Distress neutral alert on Valence
    sel_y_d, _ = _v_distress_label(sel_pred)
    cal_y_d, cal_sev_d = _v_distress_label(cal_pred)
    te_y_d, te_sev_d = _v_distress_label(te_pred)
    sel_X_d = build_risk_features(sel_pred["r_probs"][:, 0, :], sel_pred["r_u_amb"][:, 0], sel_pred["r_u_plur"][:, 0])
    cal_X_d = build_risk_features(cal_pred["r_probs"][:, 0, :], cal_pred["r_u_amb"][:, 0], cal_pred["r_u_plur"][:, 0])
    te_X_d = build_risk_features(te_pred["r_probs"][:, 0, :], te_pred["r_u_amb"][:, 0], te_pred["r_u_plur"][:, 0])
    crc_d = fit_and_calibrate_alert_crc(
        sel_X_d, sel_y_d, cal_X_d, cal_y_d, te_X_d, te_y_d,
        alpha=alpha, cal_severity=cal_sev_d, test_severity=te_sev_d, use_supervised_combiner=True,
    )
    for k, v in crc_d.items():
        out[f"distress_crc_{k}"] = v
    for k, v in evaluate_binary_alert_metrics(te_y_d, te_pred["r_probs"][:, 0, :2].sum(-1)).items():
        out[f"op_distress_plow_{k}"] = v

    # Mondrian Conformal Credal Sets on Reader Valence (j=0) and Mean across V/A/D
    for j, dname in enumerate(("V", "A", "D")):
        mc = mondrian_conformal_credal_sets(
            sel_pred["r_probs"][:, j, :], sel_pred["tgt_r_probs"][:, j, :], sel_pred["r_u_amb"][:, j], sel_pred["r_u_plur"][:, j],
            cal_pred["r_probs"][:, j, :], cal_pred["tgt_r_probs"][:, j, :], cal_pred["r_u_amb"][:, j], cal_pred["r_u_plur"][:, j],
            te_pred["r_probs"][:, j, :], te_pred["tgt_r_probs"][:, j, :], te_pred["r_u_amb"][:, j], te_pred["r_u_plur"][:, j],
            test_categories=te_pred["categories"], cal_categories=cal_pred["categories"], alpha=alpha,
        )
        for k, v in mc.items():
            out[f"conf_{dname}_{k}"] = v
    return out


def run_single(args: argparse.Namespace) -> Dict[str, object]:
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_emobank_dataframe()
    df_tr, df_sel, df_cal, df_te = get_emobank_4way_splits(df, protocol=args.protocol, seed=args.seed)
    prev = compute_train_prevalence(df_tr)
    st_tr, st_sel, st_cal, st_te = fit_struct_scaler_for_splits(df_tr, df_sel, df_cal, df_te)

    tok_path = resolve_backbone_path(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    special_ids = tuple(x for x in (tokenizer.cls_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id) if x is not None)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)

    tr_loader = DataLoader(EmoBankDistributionDataset(df_tr, st_tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    sel_loader = DataLoader(EmoBankDistributionDataset(df_sel, st_sel), batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    cal_loader = DataLoader(EmoBankDistributionDataset(df_cal, st_cal), batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    te_loader = DataLoader(EmoBankDistributionDataset(df_te, st_te), batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    model = DualPerspectiveAffectModel(
        backbone_name=args.backbone,
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        ablation_mode=args.ablation_mode,
        special_token_ids=special_ids,
    ).to(device)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    grad_accum = max(1, int(args.grad_accum))
    steps_per_epoch = max(1, (len(tr_loader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = _make_grad_scaler(enabled=(device.type == "cuda"))

    weights = {
        "kl": args.w_kl,
        "w1": args.w_w1,
        "mom": args.w_mom,
        "shift_mean": args.w_shift_mean,
        "shift_cdf": args.w_shift_cdf,
        "trans_reg": args.w_trans_reg,
        "pol_shape": args.w_pol_shape,
        "pol_focus": args.w_pol_focus,
        "pars_p": args.w_pars_p,
        "pars_a": args.w_pars_a,
        "pars_rep": args.w_pars_rep,
        "pars_bal": args.w_pars_bal,
        "delta_min": args.delta_min,
        **prev,
    }

    best_score = -1e9
    best_state = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(tr_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            st = batch["struct_feats"].to(device, non_blocking=True)
            tgt_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with _amp_autocast(enabled=(device.type == "cuda")):
                out = model(input_ids, attn, struct_feats=st)
                loss, _ = compute_model_loss(args.model_type, out, tgt_batch, weights)
                loss_scaled = loss / float(grad_accum)
            scaler.scale(loss_scaled).backward()
            running += float(loss.detach().item())
            if step_idx % grad_accum == 0 or step_idx == len(tr_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

        sel_pred = predict_loader(model, sel_loader, device)
        sel_metrics = score_predictions(sel_pred)
        sc = selection_score(args.model_type, sel_metrics)
        if sc > best_score:
            best_score = sc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[{args.model_type} | {args.ablation_mode} | {args.protocol} s={args.seed} ep {epoch}/{args.epochs}] "
            f"tr={running/max(len(tr_loader),1):.4f} "
            f"sel_w1={sel_metrics.get('r_mean_w1', float('nan')):.4f} "
            f"sel_pear={sel_metrics['r_mean_pearson']:.4f} "
            f"sel_hs_uplur={sel_metrics.get('r_hard_shape_auroc_by_uplur', float('nan')):.4f} "
            f"sel_shift_r={sel_metrics.get('shift_mean_pearson', float('nan')):.4f}"
        )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    mc_T = int(args.mc_dropout_T)
    sel_pred = predict_loader(model, sel_loader, device, mc_dropout_T=mc_T)
    cal_pred = predict_loader(model, cal_loader, device, mc_dropout_T=mc_T)
    te_pred = predict_loader(model, te_loader, device, mc_dropout_T=mc_T)

    sel_metrics = score_predictions(sel_pred)
    cal_metrics = score_predictions(cal_pred)
    te_metrics = score_predictions(te_pred)
    conformal_metrics = evaluate_operational_and_conformal(sel_pred, cal_pred, te_pred, alpha=0.10)

    tag = f"{args.protocol}_{args.backbone.replace('/', '_')}_{args.model_type}_{args.ablation_mode}_seed{args.seed}"
    if mc_T > 1:
        tag += f"_mcT{mc_T}"
    run_dir = Path(args.out_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_type": args.model_type,
        "ablation_mode": args.ablation_mode,
        "backbone": args.backbone,
        "protocol": args.protocol,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "elapsed_sec": round(time.time() - t0, 1),
        "weights": weights,
        "select_metrics": sel_metrics,
        "cal_metrics": cal_metrics,
        "test_metrics": te_metrics,
        "conformal_and_alert_metrics": conformal_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for sp_name, p_dict in (("cal_predictions.npz", cal_pred), ("test_predictions.npz", te_pred)):
        np.savez_compressed(
            run_dir / sp_name,
            **{k: v for k, v in p_dict.items() if isinstance(v, np.ndarray)},
            ids=np.array(p_dict["ids"]),
            categories=np.array(p_dict["categories"]),
            documents=np.array(p_dict["documents"]),
        )

    print(
        f"==> TEST [{tag}] r_w1={te_metrics.get('r_mean_w1', float('nan')):.4f} "
        f"r_pear={te_metrics['r_mean_pearson']:.4f} "
        f"hs_uplur={te_metrics.get('r_hard_shape_auroc_by_uplur', float('nan')):.4f} "
        f"hs_uamb={te_metrics.get('r_hard_shape_auroc_by_uamb', float('nan')):.4f} "
        f"shift_mae={te_metrics['shift_mean_mae']:.4f} shift_r={te_metrics['shift_mean_pearson']:.4f} "
        f"pol_crc_fnr={conformal_metrics.get('pol_crc_sup_test_fnr', float('nan')):.4f}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--model-type", default="patnet")
    p.add_argument("--ablation-mode", default="M4_disentangle")
    p.add_argument("--protocol", default="cv_fold_0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mc-dropout-T", type=int, default=0)
    p.add_argument("--w-kl", type=float, default=1.0)
    p.add_argument("--w-w1", type=float, default=1.0)
    p.add_argument("--w-mom", type=float, default=0.6)
    p.add_argument("--w-shift-mean", type=float, default=1.5)
    p.add_argument("--w-shift-cdf", type=float, default=0.8)
    p.add_argument("--w-trans-reg", type=float, default=0.002)
    p.add_argument("--w-pol-shape", type=float, default=0.8)
    p.add_argument("--w-pol-focus", type=float, default=1.0)
    p.add_argument("--w-pars-p", type=float, default=0.025)
    p.add_argument("--w-pars-a", type=float, default=0.45)
    p.add_argument("--w-pars-rep", type=float, default=0.60)
    p.add_argument("--w-pars-bal", type=float, default=0.05)
    p.add_argument("--delta-min", type=float, default=1.6)
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    p.add_argument("--out-dir", default=str(OUT_DIR / "emobank"))
    return p


if __name__ == "__main__":
    run_single(build_parser().parse_args())
