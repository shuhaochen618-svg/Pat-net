"""Progressive, resumable RTX 4090 server experiment pipeline runner with gate verification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

PYTHON = sys.executable


def _run_cmd(cmd: List[str], tag: str, skip_if_exists: Path | None = None, overwrite: bool = False) -> None:
    if (skip_if_exists is not None) and skip_if_exists.exists() and not overwrite:
        print(f"[SKIP] {tag} -> {skip_if_exists}")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    t0 = time.time()
    print(f"\n[RUN] {tag}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print(f"[DONE] {tag} in {time.time() - t0:.1f}s", flush=True)


def run_smoke(overwrite: bool = False) -> None:
    """Smoke sanity gate (~5-7 min): classical fold0 + roberta-base fold0 (patnet, dc_ldl, no_transport) + EPIC fold0."""
    for cm in ("C1_SparseRidge", "C2_HybridGBDT"):
        _run_cmd(
            [PYTHON, "-m", "src.classical_baselines", "--task", "emobank", "--model-name", cm, "--protocol", "cv_fold_0", "--seed", "42"],
            f"classical_{cm}_fold0",
            skip_if_exists=OUT_DIR / "emobank" / f"cv_fold_0_{cm}_seed42" / "summary.json",
            overwrite=overwrite,
        )

    for mt, ab in (
        ("patnet", "M4_disentangle"),
        ("dc_ldl", "M0_baseline"),
        ("patnet_no_transport", "M4_disentangle"),
    ):
        tag = f"cv_fold_0_roberta-base_{mt}_{ab}_seed42"
        _run_cmd(
            [
                PYTHON, "-m", "src.train_emobank",
                "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                "--protocol", "cv_fold_0", "--seed", "42", "--epochs", "6", "--batch-size", "32", "--num-workers", "2",
            ],
            tag,
            skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
            overwrite=overwrite,
        )

    for mt, ab in (("patnet", "M4_disentangle"), ("patnet_no_transport", "M4_disentangle"), ("beta_binom", "M0_baseline")):
        tag = f"cv0_roberta-base_{mt}_{ab}_seed42"
        _run_cmd(
            [
                PYTHON, "-m", "src.epic_exp",
                "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                "--cv-fold", "0", "--seed", "42", "--epochs", "5", "--batch-size", "16", "--num-workers", "2",
            ],
            f"epic_{tag}",
            skip_if_exists=OUT_DIR / "epic" / tag / "summary.json",
            overwrite=overwrite,
        )


def run_stage1_emobank(overwrite: bool = False) -> None:
    """Full 5-fold StratifiedGroupKFold CV across all baselines, PAT-Net M4/M5, MC-Dropout T=10, and multi-seed ensemble."""
    for f in range(5):
        proto = f"cv_fold_{f}"
        for cm in ("C1_SparseRidge", "C2_HybridGBDT"):
            _run_cmd(
                [PYTHON, "-m", "src.classical_baselines", "--task", "emobank", "--model-name", cm, "--protocol", proto, "--seed", "42"],
                f"classical_{cm}_{proto}",
                skip_if_exists=OUT_DIR / "emobank" / f"{proto}_{cm}_seed42" / "summary.json",
                overwrite=overwrite,
            )

    specs = [
        ("scalar_mse", "M0_baseline", 0),
        ("gaussian_nll", "M0_baseline", 0),
        ("gaussian_nll", "M0_baseline", 10),
        ("nig_evidential", "M0_baseline", 0),
        ("sord", "M0_baseline", 0),
        ("dldl_v2", "M0_baseline", 0),
        ("dc_ldl", "M0_baseline", 0),
        ("dc_ldl", "M0_baseline", 10),
        ("uw_ldl", "M0_baseline", 0),
        ("beta_nll", "M0_baseline", 0),
        ("patnet_unimodal", "M4_disentangle", 0),
        ("patnet_no_transport", "M4_disentangle", 0),
        ("patnet", "M4_disentangle", 0),
        ("patnet", "M5_full_struct", 0),
    ]
    for mt, ab, mc_t in specs:
        for f in range(5):
            proto = f"cv_fold_{f}"
            tag = f"{proto}_roberta-base_{mt}_{ab}_seed42" + (f"_mcT{mc_t}" if mc_t > 1 else "")
            _run_cmd(
                [
                    PYTHON, "-m", "src.train_emobank",
                    "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                    "--protocol", proto, "--seed", "42", "--epochs", "6", "--batch-size", "32",
                    "--mc-dropout-T", str(mc_t), "--num-workers", "2",
                ],
                tag,
                skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
                overwrite=overwrite,
            )

    # Multi-seed Deep Ensemble E=5 reference on cv_fold_0 (for patnet + dc_ldl)
    for mt, ab in (("dc_ldl", "M0_baseline"), ("patnet", "M4_disentangle")):
        for s in (43, 44, 45, 46):
            tag = f"cv_fold_0_roberta-base_{mt}_{ab}_seed{s}"
            _run_cmd(
                [
                    PYTHON, "-m", "src.train_emobank",
                    "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                    "--protocol", "cv_fold_0", "--seed", str(s), "--epochs", "6", "--batch-size", "32", "--num-workers", "2",
                ],
                tag,
                skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
                overwrite=overwrite,
            )

    _run_cmd([PYTHON, "scripts/aggregate_eval_gates.py"], "aggregate_gates")


def run_stage2_ablations(overwrite: bool = False) -> None:
    """M0..M3 modular reader ablations, DeBERTa-v3 backbone replication, Leave-One-Genre-Out, and hardware latency/VRAM."""
    for ab in ("M0_baseline", "M1_multilayer_conv", "M2_bipolar_pool", "M3_tension_route"):
        for f in range(5):
            proto = f"cv_fold_{f}"
            tag = f"{proto}_roberta-base_patnet_{ab}_seed42"
            _run_cmd(
                [
                    PYTHON, "-m", "src.train_emobank",
                    "--backbone", "roberta-base", "--model-type", "patnet", "--ablation-mode", ab,
                    "--protocol", proto, "--seed", "42", "--epochs", "6", "--batch-size", "32", "--num-workers", "2",
                ],
                tag,
                skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
                overwrite=overwrite,
            )

    # DeBERTa-v3-base across cv_fold_0..4
    for mt, ab in (
        ("scalar_mse", "M0_baseline"),
        ("sord", "M0_baseline"),
        ("dc_ldl", "M0_baseline"),
        ("beta_nll", "M0_baseline"),
        ("patnet_no_transport", "M4_disentangle"),
        ("patnet", "M4_disentangle"),
    ):
        for f in range(5):
            proto = f"cv_fold_{f}"
            tag = f"{proto}_microsoft_deberta-v3-base_{mt}_{ab}_seed42"
            _run_cmd(
                [
                    PYTHON, "-m", "src.train_emobank",
                    "--backbone", "microsoft/deberta-v3-base", "--model-type", mt, "--ablation-mode", ab,
                    "--protocol", proto, "--seed", "42", "--epochs", "6", "--batch-size", "16", "--grad-accum", "2",
                    "--encoder-lr", "1.5e-5", "--num-workers", "2",
                ],
                tag,
                skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
                overwrite=overwrite,
            )

    # Leave-One-Genre-Out across all 7 EmoBank genres
    for genre in ("fiction", "letters", "blog", "newspaper", "SemEval", "essays", "travel-guides"):
        proto = f"logo_{genre}"
        for mt, ab in (("dc_ldl", "M0_baseline"), ("patnet_no_transport", "M4_disentangle"), ("patnet", "M4_disentangle")):
            tag = f"{proto}_roberta-base_{mt}_{ab}_seed42"
            _run_cmd(
                [
                    PYTHON, "-m", "src.train_emobank",
                    "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                    "--protocol", proto, "--seed", "42", "--epochs", "6", "--batch-size", "32", "--num-workers", "2",
                ],
                tag,
                skip_if_exists=OUT_DIR / "emobank" / tag / "summary.json",
                overwrite=overwrite,
            )

    _run_cmd(
        [PYTHON, "scripts/benchmark_latency_vram.py", "--backbones", "roberta-base", "microsoft/deberta-v3-base"],
        "hardware_benchmark",
        skip_if_exists=OUT_DIR / "hardware_latency_vram.json",
        overwrite=overwrite,
    )


def run_stage3_aux(overwrite: bool = False) -> None:
    """EPIC 5-fold Beta-Binomial transport + BRIGHTER EN/CHN/DEU/ESP cross-lingual ordinal calibration."""
    for mt, ab in (
        ("scalar_bce", "M0_baseline"),
        ("beta_binom", "M0_baseline"),
        ("patnet_no_transport", "M4_disentangle"),
        ("patnet", "M4_disentangle"),
    ):
        for f in range(5):
            tag = f"cv{f}_roberta-base_{mt}_{ab}_seed42"
            _run_cmd(
                [
                    PYTHON, "-m", "src.epic_exp",
                    "--backbone", "roberta-base", "--model-type", mt, "--ablation-mode", ab,
                    "--cv-fold", str(f), "--seed", "42", "--epochs", "5", "--batch-size", "16", "--num-workers", "2",
                ],
                f"epic_{tag}",
                skip_if_exists=OUT_DIR / "epic" / tag / "summary.json",
                overwrite=overwrite,
            )

    # EPIC DeBERTa-v3 comparison (folds 0..2) + classical baselines
    for f in range(3):
        for mt, ab in (("patnet_no_transport", "M4_disentangle"), ("patnet", "M4_disentangle")):
            tag = f"cv{f}_microsoft__deberta-v3-base_{mt}_{ab}_seed42"
            _run_cmd(
                [
                    PYTHON, "-m", "src.epic_exp",
                    "--backbone", "microsoft/deberta-v3-base", "--model-type", mt, "--ablation-mode", ab,
                    "--cv-fold", str(f), "--seed", "42", "--epochs", "5", "--batch-size", "16",
                    "--grad-accum", "2", "--encoder-lr", "1.5e-5", "--num-workers", "2",
                ],
                f"epic_{tag}",
                skip_if_exists=OUT_DIR / "epic" / tag / "summary.json",
                overwrite=overwrite,
            )
    for f in range(5):
        for cm in ("C1_SparseRidge", "C2_HybridGBDT"):
            tag = f"cv{f}_classical_{cm}_seed42"
            _run_cmd(
                [PYTHON, "-m", "src.classical_baselines", "--task", "epic", "--model-name", cm, "--cv-fold", str(f), "--seed", "42"],
                f"epic_{tag}",
                skip_if_exists=OUT_DIR / "epic" / tag / "summary.json",
                overwrite=overwrite,
            )

    # BRIGHTER Chinese track (shared cache backbones + BiGRU + classical)
    chn_specs = [
        ("hfl/chinese-roberta-wwm-ext", "scalar_mse", "M0_baseline"),
        ("hfl/chinese-roberta-wwm-ext", "softmax_ce", "M0_baseline"),
        ("hfl/chinese-roberta-wwm-ext", "sord", "M0_baseline"),
        ("hfl/chinese-roberta-wwm-ext", "uw_ldl", "M0_baseline"),
        ("hfl/chinese-roberta-wwm-ext", "dml_ordinal", "M0_baseline"),
        ("hfl/chinese-roberta-wwm-ext", "dml_ordinal", "M4_disentangle"),
        ("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese", "sord", "M0_baseline"),
        ("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese", "dml_ordinal", "M0_baseline"),
        ("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese", "dml_ordinal", "M4_disentangle"),
        ("bigru_jieba", "bigru_bipolar", "M4_disentangle"),
    ]
    for bb, mt, ab in chn_specs:
        ab_suf = f"_{ab}" if ((mt in ("dml_ordinal", "bigru_bipolar") and ab != "M4_disentangle") or (mt not in ("dml_ordinal", "bigru_bipolar") and ab != "M0_baseline")) else ""
        tag = f"chn_official_{bb.replace('/', '__')}_{mt}{ab_suf}_seed42"
        _run_cmd(
            [
                PYTHON, "-m", "src.brighter_exp",
                "--lang", "chn", "--backbone", bb, "--model-type", mt, "--ablation-mode", ab,
                "--cv-fold", "-1", "--seed", "42", "--epochs", "12" if bb == "bigru_jieba" else "5", "--batch-size", "32", "--num-workers", "2",
            ],
            f"brighter_{tag}",
            skip_if_exists=OUT_DIR / "brighter" / tag / "summary.json",
            overwrite=overwrite,
        )

    # BRIGHTER English + Multilingual (deu, esp) + Classical across all 4 languages
    for lang in ("chn", "eng", "deu", "esp"):
        for cm in ("C1_SparseRidge", "C2_HybridGBDT"):
            tag = f"{lang}_official_classical_{cm}_seed42"
            _run_cmd(
                [PYTHON, "-m", "src.classical_baselines", "--task", "brighter", "--lang", lang, "--model-name", cm, "--cv-fold", "-1", "--seed", "42"],
                f"brighter_{tag}",
                skip_if_exists=OUT_DIR / "brighter" / tag / "summary.json",
                overwrite=overwrite,
            )

    for bb in ("roberta-base", "microsoft/deberta-v3-base"):
        eng_models = [("scalar_mse", "M0_baseline"), ("sord", "M0_baseline"), ("uw_ldl", "M0_baseline"), ("dml_ordinal", "M4_disentangle")]
        if bb == "roberta-base":
            eng_models.append(("dml_ordinal", "M0_baseline"))
        for mt, ab in eng_models:
            ab_suf = f"_{ab}" if (mt == "dml_ordinal" and ab != "M4_disentangle") else ""
            tag = f"eng_official_{bb.replace('/', '__')}_{mt}{ab_suf}_seed42"
            cmd = [
                PYTHON, "-m", "src.brighter_exp",
                "--lang", "eng", "--backbone", bb, "--model-type", mt, "--ablation-mode", ab,
                "--cv-fold", "-1", "--seed", "42", "--epochs", "5",
                "--batch-size", "16" if "deberta" in bb else "32", "--num-workers", "2",
            ]
            if "deberta" in bb:
                cmd.extend(["--grad-accum", "2", "--encoder-lr", "1.5e-5"])
            _run_cmd(
                cmd,
                f"brighter_{tag}",
                skip_if_exists=OUT_DIR / "brighter" / tag / "summary.json",
                overwrite=overwrite,
            )

    for lang in ("deu", "esp"):
        # Enforce a >=500-step floor across all models equally on smaller-N Spanish split (train=1,696 -> 53 steps/ep * 10 = 530 steps w/ bs=32 eff)
        # Use grad_accum=2 & encoder_lr=1.5e-5 identically on esp to prevent XLM-R optimization instability (matching DeBERTa stabilization)
        n_ep = "10" if lang == "esp" else "5"
        bs = "16" if lang == "esp" else "24"
        for mt, ab in (("scalar_mse", "M0_baseline"), ("sord", "M0_baseline"), ("uw_ldl", "M0_baseline"), ("dml_ordinal", "M4_disentangle")):
            tag = f"{lang}_official_xlm-roberta-base_{mt}_seed42"
            cmd = [
                PYTHON, "-m", "src.brighter_exp",
                "--lang", lang, "--backbone", "xlm-roberta-base", "--model-type", mt, "--ablation-mode", ab,
                "--cv-fold", "-1", "--seed", "42", "--epochs", n_ep, "--batch-size", bs, "--num-workers", "2",
            ]
            if lang == "esp":
                cmd.extend(["--grad-accum", "2", "--encoder-lr", "1.5e-5"])
            _run_cmd(
                cmd,
                f"brighter_{tag}",
                skip_if_exists=OUT_DIR / "brighter" / tag / "summary.json",
                overwrite=overwrite,
            )

    _run_cmd([PYTHON, "scripts/aggregate_eval_gates.py"], "aggregate_gates_final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["smoke", "stage1", "stage2", "stage3"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stage == "smoke":
        run_smoke(overwrite=args.overwrite)
    elif args.stage == "stage1":
        run_stage1_emobank(overwrite=args.overwrite)
    elif args.stage == "stage2":
        run_stage2_ablations(overwrite=args.overwrite)
    elif args.stage == "stage3":
        run_stage3_aux(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
