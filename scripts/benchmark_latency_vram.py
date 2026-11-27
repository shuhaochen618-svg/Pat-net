"""Exact GPU inference latency (p50/p95/p99), batch QPS, parameter count, and peak VRAM benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import DualPerspectiveAffectModel
from src.paths import OUT_DIR, resolve_backbone_path
from src.struct_features import extract_struct_matrix

SAMPLE_SENTENCES = [
    "This policy completely failed some residents while helping others tremendously, leaving everyone divided.",
    "The quarterly report arrived right on schedule at nine o'clock this morning.",
    "Wait — how could you possibly call that fair when half the neighborhood loses power?!",
    "I'm honestly terrified and exhausted, though maybe everything will work out fine.",
]


@torch.no_grad()
def benchmark_model_config(
    backbone_name: str,
    model_type: str,
    ablation_mode: str = "M4_disentangle",
    mc_T: int = 1,
    n_warmup: int = 15,
    n_iters: int = 80,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    resolved = resolve_backbone_path(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved)

    model = DualPerspectiveAffectModel(
        backbone_name=backbone_name,
        model_type=model_type,
        ablation_mode=ablation_mode,
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_head_params = int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.")))

    if mc_T > 1:
        model.train()
    else:
        model.eval()

    def _run_bs(bs: int) -> List[float]:
        texts = (SAMPLE_SENTENCES * ((bs // len(SAMPLE_SENTENCES)) + 1))[:bs]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        sf = torch.from_numpy(extract_struct_matrix(texts)).to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        for _ in range(n_warmup):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                for _ in range(mc_T):
                    _ = model(ids, mask, struct_feats=sf)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        lat_ms: List[float] = []
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                for _ in range(mc_T):
                    _ = model(ids, mask, struct_feats=sf)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
        return lat_ms

    lat_bs1 = _run_bs(1)
    peak_bs1 = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0

    lat_bs32 = _run_bs(32)
    peak_bs32 = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0
    qps_bs32 = 32.0 / (float(np.mean(lat_bs32)) / 1000.0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "backbone": backbone_name,
        "model_type": model_type,
        "ablation_mode": ablation_mode,
        "mc_dropout_T": mc_T,
        "total_params_M": round(n_params / 1e6, 3),
        "head_reader_params_M": round(n_head_params / 1e6, 3),
        "bs1_p50_ms": round(float(np.percentile(lat_bs1, 50)), 2),
        "bs1_p95_ms": round(float(np.percentile(lat_bs1, 95)), 2),
        "bs1_p99_ms": round(float(np.percentile(lat_bs1, 99)), 2),
        "bs32_mean_ms": round(float(np.mean(lat_bs32)), 2),
        "bs32_qps": round(qps_bs32, 1),
        "peak_vram_bs1_mb": round(peak_bs1, 1),
        "peak_vram_bs32_mb": round(peak_bs32, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=["roberta-base"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in args.backbones:
        configs = [
            ("scalar_mse", "M0_baseline", 1),
            ("dc_ldl", "M0_baseline", 1),
            ("patnet", "M0_baseline", 1),
            ("patnet", "M4_disentangle", 1),
            ("patnet", "M5_full_struct", 1),
            ("gaussian_nll", "M0_baseline", 10),
        ]
        for mt, ab, mc_t in configs:
            try:
                res = benchmark_model_config(bb, mt, ab, mc_T=mc_t)
                rows.append(res)
                print(json.dumps(res))
            except Exception as exc:
                print(f"[WARN] Skipping {bb}/{mt}/{ab}: {exc}")

    # Derived 5-model sequential deep ensemble reference: latency/params scale 5x, QPS scales 1/5x, peak VRAM reflects 1x sequential load
    for r in list(rows):
        if r["model_type"] == "dc_ldl" and r["mc_dropout_T"] == 1:
            rows.append(
                {
                    **r,
                    "model_type": "deep_ensemble_E5(dc_ldl)",
                    "total_params_M": round(r["total_params_M"] * 5, 3),
                    "head_reader_params_M": round(r["head_reader_params_M"] * 5, 3),
                    "bs1_p50_ms": round(r["bs1_p50_ms"] * 5, 2),
                    "bs1_p95_ms": round(r["bs1_p95_ms"] * 5, 2),
                    "bs1_p99_ms": round(r["bs1_p99_ms"] * 5, 2),
                    "bs32_mean_ms": round(r["bs32_mean_ms"] * 5, 2),
                    "bs32_qps": round(r["bs32_qps"] / 5.0, 1),
                    "vram_note": "sequential_1x_peak_vram (5x_resident_weights)",
                }
            )

    (OUT_DIR / "hardware_latency_vram.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
