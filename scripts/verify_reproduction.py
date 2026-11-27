"""Verify dataset integrity, split isolation, configuration, visual assets, and repository hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PROC = ROOT / "data" / "processed"
OUT_ROOT = ROOT / "outputs"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {msg}")


def approx(a: float, b: float, tol: float, label: str) -> None:
    check(abs(a - b) <= tol, f"{label}: got {a}, expected {b} +/- {tol}")


def verify_datasets() -> None:
    eb = pd.read_parquet(DATA_PROC / "emobank_full.parquet")
    ep = pd.read_parquet(DATA_PROC / "epic_agg.parquet")
    ge = pd.read_parquet(DATA_PROC / "goemotions_raw_agg.parquet")
    check(len(eb) == 10062, f"EmoBank rows={len(eb)}")
    check(len(ep) == 3000, f"EPIC rows={len(ep)}")
    check(len(ge) == 58011, f"GoEmotions rows={len(ge)}")

    # Verify strict document isolation across all 5 CV folds of EmoBank
    from src.datasets import get_emobank_4way_splits

    for fold in range(5):
        tr, sel, cal, te = get_emobank_4way_splits(eb, protocol=f"cv_fold_{fold}")
        d_tr = set(tr["document"])
        d_sel = set(sel["document"])
        d_cal = set(cal["document"])
        d_te = set(te["document"])
        check(
            len(d_tr & d_sel) == 0
            and len(d_tr & d_cal) == 0
            and len(d_tr & d_te) == 0
            and len(d_sel & d_cal) == 0
            and len(d_sel & d_te) == 0
            and len(d_cal & d_te) == 0,
            f"Document overlap detected in EmoBank cv_fold_{fold}",
        )

    # Verify strict group_key isolation across EPIC folds
    from src.epic_exp import get_epic_4way_splits

    for fold in range(5):
        tr, sel, cal, te = get_epic_4way_splits(ep, cv_fold=fold)
        g_sets = [set(x["group_key"]) for x in (tr, sel, cal, te)]
        for i in range(4):
            for j in range(i + 1, 4):
                check(len(g_sets[i] & g_sets[j]) == 0, f"Group overlap in EPIC cv{fold}")

    print("[OK] Datasets verified: EmoBank (10,062), EPIC (3,000), GoEmotions (58,011), 0 CV leakage.")


def verify_configs_and_assets() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    check("emobank" in cfg and cfg["emobank"]["epochs"] == 6, "Unexpected EmoBank settings in default.yaml")
    check(cfg["emobank"]["ensemble_fold0_seeds"] == [42, 43, 44, 45, 46], "Unexpected ensemble seeds")

    expected_figs = [
        "fig1_framework.png",
        "fig2_empirical_phenomena_and_audit.png",
        "fig3_emobank_pareto_and_efficiency.png",
        "fig4_disentanglement_and_ablation.png",
        "fig5_perspective_transport_and_logo.png",
        "fig6_cross_dataset_epic_brighter.png",
        "fig7_bootstrap_forest_and_conformal.png",
    ]
    for fn in expected_figs:
        fp = ROOT / "assets" / fn
        check(fp.exists() and fp.stat().st_size > 10000, f"Missing visual asset: {fp}")

    gates_ref = ROOT / "assets" / "results" / "all_gates_report.json"
    check(gates_ref.exists(), f"Missing curated results JSON: {gates_ref}")
    print("[OK] Configs, curated results, and figure assets verified.")


def verify_gate_report(report_path: Path) -> None:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    gates = rep["gates"]
    check(gates["gate1_distributional_win_no_scalar_drop"]["passed"] is True, "Gate 1 failed")
    check(gates["gate2_non_circular_shape_disentanglement"]["passed"] is True, "Gate 2 failed")
    check(gates["gate3_causal_perspective_transport"]["passed"] is True, "Gate 3 failed")
    check(gates["gate4_epic_5fold_and_brighter"]["epic_passed"] is True, "Gate 4 EPIC failed")
    check(len(gates["gate4_epic_5fold_and_brighter"]["brighter"]) == 35, "Expected 35 BRIGHTER runs")

    g1 = gates["gate1_distributional_win_no_scalar_drop"]
    approx(g1["primary_r_w1"], 0.2612, 1e-4, "Gate 1 PAT-Net M4 r_w1")
    g3 = gates["gate3_causal_perspective_transport"]
    approx(g3["primary_shift_mae"], 0.2606, 1e-4, "Gate 3 PAT-Net M4 shift_mae")
    print(f"[OK] Gate report verified cleanly: {report_path}")


def verify_hygiene() -> None:
    pat = re.compile(r"(/root/|/home/|(?<![A-Za-z])[CD]:[/\\]|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.|seetacloud)", re.I)
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in {".git", "__pycache__", "outputs", ".venv", "pretrained", "logs"}]
        for f in fns:
            if not f.endswith((".py", ".yaml", ".yml", ".json", ".md", ".txt", ".csv", ".toml")):
                continue
            if f == "verify_reproduction.py":
                continue
            fp = Path(dp) / f
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            check(pat.search(txt) is None, f"Private path/IP pattern found in {fp.relative_to(ROOT)}")

    expected_md5 = {
        "src/models.py": "ea460c00e8d5fbdf9364a40f53820e5e",
        "src/losses.py": "ab69e62e8c794a454e4365e1cb84be4b",
        "scripts/aggregate_eval_gates.py": "f5df4af7cbf80a12db1ae5bbd12c2ed6",
    }
    for rel, want in expected_md5.items():
        got = hashlib.md5((ROOT / rel).read_bytes()).hexdigest()
        check(got == want, f"Checksum mismatch on {rel}: {got} != {want}")
    print("[OK] Repository hygiene & core checksums verified (0 private paths/IPs).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PAT-Net dataset splits, hygiene, configs, and gate reports.")
    parser.add_argument("--with-outputs", action="store_true", help="Also verify runtime outputs/all_gates_report.json.")
    args = parser.parse_args()

    verify_datasets()
    verify_configs_and_assets()
    verify_hygiene()
    verify_gate_report(ROOT / "assets" / "results" / "all_gates_report.json")

    runtime_report = OUT_ROOT / "all_gates_report.json"
    if args.with_outputs:
        check(runtime_report.exists(), f"Missing runtime output: {runtime_report}")
        verify_gate_report(runtime_report)
    elif runtime_report.exists():
        verify_gate_report(runtime_report)

    print("\nALL PAT-NET REPOSITORY CHECKS PASSED CLEANLY.")


if __name__ == "__main__":
    main()
