"""End-to-end synthetic forward/backward/AMP/classical checks across EmoBank, EPIC, BRIGHTER, and M0-M5."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.brighter_exp import ALL_EMOS, BrighterOrdinalModel, compute_brighter_loss, make_brighter_collate_fn, BrighterOrdinalDataset
from src.classical_baselines import _build_sparse_design, _extract_oof_and_gbdt_features_emobank
from src.datasets import fit_struct_scaler_for_splits
from src.epic_exp import EpicDataset, EpicPerspectivistModel, compute_epic_loss, make_epic_collate_fn
from src.losses import compute_model_loss
from src.models import DualPerspectiveAffectModel
from src.paths import OUT_DIR
from src.struct_features import extract_struct_matrix


def build_tiny_roberta_dir(out_dir: Path) -> Path:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_rb = ROOT / "pretrained" / "roberta-base"
    if local_rb.exists() and (local_rb / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(str(local_rb))
        tok.save_pretrained(out_dir)
        vocab_sz = len(tok)
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "but": 4, "good": 5, "bad": 6, "not": 7, "policy": 8, "happy": 9, "sad": 10, "result": 11, "day": 12}
        t = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        t.pre_tokenizer = Whitespace()
        pt = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>", unk_token="<unk>")
        pt.save_pretrained(out_dir)
        vocab_sz = 50

    cfg = RobertaConfig(
        vocab_size=vocab_sz,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=66,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = RobertaModel(cfg)
    model.save_pretrained(out_dir)
    return out_dir


def test_emobank_all_models_and_ablations(tiny_dir: Path, device: torch.device) -> None:
    tok = AutoTokenizer.from_pretrained(str(tiny_dir))
    texts = [
        "good happy policy but bad sad result",
        "policy not good",
        "happy good policy",
        "not sad but good",
        "bad result not happy",
        "policy policy",
    ]
    enc = tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    sf = torch.from_numpy(extract_struct_matrix(texts)).to(device)
    B = len(texts)
    r_probs = torch.softmax(torch.randn(B, 3, 5), dim=-1).to(device)
    w_probs = torch.softmax(torch.randn(B, 3, 5), dim=-1).to(device)
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    batch = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "struct_feats": sf,
        "r_probs": r_probs,
        "w_probs": w_probs,
        "r_mean": (r_probs * v).sum(-1),
        "w_mean": (w_probs * v).sum(-1),
        "r_std": torch.full((B, 3), 0.7, device=device),
        "w_std": torch.full((B, 3), 0.7, device=device),
        "r_has_hist": torch.ones(B, dtype=torch.bool, device=device),
        "w_has_hist": torch.tensor([True, True, False, True, False, True], device=device),
    }
    weights = {"kl": 1.0, "w1": 1.0, "pol_shape": 0.5, "w_hist_prev": 0.5, "r_hist_prev": 1.0, "wr_hist_prev": 0.5}
    amp_ok = device.type == "cuda"

    for mt in ("scalar_mse", "gaussian_nll", "nig_evidential", "softmax_ldl", "sord", "dldl_v2", "dc_ldl", "uw_ldl", "beta_nll", "patnet_unimodal", "patnet_no_transport", "patnet"):
        m = DualPerspectiveAffectModel(backbone_name=str(tiny_dir), model_type=mt, latent_dim=32, ablation_mode="M5_full_struct").to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp_ok):
            out = m(batch["input_ids"], batch["attention_mask"], struct_feats=batch["struct_feats"])
            loss, _ = compute_model_loss(mt, out, batch, weights)
        assert torch.isfinite(loss), f"NaN/Inf loss in {mt}"
        loss.backward()
        opt.step()

    for ab in ("M0_baseline", "M1_multilayer_conv", "M2_bipolar_pool", "M3_tension_route", "M4_disentangle", "M5_full_struct"):
        m = DualPerspectiveAffectModel(backbone_name=str(tiny_dir), model_type="patnet", latent_dim=32, ablation_mode=ab).to(device)
        with torch.amp.autocast("cuda", enabled=amp_ok):
            out = m(batch["input_ids"], batch["attention_mask"], struct_feats=batch["struct_feats"])
            loss, _ = compute_model_loss("patnet", out, batch, weights)
        assert torch.isfinite(loss)
        loss.backward()
    print("[PASS] test_emobank_all_models_and_ablations")


def test_epic_pipeline_synthetic(tiny_dir: Path, device: torch.device) -> None:
    tok = AutoTokenizer.from_pretrained(str(tiny_dir))
    df = pd.DataFrame({
        "parent_text": ["", "good policy", "bad result", "happy day"],
        "text": ["not good", "policy happy", "bad bad", "sad result"],
        "n_annot": [5, 6, 4, 5],
        "k_iro": [0, 3, 4, 2],
        "p_iro": [0.0, 0.5, 1.0, 0.4],
        "n_male": [3, 3, 2, 2],
        "k_male": [0, 0, 2, 1],
        "p_iro_male": [0.0, 0.0, 1.0, 0.5],
        "n_female": [2, 3, 2, 3],
        "k_female": [0, 3, 2, 1],
        "p_iro_female": [0.0, 1.0, 1.0, 0.33],
        "is_polarized": [False, True, False, True],
        "is_unanimous": [True, False, True, False],
        "group_key": ["g0", "g1", "g2", "g3"],
        "n_nat_uk": [2, 1, 0, 2], "k_nat_uk": [0, 1, 0, 1], "p_nat_uk": [0.0, 1.0, np.nan, 0.5],
        "n_nat_us": [1, 2, 2, 1], "k_nat_us": [0, 1, 2, 0], "p_nat_us": [0.0, 0.5, 1.0, 0.0],
        "n_nat_au": [0, 1, 1, 0], "k_nat_au": [0, 0, 1, 0], "p_nat_au": [np.nan, 0.0, 1.0, np.nan],
        "n_nat_ie": [1, 1, 1, 1], "k_nat_ie": [0, 1, 1, 1], "p_nat_ie": [0.0, 1.0, 1.0, 1.0],
        "n_nat_in": [1, 1, 0, 1], "k_nat_in": [0, 0, 0, 0], "p_nat_in": [0.0, 0.0, np.nan, 0.0],
    })
    ds = EpicDataset(df, extract_struct_matrix(df["text"].tolist()))
    batch = make_epic_collate_fn(tok, max_length=32)([ds[i] for i in range(4)])
    amp_ok = device.type == "cuda"
    for mt in ("scalar_bce", "beta_binom", "patnet_no_transport", "patnet"):
        m = EpicPerspectivistModel(backbone_name=str(tiny_dir), model_type=mt, work_dim=32).to(device)
        with torch.amp.autocast("cuda", enabled=amp_ok):
            out = m(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["seg_ids"].to(device), batch["struct_feats"].to(device))
            loss, _ = compute_epic_loss(mt, out, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        assert torch.isfinite(loss)
        loss.backward()
    print("[PASS] test_epic_pipeline_synthetic")


def test_brighter_pipeline_synthetic(tiny_dir: Path, device: torch.device) -> None:
    tok = AutoTokenizer.from_pretrained(str(tiny_dir))
    active = list(ALL_EMOS[:5])
    # Include deliberate NaNs in labels to test valid_mask protection!
    df = pd.DataFrame({
        "text": ["happy policy", "sad result not good", "虽然开心但是害怕！", "good"],
        "anger": [0.0, 2.0, np.nan, 1.0],
        "disgust": [np.nan, 1.0, 0.0, 0.0],
        "fear": [0.0, 3.0, 2.0, np.nan],
        "joy": [3.0, 0.0, 1.0, 2.0],
        "sadness": [0.0, 2.0, np.nan, 0.0],
    })
    ds = BrighterOrdinalDataset(df, active, extract_struct_matrix(df["text"].tolist()))
    assert not ds.valid_mask[0, 1] and ds.valid_mask[0, 0]

    for mt, use_bigru, bb in (
        ("dml_ordinal", False, str(tiny_dir)),
        ("uw_ldl", False, str(tiny_dir)),
        ("sord", False, str(tiny_dir)),
        ("dldl_v2", False, str(tiny_dir)),
        ("softmax_ce", False, str(tiny_dir)),
        ("gaussian_nll", False, str(tiny_dir)),
        ("scalar_mse", False, str(tiny_dir)),
        ("bigru_bipolar", True, "bigru_jieba"),
    ):
        collate = make_brighter_collate_fn(None if use_bigru else tok, use_bigru=use_bigru, max_length=32)
        b = collate([ds[i] for i in range(4)])
        m = BrighterOrdinalModel(backbone_name=bb, num_emos=len(active), model_type=mt, work_dim=32).to(device)
        out = m(b["input_ids"].to(device), b["attention_mask"].to(device), b["struct_feats"].to(device))
        loss = compute_brighter_loss(mt, out, b["y_int"].to(device), b["y_cont"].to(device), b["valid_mask"].to(device))
        assert torch.isfinite(loss)
        loss.backward()
    print("[PASS] test_brighter_pipeline_synthetic")


def test_classical_oof_pipeline() -> None:
    rng = np.random.default_rng(7)
    n = 30
    texts = [f"policy {i % 5} good {'' if i % 2 else 'but sad not happy'}" for i in range(n)]
    probs = rng.dirichlet(np.ones(5), size=(n, 3, 2))
    df_tr = pd.DataFrame({
        "id": [f"i{i}" for i in range(n)],
        "text": texts,
        "category": ["blog"] * n,
        "document": [f"d{i // 3}" for i in range(n)],
        "r_V_has_hist": [True] * n,
        "w_V_has_hist": [i % 2 == 0 for i in range(n)],
    })
    for persp, idx_p in (("w", 0), ("r", 1)):
        for j, d in enumerate(("V", "A", "D")):
            df_tr[f"{persp}_{d}_pub_mean"] = rng.uniform(2.0, 4.0, size=n)
            df_tr[f"{persp}_{d}_pub_std"] = rng.uniform(0.3, 1.0, size=n)
            for k in range(1, 6):
                df_tr[f"{persp}_{d}_p{k}"] = probs[:, j, idx_p, k - 1]
    sf = extract_struct_matrix(texts)
    X_tr, X_s, X_c, X_t = _build_sparse_design(texts, texts[:5], texts[:5], texts[:5], word_max=500, char_max=500)
    G_tr, G_s, G_c, G_t, rp = _extract_oof_and_gbdt_features_emobank(X_tr, X_s, X_c, X_t, df_tr, sf, sf[:5], sf[:5], sf[:5])
    assert G_tr.shape[0] == n and np.isfinite(G_tr).all() and rp["te_r_probs"].shape == (5, 3, 5)
    print("[PASS] test_classical_oof_pipeline")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running synthetic suite on device={device}")
    tiny_dir = build_tiny_roberta_dir(OUT_DIR / "_tiny_roberta")
    try:
        test_emobank_all_models_and_ablations(tiny_dir, device)
        test_epic_pipeline_synthetic(tiny_dir, device)
        test_brighter_pipeline_synthetic(tiny_dir, device)
        test_classical_oof_pipeline()
    finally:
        shutil.rmtree(OUT_DIR / "_tiny_roberta", ignore_errors=True)
    print("ALL SYNTHETIC PIPELINE CHECKS PASSED.")


if __name__ == "__main__":
    main()
