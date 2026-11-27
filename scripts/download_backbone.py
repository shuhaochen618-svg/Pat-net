"""Download PLM backbone cleanly via streaming requests into local pretrained/ dir."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = ROOT / "pretrained"


def fetch_model_files(repo_id: str) -> None:
    target_dir = PRETRAINED_DIR / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_urls = [
        f"https://hf-mirror.com/api/models/{repo_id}",
        f"https://huggingface.co/api/models/{repo_id}",
    ]
    siblings = None
    for mu in meta_urls:
        try:
            r = requests.get(mu, timeout=15)
            if r.status_code == 200:
                siblings = [s["rfilename"] for s in r.json().get("siblings", [])]
                break
        except Exception:  # noqa: BLE001
            continue
    if siblings is None:
        raise RuntimeError(f"Could not fetch file listing for {repo_id}")

    wanted = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
        "model.safetensors",
    }
    if "model.safetensors" not in siblings and "pytorch_model.bin" in siblings:
        wanted.add("pytorch_model.bin")

    to_get = [f for f in siblings if f in wanted]
    print(f"Downloading {repo_id}: {to_get}")
    for fname in to_get:
        dest = target_dir / fname
        if dest.exists() and dest.stat().st_size > 100:
            print(f"[SKIP] {fname} ({dest.stat().st_size / 1024:.1f} KB)")
            continue
        urls = [
            f"https://hf-mirror.com/{repo_id}/resolve/main/{fname}",
            f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
        ]
        ok = False
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    total = 0
                    with tmp.open("wb") as fp:
                        for chunk in r.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                fp.write(chunk)
                                total += len(chunk)
                    tmp.replace(dest)
                    print(f"[OK] {fname}: {total / (1024*1024):.2f} MB")
                    ok = True
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {url} failed: {exc}")
        if not ok:
            raise RuntimeError(f"Failed downloading {fname} for {repo_id}")
    print(f"Ready locally at: {target_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="roberta-base")
    args = p.parse_args()
    fetch_model_files(args.repo)
