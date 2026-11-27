"""Robust streaming downloader for all verified open datasets used in 002idea."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def download_stream(urls: list[str], dest: Path, timeout: int = 30) -> None:
    """Download file via streaming HTTP with timeout, retry, and atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        print(f"[SKIP] {dest.relative_to(RAW_DIR)} ({dest.stat().st_size / 1024:.1f} KB)")
        return
    if dest.exists():
        dest.unlink()

    last_err: Exception | None = None
    for url in urls:
        for attempt in range(2):
            try:
                with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    total = 0
                    with tmp.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                                total += len(chunk)
                    if total < 50:
                        raise ValueError(f"Empty/tiny response ({total} B) from {url}")
                    tmp.replace(dest)
                print(f"[OK] {dest.relative_to(RAW_DIR)}: {total / 1024:.1f} KB ({url.split('/')[2]})")
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[WARN] attempt {attempt+1} failed for {url}: {exc}")
                time.sleep(1.0)
    raise RuntimeError(f"Failed downloading {dest}: {last_err}")


def download_emobank() -> None:
    eb_dir = RAW_DIR / "emobank"
    bases = [
        "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/",
        "https://ghfast.top/https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/",
    ]
    for fn in [
        "emobank.csv",
        "individual_writer_ratings.csv",
        "individual_reader_ratings.csv",
        "writer.csv",
        "reader.csv",
        "meta.tsv",
    ]:
        download_stream([b + fn for b in bases], eb_dir / fn)


def hf_resolve_urls(repo_id: str, rel_path: str) -> list[str]:
    return [
        f"https://hf-mirror.com/datasets/{repo_id}/resolve/main/{rel_path}",
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}",
    ]


def clean_stale_hf_cache(folder: Path) -> None:
    cache_dir = folder / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def download_hf_datasets() -> None:
    # 1. GoEmotions (original GCS raw CSVs with true example_very_unclear + simplified)
    ge_dir = RAW_DIR / "go_emotions"
    clean_stale_hf_cache(ge_dir)
    for idx in (1, 2, 3):
        download_stream(
            [f"https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_{idx}.csv"],
            ge_dir / "full_dataset" / f"goemotions_{idx}.csv",
            timeout=30,
        )
    for rel in [
        "raw/train-00000-of-00001.parquet",
        "simplified/train-00000-of-00001.parquet",
        "simplified/validation-00000-of-00001.parquet",
        "simplified/test-00000-of-00001.parquet",
    ]:
        download_stream(hf_resolve_urls("google-research-datasets/go_emotions", rel), ge_dir / rel, timeout=30)

    # 2. BRIGHTER (6 representative languages: eng, chn, deu, esp, rus, ptbr)
    b_dir = RAW_DIR / "brighter"

    for lang in ["eng", "chn", "deu", "esp", "rus", "ptbr"]:
        for split in ["train", "dev", "test"]:
            rel = f"{lang}/{split}-00000-of-00001.parquet"
            download_stream(
                hf_resolve_urls("brighter-dataset/BRIGHTER-emotion-intensities", rel),
                b_dir / rel,
                timeout=30,
            )

    # 3. EPIC & MultiPICo (optional slow mirror fallback at end)
    download_stream(
        hf_resolve_urls("Multilingual-Perspectivist-NLU/EPIC", "EPICorpus.csv"),
        RAW_DIR / "epic" / "EPICorpus.csv",
        timeout=25,
    )
    try:
        download_stream(
            hf_resolve_urls("Multilingual-Perspectivist-NLU/MultiPICo", "multipico_anonimized.csv"),
            RAW_DIR / "multipico" / "multipico_anonimized.csv",
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MultiPICo deferred due to upstream stall: {exc}")


def main() -> None:
    print("=== 1. EmoBank ===")
    download_emobank()
    print("\n=== 2. GoEmotions / EPIC / MultiPICo / BRIGHTER ===")
    download_hf_datasets()
    print("\nAll verified datasets downloaded cleanly.")


if __name__ == "__main__":
    main()
