"""Portable workspace paths and offline/shared backbone resolution for PAT-Net."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
PRETRAINED_DIR = ROOT / "pretrained"
OUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"


def resolve_backbone_path(name: str) -> str:
    """Resolve pretrained model path from explicit path, local pretrained/, optional env hub cache, or HF ID."""
    p = Path(name)
    if p.exists():
        return str(p)
    local_dir = PRETRAINED_DIR / name.replace("/", "__")
    if local_dir.exists() and (local_dir / "config.json").exists():
        return str(local_dir)
    raw_dir = PRETRAINED_DIR / name
    if raw_dir.exists() and (raw_dir / "config.json").exists():
        return str(raw_dir)
    shared_hub = os.environ.get("PATNET_SHARED_HF_HUB", "").strip()
    if shared_hub:
        hub_dir = Path(shared_hub) / f"models--{name.replace('/', '--')}" / "snapshots"
        if hub_dir.exists():
            snaps = sorted([d for d in hub_dir.iterdir() if d.is_dir() and (d / "config.json").exists()])
            if snaps:
                return str(snaps[-1])
    return name
