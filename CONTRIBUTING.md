# Contributing to PAT-Net

Thank you for your interest in contributing to **PAT-Net**.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verification Before Submitting a Pull Request

Please run offline syntax, unit math, synthetic pipeline, and repository hygiene checks:

```bash
python -m compileall -q src scripts
python scripts/verify_reproduction.py
python scripts/test_core.py
python scripts/test_pipelines_synthetic.py
```

Ensure no local machine paths, credentials, internal server addresses, raw third-party datasets (`data/raw/`), or large model checkpoints (`*.pt`, `*.safetensors`, `pretrained/`) are committed.
