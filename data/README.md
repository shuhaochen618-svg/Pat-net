# Benchmark Data Preparation & Provenance

This directory holds the distribution-preserving benchmark datasets used across **PAT-Net**.

## 1. Included Processed Splits (`data/processed/`)

For zero-friction offline verification and primary/perspectivist experimentation, curated `.parquet` / `.npz` artifacts are provided under `data/processed/`:

| File | Corpus | Task & Disaggregated Labels | Size / Coverage |
| :--- | :--- | :--- | :--- |
| `emobank_full.parquet` | EmoBank (`JULIELab/EmoBank`) | Dual-perspective (Writer & Reader) discrete rating histograms over `{1..5}` across Valence, Arousal, Dominance + genre & document IDs | `10,062` sentences (`136` unique documents, `0` unmatched QC items) |
| `emobank_distributions.parquet` | EmoBank QC Histograms | Reconstructed exact per-rating Writer & Reader histograms | `10,062` sentences |
| `epic_agg.parquet` | EPIC (`Multilingual-Perspectivist-NLU/EPIC`) | Perspectivist irony disagreement (`k_iro / n_annot`) + rater gender & nationality breakdowns | `3,000` items (`14,172` annotations, `74` raters, strict `parent_text` grouping) |
| `goemotions_raw_agg.parquet` | GoEmotions Raw (`google-research-datasets/go_emotions`) | 28-emotion soft label proportions + `example_very_unclear` ambiguity vs. clear polarization flags | `58,011` unique items (`211,225` annotations, `82` raters) |
| `goemotions_cooccur_f16.npz` | GoEmotions Rater Co-occurrence | Exact second-order empirical label co-occurrence matrices | `58,011` items (`28 x 28` float16) |

## 2. Raw Corpora & Multilingual BRIGHTER Splits (`data/raw/`)

Raw third-party corpora remain subject to their respective upstream licenses and are excluded by `.gitignore`. Download all verified raw corpora automatically:

```bash
python scripts/download_data.py
```

This fetches:
1. **EmoBank**: `JULIELab/EmoBank` CSVs into `data/raw/emobank/`
2. **GoEmotions Raw**: raw GCS CSVs (`goemotions_{1,2,3}.csv`) and Parquet splits into `data/raw/go_emotions/`
3. **EPIC**: `Multilingual-Perspectivist-NLU/EPIC` (`EPICorpus.csv`) into `data/raw/epic/`
4. **BRIGHTER**: `brighter-dataset/BRIGHTER-emotion-intensities` Parquet splits (`chn`, `eng`, `deu`, `esp`, `ptbr`, `rus`) into `data/raw/brighter/` (required when running `BRIGHTER` ordinal calibration benchmarks)

To rebuild and audit `data/processed/` from raw corpora:

```bash
python scripts/prepare_and_audit_emobank.py
python scripts/prepare_and_audit_epic.py
python scripts/prepare_and_audit_goemotions.py
```
