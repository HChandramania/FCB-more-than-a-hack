# Reproducibility

## What is pinned

| Item | Pinned by |
|---|---|
| Python | `requires-python = "==3.14.*"` |
| Packages | `uv.lock` (lightgbm 4.7.0, numba 0.66.0, numpy 2.4.6, pandas 3.0.5, torch 2.13.0) |
| Shipped classifiers | `models/SHA256SUMS` |
| Competition constants | `possession/competition.py`, copied from the organisers' `games.py` at commit `f8f2629ff1e0985f81d7e59d8413678514ee1530` |
| External downloads | `scripts/external_sha256.txt` |

Check the model files:

```bash
cd models && sha256sum -c SHA256SUMS
```

All 12 lines must read `OK`.

## Verified path

These checks were run in this checkout with the match bundle in `data/`:

| Check | Result |
|---|---|
| `uv sync` from `uv.lock` | succeeds |
| `uv run pytest` | **12 passed** |
| `uv run python scripts/predict.py submission.csv` | writes 199,319 rows |
| `sha256sum submission.csv` | `2467a1396f08f2e0f4f65bb8ae261da82f44cae9678b8547dcbe4d2b17ac6f8e` |
| Official `validate_submission.py` | **OK**: 199,319 rows, frames 0–199,318 |
| Official public `run_evaluation.py` | macro-F1 0.9391, spellF1 0.6667 |

The two match-data tests recompute the submission and compare the hash, so a passing suite is
also a byte-for-byte reproduction check. Prediction is CPU-only.

## What can be rebuilt

- **Prediction** from the shipped models: byte-identical, tested.
- **Training** from the external data: code and downloader are here. LightGBM is deterministic;
  TCNs are deterministic on the same hardware/CUDA/PyTorch combination. The README records a
  rebuild on the development machine that produced byte-identical model files.
- **Caches**: always regenerable; deleting `cache/` is safe.

## What cannot be rebuilt from this checkout

1. **The external validation figures.** The Metrica and DFL matches are not included, and the
   experiment scripts and raw validation logs (`scripts/experiments/*`,
   `results/experiment_logs/*`) named in [experiments.md](experiments.md) are not shipped.
2. **The nested selection loop.** The production settings are implemented, but the code that
   re-ran every leave-one-match-out fold is not part of this repository.
3. **The private full-match score.** It is not knowable locally; only the public window can be
   scored.
4. **The match data itself.** Gated and intentionally excluded from git.

## Why the TCN determinism matters

The submission must be the same file on every run. Two sources of nondeterminism were removed:

- `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- pooled `_H`/`_A` normalisation statistics and swapped training crops, so the network is exactly
  team-symmetric rather than approximately so.

Older TCN files that predate these fixes were retrained before submission. The shipped weights are
the authority if a different GPU rounds differently.

## Environment caveats

- The lock resolves a CUDA 13.0 PyTorch wheel even on a CPU-only machine; inference still works.
- Training on a different GPU model may round differently. That affects retraining, not the
  shipped predictions.
- `scripts/download_external.py` depends on the figshare API and the Metrica GitHub raw URLs. If a
  URL moves, the checksums still tell you whether a file is the expected one.
