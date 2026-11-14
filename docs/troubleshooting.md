# Troubleshooting

## `FileNotFoundError: .../data/tactical/BAR-ATH_tactical_atd.txt`

The match bundle is missing, in the wrong place, or the two required tracking files were renamed.
Prediction needs exactly:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
```

Check the layout in [dataset.md](dataset.md), or point `MTAH_DATA_DIR` at the folder that contains
`tactical/`:

```bash
export MTAH_DATA_DIR=/path/to/bundle
uv run python scripts/predict.py submission.csv
```

## The two match-data tests are skipped

`tests/conftest.py` skips them when `paths.DATA_DIR / "tactical"` is not a directory. The unit tests
still run. Fix the data path as above; no other configuration is involved.

## `ModuleNotFoundError: games` or `possession_metrics`

Those imports were removed in the local setup. The repository now carries the needed BAR-ATH
constants in `possession/competition.py` and label codes in `possession/external.py`. If you see
this error, the checkout is older than the setup fix; update it rather than cloning the organisers'
repository.

## The submission hash differs

The expected hash is in [reproducibility.md](reproducibility.md). Check, in order:

1. `cd models && sha256sum -c SHA256SUMS` — all 12 files must be `OK`.
2. `uv run pytest` — the full suite must pass; a stale model shows up here.
3. The Python and package versions: `uv sync` from the locked `uv.lock`, Python 3.14.
4. The tracking file is the original bundle file, not a re-exported or edited copy.
5. The CSV was written on Linux. Python's text mode translates `\n` on Windows, so the bytes (and
   therefore the checksum) differ even when the labels are identical.

## `FileNotFoundError: .../models/*.txt` or `*.pt`

The trained classifiers are missing. Restore the shipped files from `models/` (and verify
`SHA256SUMS`), or retrain with `uv run python scripts/train.py`.

## Training says an external file is missing

The competition bundle is not used for training. Download the external matches first:

```bash
uv run python scripts/download_external.py
```

This writes `external/metrica/` and `external/dfl/` and verifies
`scripts/external_sha256.txt`. If a download is interrupted, the `.partial` file is ignored and the
next run starts again.

## A retrain gives different rows or scores

The caches in `cache/` are keyed by kind and match name, not by a hash of the code. After changing
features, labels, augmentation, or the external files, delete the relevant cache:

```bash
rm -rf cache/rows cache/steps      # feature caches
rm -rf cache/matches               # parsed matches
```

`cache/` is disposable; rebuilding is slower but safe.

## `uv sync` fails or installs an unexpected PyTorch

The project uses the `pytorch-cu130` index from `pyproject.toml`. `uv sync` installs exactly what
`uv.lock` records. If the machine cannot reach that index, use a machine with network access or
pre-populate the uv cache; do not edit `pyproject.toml` casually, because the lock and the
reproduction claim depend on it. CPU-only inference still works with the CUDA wheel.

## The public scorer fails to import or find `possession_metrics`

The scorer lives in the organisers' repository. Clone it, then run its script from that repository
root (or pass absolute paths), as shown in [evaluation.md](evaluation.md). This repository no longer
needs the scorer for prediction or tests.

## `uv run pytest` cannot find tests

Run it from the repository root. `testpaths = ["tests"]` is set in `pyproject.toml`.

## Memory use during feature computation

`features._nearest_opponent_distance` processes frames in chunks of 5,000 specifically to bound the
temporary `(frames, slots, slots)` array. If you change that code, keep the chunking; at 46 slots
the full array would be hundreds of MB.
