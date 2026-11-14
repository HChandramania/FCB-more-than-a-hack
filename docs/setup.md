# Setup

## Requirements

- Linux (the tested platform), Python **3.14**, [uv](https://docs.astral.sh/uv/).
- About 8 GB of disk for the environment and the model files; the match bundle is about 20 GB.
- A GPU is optional. Prediction and all tests run on the CPU.

The exact Python packages are pinned in `uv.lock` (`lightgbm`, `numba`, `numpy`, `pandas`,
`torch`, `pytest`).

## 1. Install the environment

```bash
uv sync
```

This creates `.venv/` from `uv.lock`. The PyTorch wheel comes from the `pytorch-cu130` index
declared in `pyproject.toml`; on a machine without CUDA it still runs on the CPU.

## 2. Put the match data in `data/`

The bundle is gated by the organisers. Unpack it so that the repository looks like this:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical_atd.txt             required by prediction
│  ├─ BAR-ATH_tactical_atd.xml             required by prediction
│  ├─ BAR-ATH_tactical_homography.csv      optional, unused
│  ├─ BAR-ATH_tactical_smart_tagging.xml   optional, unused
│  └─ BAR-ATH_tactical-002.mp4             optional, unused (the frame-index reference)
└─ broadcast/
   ├─ BAR-ATH_broadcast_atd.txt            optional, unused
   ├─ BAR-ATH_broadcast_atd.xml            optional, unused
   ├─ BAR-ATH_broadcast_homography.csv     optional, unused
   └─ BAR-ATH_broadcast-001.mp4            optional, unused (not synchronised to tactical)
```

Only the two tactical ATD files are read. The MP4 names in the downloaded bundle may carry a
`-001`/`-002` suffix; no code in this repository opens a video. Keep the tracking file names
exactly as shown.

The optional `data/README.md` repeats the gated-data warning for the local checkout;
[dataset.md](dataset.md) is the tracked description of the layout and the gating rules.

To keep the data somewhere else, set `MTAH_DATA_DIR` to the folder that contains `tactical/` and
`broadcast/`:

```bash
export MTAH_DATA_DIR=/mnt/big-disk/mtah2027
```

## 3. Verify the checkout

```bash
uv run pytest
```

Expected: **12 passed** in roughly 3–4 minutes. The two match-data tests load the real match:
one checks team symmetry and the other recomputes the submission byte for byte. They are skipped
if `data/tactical/` is absent. The unit tests (control rule, Viterbi, gap fill) need no data.

Write the submission:

```bash
uv run python scripts/predict.py submission.csv
sha256sum submission.csv
```

The checksum must be:

```
2467a1396f08f2e0f4f65bb8ae261da82f44cae9678b8547dcbe4d2b17ac6f8e
```

Prediction takes about two minutes on a CPU and writes `frame,label` for all 199,319 frames.
Frames outside the two halves are `X`.

## 4. Optional: evaluate

The official scorer lives in the organisers' repository, not here. See
[evaluation.md](evaluation.md) for the command and the meaning of the metrics.

## 5. Optional: train

Training needs the external matches (about 2.6 GB), not the competition bundle. See
[training.md](training.md).

## Path overrides

| Variable | Default | Meaning |
|---|---|---|
| `MTAH_DATA_DIR` | `<repo>/data` | Folder containing `tactical/` and `broadcast/`. |
| `MTAH_EXTERNAL_DIR` | `<repo>/external` | Folder containing `dfl/` and `metrica/`. |
| `MTAH_CACHE_DIR` | `<repo>/cache` | Parsed matches and feature tables; safe to delete. |
| `MTAH_MODEL_DIR` | `<repo>/models` | Trained classifier files. |

All paths are resolved in `possession/paths.py`; nothing else hard-codes a folder.
