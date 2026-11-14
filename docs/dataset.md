# Dataset

## Where the data lives

The repository expects the gated match bundle in **`data/`** by default. The folder already
contains the current checkout's bundle; this is the layout the code was written for:

```
data/
├─ tactical/                                  the evaluation reference (required subset)
│  ├─ BAR-ATH_tactical_atd.txt                raw ATD tracking: players and ball
│  ├─ BAR-ATH_tactical_atd.xml                team and player metadata
│  ├─ BAR-ATH_tactical_homography.csv         one 3x3 matrix per frame
│  ├─ BAR-ATH_tactical_smart_tagging.xml      638 tactical phases and set pieces
│  └─ BAR-ATH_tactical-002.mp4                tactical video, 199,319 frames
└─ broadcast/                                 optional, NOT synchronised to tactical
   ├─ BAR-ATH_broadcast_atd.txt
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_homography.csv
   └─ BAR-ATH_broadcast-001.mp4
```

The MP4 suffix (`-002`, `-001`) is how the bundle was downloaded. `possession/competition.py`
accepts either the canonical `BAR-ATH_tactical.mp4` name or a single `...-*.mp4` variant when it
resolves the video path.

Set `MTAH_DATA_DIR` to use a different root; the `tactical/` and `broadcast/` subfolders must keep
their names.

## Which files the pipeline actually uses

| File | Used? | Role |
|---|---|---|
| `BAR-ATH_tactical_atd.txt` | **yes** | One row per tracking frame: 46 slots × 3 columns (x, y, z) plus ball (x, y). Parsed as float64. |
| `BAR-ATH_tactical_atd.xml` | **yes** | Player → team mapping and player order, which fixes the 46 tracking slots. |
| `BAR-ATH_tactical.mp4` | no | The frame-index authority and event-playback reference. Never opened. |
| `BAR-ATH_tactical_homography.csv` | no | Broadcast-to-pitch mapping. Tracking is already in pitch coordinates. |
| `BAR-ATH_tactical_smart_tagging.xml` | no | Phase/set-piece annotations. Not needed for frame labels. |
| `data/broadcast/*` | no | A second camera with its own timeline. Not aligned to the submission frames. |

The challenge permits using only the automatic tactical tracking. That is exactly what the
pipeline does: video, broadcast tracking, homography and Smart Tagging are optional inputs that
were not used.

## Coordinates and timelines

- ATD positions are normalised to `[0, 1]` and multiplied by the real pitch size
  **(105.3 m, 68.0 m)**. Positions are metres from the pitch corner. The ball is in the same frame.
- Tracking frame IDs are 1-based and contiguous. The loader checks this and derives
  `video_frame = frame_id - 1`.
- The two halves have different tracking-to-video offsets (`1702057` and `1702058`). The resulting
  graded video ranges are:

| Half | Video frames | Frames |
|---|---|---|
| 1 | 11,185 – 90,651 | 79,467 |
| 2 | 110,808 – 190,197 | 79,390 |
| Total graded | | **158,857** |

Everything outside those blocks is written as `X` and is never scored. Halftime is a 20,156-frame
gap in the middle of the submission.

## Missing tracking

The competition tracking is less complete than the external training data:

- the ball is absent on **40.69 %** of graded frames;
- about **7.6 %** of player-frames are missing.

The pipeline handles this in three places: classifiers are trained on copies with simulated
ball/player loss, short interior player gaps (≤ 10 s) are linearly interpolated before inference,
and the encoder uses presence masks so a missing signal is distinguishable from a zero.

## Labels

The competition match has **no ground truth in this repository**. Nothing in `data/` is read as a
label. The public five-minute ground truth is part of the organisers' evaluation repository
(`evaluation/gt/BAR-ATH_possession_gt_public.csv`) and is only used by their scorer. The pipeline
never reads it.

## External training data

Training uses ten other matches under **`external/`**:

```
external/
├─ metrica/     Metrica Sports sample games 1, 2 and 3
└─ dfl/         seven Sportec Open DFL matches
```

`uv run python scripts/download_external.py` downloads them (about 2.6 GB) and verifies
`scripts/external_sha256.txt`. They are not needed for prediction. The sources and licences are:

- **Sportec Open DFL** — Bassek et al. 2025, *Scientific Data* 12:195,
  [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. Labels come from
  the provider's per-frame `BallPossession` and `BallStatus` flags.
- **Metrica Sports sample data**, games 1–3 —
  [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data). Metrica publishes
  events rather than per-frame possession, so labels are derived from the event stream in
  `possession/external.py`.

No external match covers the competition match.

## Gated data and redistribution

The match data is licensed to competition participants only. It is **not** covered by the
repository's MIT licence and must not be committed or redistributed, including in a public
submission fork. `.gitignore` therefore excludes `/data/`, `/cache/`, `/external/`, `*.mp4` and
`*.csv` (except `scripts/*.csv`).

## Cache

`cache/` is created on demand and can be deleted at any time:

| Folder | Contents |
|---|---|
| `cache/matches/` | Parsed competition and external matches as `.npz`. |
| `cache/rows/` | LightGBM training rows, one file per match and kind. |
| `cache/steps/` | TCN inputs at 5 steps/s, one file per match and augmentation copy. |

Rebuilding the cache is cheap compared with retraining.
