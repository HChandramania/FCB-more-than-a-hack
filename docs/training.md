# Training

Prediction uses the shipped `models/` and does not need anything on this page. Retraining needs
the external matches, which are **not** the competition match.

## External data

| Set | Matches | Labels | Source |
|---|---|---|---|
| Metrica sample | 1, 2, 3 | derived from the event stream by `possession/external.py` | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), fixed commit `e706dd5…` |
| Sportec Open DFL | `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9` | provider `BallPossession` + `BallStatus` per frame | Bassek et al. 2025, *Scientific Data* 12:195, [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0 |

`scripts/download_external.py` downloads both sets (about 2.6 GB) into `external/` and checks every
file against `scripts/external_sha256.txt`. Existing files with the right checksum are skipped.

```bash
uv run python scripts/download_external.py
```

The Metrica labels are an approximation of the competition convention: a team keeps possession
until an opponent makes an on-ball event, and the ball is dead from a stoppage event to the next
on-ball event. The derivation is code, not hand annotation, and is documented in
`possession/external.py`.

## Train everything

```bash
uv run python scripts/train.py
```

This fits the four LightGBM kinds and the eight TCN kinds, then writes `models/`. LightGBM runs on
the CPU; each TCN takes about 15 s on an RTX 5080 and some minutes on a CPU. The full rebuild is
about 30 minutes on the development machine.

Train a subset:

```bash
uv run python scripts/train.py --kinds dead_team_sym team_ball_advance tcn_dead
```

The valid kinds are `gbm.KINDS + tcn.KINDS`:

```
dead_team_sym  dead_team_sym_pd  team_ball_advance  team_ball_advance_pd
tcn_dead  tcn_dead_pd  tcn_team_change  tcn_team_change_s1  tcn_team_change_s2
tcn_team_change_pd  tcn_team_change_s1_pd  tcn_team_change_s2_pd
```

After training, `uv run python scripts/predict.py submission.csv` must again give the checksum in
[reproducibility.md](reproducibility.md). If a retrained TCN rounds differently on other hardware,
the shipped weights are the authority.

## How training data is built

1. Load an external match with `data.load_dfl` or `data.load_metrica`. Labels use the competition
   `H/A/D/X` codes and the 2 s control rule.
2. For team kinds, add ball-loss copies at 0 % (live placement), 15 % and 30 % (uniform
   placement). For every `_pd` kind, add a copy with 10 % player dropout.
3. Compute the signals in `features.py` at 25 fps.
4. LightGBM: keep every 5th labelled frame and append the team-exchanged copy. Rows are cached in
   `cache/rows/`.
5. TCN: average blocks of 5 frames to 5 steps/s, standardise with pooled `_H`/`_A` statistics and
   append a presence mask. Inputs are cached in `cache/steps/`.

Metrica and DFL receive equal total weight in both families, so the 7 DFL matches do not
outweigh the 3 Metrica matches.

## Caches

| Folder | Key | Rebuild cost |
|---|---|---|
| `cache/matches/` | match name | seconds to minutes |
| `cache/rows/` | base kind + match | feature computation per match |
| `cache/steps/` | signal set + match + placement + share | feature computation per copy |

Delete `cache/` after changing features, labels, augmentation or the external data. The cache is
keyed by kind and match name, not by a code hash, so stale files are the most likely cause of a
surprising retrain.

## Determinism

- **LightGBM** fits are deterministic. Two fits from the same rows give identical predictions.
- **TCN** training sets `torch.use_deterministic_algorithms(True)`,
  `torch.backends.cudnn.benchmark = False` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. On the same GPU
  architecture, CUDA and PyTorch versions, two runs give identical weights.
- Different GPU hardware can round differently. The shipped weights remove that dependency, and
  prediction is CPU-only.

The README records a rebuild from empty folders on the development machine (RTX 5080, CUDA 13.0,
the locked versions) that produced model files identical byte for byte to the shipped ones.

## Validation protocol

The reported external scores use **nested leave-one-match-out** validation: each match is scored by
models that never saw it, and hyperparameters are chosen without it. Both sets are weighted
equally. The competition match and its labels are never part of this loop.

The experiment scripts and raw validation reports referenced by [experiments.md](experiments.md)
are **not shipped** in this checkout. The training and prediction code is complete; the research
workspace that produced the nested validation numbers is not. This is the main reproducibility gap
(see [reproducibility.md](reproducibility.md)).
