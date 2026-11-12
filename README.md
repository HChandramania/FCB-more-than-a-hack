# Ball possession from automatic tracking — More than a Hack 2027

This repository labels each frame of the competition match (FC Barcelona – Athletic Club) as home
possession (`H`), away possession (`A`) or dead ball (`D`). It uses only the provided automatic
tracking. Every classifier is trained on other matches; no label of the competition match is
read anywhere in the code.

## Method in one paragraph

Per-frame signals of the players and the ball (team layout, speeds, touch timing, ball movement
along each attack direction) feed two kinds of classifier: LightGBM on multi-scale window
features, and a temporal convolutional network (TCN) at 5 steps per second. P(dead) averages four
classifiers and P(home | live) averages eight. A 3-state Viterbi decoder turns the per-frame
probabilities into possession runs. Then the control rule of `CHALLENGE.md` applies: a run of the
other team shorter than 2 s is a touch or a contest, not a change of possession. The external
tracking data has almost no gaps, while the competition tracking loses the ball on 41 % of
graded frames and about 7.6 % of player-frames. So the classifiers also train on copies with
simulated ball loss and simulated lost player tracks. Before inference, gaps of up to 10 s inside
one player track are filled by interpolation. All classifiers are exactly team-symmetric:
exchanging home and away mirrors P(home) and keeps P(dead).

## Results on the external matches

Nested leave-one-match-out validation: each match is scored by models that never saw it, with
settings chosen without it. spellF1 is the official ranking metric.

| validation set | spellF1 | spellF1, 10 % of player-frames removed |
|---|---|---|
| Metrica sample games 1–3 | 0.816 | 0.792 |
| Sportec Open DFL, 7 matches | 0.769 | 0.752 |

`docs/experiments.md` lists every experiment, including the rejected ones, with its numbers.

## Reproduce the submission

Requirements: Linux, Python 3.14, [uv](https://docs.astral.sh/uv/). A GPU is optional.

```bash
uv sync                                   # exact versions from uv.lock
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
# put the match data bundle into ../more-than-a-hack-2027/data/ (see that repository's DATA.md)
uv run python scripts/predict.py submission.csv
```

This writes the submitted file from the trained classifiers in `models/` (about 2 minutes on a
CPU). `sha256sum submission.csv` must give the value in `tests/test_match_data.py`.

To train the classifiers again from the external data:

```bash
uv run python scripts/download_external.py   # about 2.6 GB, checked against scripts/external_sha256.txt
uv run python scripts/train.py               # writes models/; about 30 minutes, TCNs on the GPU if present
uv run python scripts/predict.py submission.csv
```

A rebuild from empty folders on the development machine (RTX 5080, CUDA 13.0, the versions in
`uv.lock`) gives model files identical byte for byte to the shipped ones and an identical
submission. GPU training is deterministic
(`torch.use_deterministic_algorithms`). Other GPU models can round differently, so a retrained
TCN can differ in its last bits on other hardware; the shipped weights remove that dependency.
Prediction runs on the CPU only.
`models/SHA256SUMS` holds the checksums of the shipped weights.

Paths can be changed with environment variables: `MTAH_COMPETITION_REPO` (the organisers'
repository with `data/`), `MTAH_EXTERNAL_DIR`, `MTAH_CACHE_DIR` and `MTAH_MODEL_DIR`.

## Tests

```bash
uv run pytest
```

The unit tests need no data. The tests in `tests/test_match_data.py` need the match data. They
check the team symmetry of every classifier and that `scripts/predict.py` writes the submitted
file byte for byte.

## Data

| data | use | source and licence |
|---|---|---|
| Competition match: tactical tracking (ATD) and the graded-frame table | input of the prediction only | provided by the organisers; not redistributed here |
| Sportec Open DFL dataset, 7 matches | training and validation | Bassek et al. 2025, *Scientific Data* 12:195, [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. Labels: the provider's per-frame BallPossession and BallStatus flags |
| Metrica Sports sample data, games 1–3 | training and validation | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data). Labels: derived in code from Metrica's event data (`possession/external.py`) |

Not used: the broadcast video and its tracking, the Smart Tagging, the homography, and the public
5-minute ground truth (only the organisers' `run_evaluation.py` reads it, as a pipeline check).
No external data covers the competition match.

## Layout

```
possession/      the package
  data.py          match loaders, the control rule, the graded frames
  external.py      readers of the DFL and Metrica files and the Metrica label derivation
  augment.py       simulated ball and player loss for training; the player-gap fill
  features.py      per-frame signals
  gbm.py           LightGBM classifiers
  tcn.py           TCN classifiers
  decoding.py      Viterbi decoder
  pipeline.py      the submitted model: ensembles, decoder, control rule
scripts/         download_external.py, train.py, predict.py
models/          trained classifiers of the submission
tests/           unit tests and checks on the match data
docs/            experiment log
```
