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

Historical development results, not independently reproduced in the final audit. The development
log describes nested leave-one-match-out selection, but its experiment scripts and per-fold records
are not included here. Out-of-sample predictions and nested selection cannot be independently
verified from this checkout. spellF1 is the official ranking metric.

| validation set | spellF1 | spellF1, 10 % of player-frames removed |
|---|---|---|
| Metrica sample games 1–3 | 0.816 | 0.792 |
| Sportec Open DFL, 7 matches | 0.769 | 0.752 |

These four rounded values originate in this README; the available experiment log does not provide
the per-fold values needed to recalculate them. macroF1 was not recorded in the available final-result
evidence. See [external evaluation audit](docs/external_evaluation.md).

## Reproduce the submission

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/). Windows CPU inference has passed the
local audit. A GPU is not required for prediction. Authorized access to both tactical ATD files
(TXT and original XML) is required.

```bash
uv sync --locked                          # exact versions from uv.lock
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
# put the match data bundle into ../more-than-a-hack-2027/data/ (see that repository's DATA.md)
uv run python scripts/predict.py submission.csv
```

This writes a new file from the trained classifiers in `models/`. The audited Windows CPU run took
300.91 seconds. Do not run this command over an existing validated submission. Use
[reproduction instructions](docs/reproduction.md) for an isolated output, explicit dataset paths,
model checksums, and a user-run byte-for-byte comparison that preserves the original CSV.

To train the classifiers again from the external data:

```bash
uv run python scripts/download_external.py   # about 2.6 GB, checked against scripts/external_sha256.txt
uv run python scripts/train.py               # writes models/; about 30 minutes, TCNs on the GPU if present
uv run python scripts/predict.py submission.csv
```

The development log reports identical rebuilds on its original machine; these claims were not
independently reproduced in the final audit. The current Windows submission passed the official
format validator, but a second-run hash comparison has not been performed. The historical hash in
`tests/test_match_data.py` has not been checked against that submission. Windows writes CRLF while
Linux normally writes LF; both validate, but their file hashes differ. Prediction uses the CPU.
Deterministic training settings are present, but identical retraining across hardware is not assured.
`models/SHA256SUMS` holds the checksums of the shipped weights.

Paths can be changed with environment variables: `MTAH_COMPETITION_REPO` (the organisers'
repository), `MTAH_DATA_DIR` (dataset root containing `tactical/`), `MTAH_EXTERNAL_DIR`,
`MTAH_CACHE_DIR` and `MTAH_MODEL_DIR`. Use a fresh cache for a new dataset: match caches do not
validate source provenance. Keep the LightGBM files byte-identical; `.gitattributes` disables
Git text conversion for `models/*.txt`.

## Tests

```bash
uv run --no-sync python -B -m pytest -q -p no:cacheprovider tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

The ten selected unit tests need no match data. Do not run the complete suite automatically:
`tests/test_match_data.py` invokes full-match inference. Its skip condition checks the default
dataset directory rather than `MTAH_DATA_DIR`, so it can skip despite a configured external dataset.

## Data

| data | use | source and licence |
|---|---|---|
| Competition match: tactical tracking (ATD) and the graded-frame table | input of the prediction only | provided by the organisers; not redistributed here |
| Sportec Open DFL dataset, 7 matches | training and validation | Bassek et al. 2025, *Scientific Data* 12:195, [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. Labels: the provider's per-frame BallPossession and BallStatus flags |
| Metrica Sports sample data, games 1–3 | training and validation | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data). Labels: derived in code from Metrica's event data (`possession/external.py`) |

Not used: the broadcast video and its tracking, the Smart Tagging, the homography, and the public
5-minute ground truth (only the organisers' `run_evaluation.py` reads it, as a pipeline check).
No external data covers the competition match.

## Submission documents and privacy

The [technical report](docs/technical_report.md) describes the implementation, provenance,
limitations and evidence status. The [readiness checklist](docs/submission_readiness.md) records
remaining delivery checks. The final filenames must use the registered team name; the validated
working CSV is preserved separately. Markdown is an accepted technical-document format.

Do not commit provided data, caches, generated predictions or secrets. Do not send provided data,
predictions or local logs to AI services. Run any further match validation or reproduction locally
yourself and share sanitized PASS/FAIL results only. Do not evaluate against competition ground
truth, inspect video to check predictions, or tune on this match.

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
