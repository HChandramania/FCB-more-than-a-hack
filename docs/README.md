# Documentation

Short, task-oriented notes. Start with **setup** if you just want to reproduce the submission;
read **architecture** and **model** if you want to change it.

| Document | What it answers |
|---|---|
| [setup.md](setup.md) | Install, place the data, run the tests and write a submission. |
| [dataset.md](dataset.md) | What goes in `data/`, which files the pipeline uses, external training data, and the gating rules. |
| [architecture.md](architecture.md) | End-to-end data flow and the responsibility of every module. |
| [model.md](model.md) | Features, the 12 classifiers, the ensemble, the Viterbi decoder and the control rule. |
| [training.md](training.md) | How the models were trained, the external labels, caches and determinism. |
| [evaluation.md](evaluation.md) | Official metrics, verified results, the public-window check and its limits. |
| [reproducibility.md](reproducibility.md) | Checksums, the exact reproduction path, and what cannot be rebuilt from this checkout. |
| [troubleshooting.md](troubleshooting.md) | Common setup errors and their fixes. |
| [third-party.md](third-party.md) | Licences for the vendored constants and the data. |
| [experiments.md](experiments.md) | The full experiment log that chose every setting (version v16 is the submission). |

## Conventions used everywhere

- A **frame** is a 0-based **video** frame of the tactical recording. The submission has one
  row per frame, `frame,label`, for all 199,319 frames.
- Labels are `H` (home possession), `A` (away possession), `D` (dead ball) and `X` (no
  label / abstain). Only the two playing halves are graded; every other frame is `X`.
- Tracking runs at **25 fps**. Internal network steps are 5 per second (one step is 5 frames).
- `P(home)` below always means `P(home | live)`; `P(dead)` is separate.
- "The match" is **FC Barcelona – Athletic Club** (`BAR-ATH`), the only competition match.

## Repository map

```
possession/          the Python package
  competition.py     BAR-ATH constants and asset paths (vendored from the organisers' games.py)
  paths.py           data/, external/, cache/ and models/ locations
  data.py            match loaders, graded frames, control rule
  external.py        DFL and Metrica readers plus Metrica label derivation
  augment.py         simulated ball/player loss; player-gap interpolation
  features.py        per-frame signals and multi-scale windows
  gbm.py             LightGBM classifiers
  tcn.py             temporal convolutional classifiers
  decoding.py        3-state Viterbi decoder
  pipeline.py        the submitted model
scripts/             download_external.py, train.py, predict.py
models/              12 trained classifiers and SHA256SUMS
tests/               unit tests plus checks on the real match
data/                the match bundle (gated, not committed)
external/            external training matches (not committed)
cache/               regenerable parsed matches and feature tables
docs/                this folder
```
