# Visca Barca: baseline possession classifier

This is a fresh-history publication package of the submitted baseline from source commit
`190fbdefcec376f674727a194cd8ac34315a7d70`. It contains no experimental temporal-isolation
changes. Production Python files, tests, pretrained checkpoints, `pyproject.toml` and `uv.lock`
are byte-identical to that baseline. Only documentation and `.gitattributes` were added or corrected.

## Verified exact reproduction

A fresh Windows CPU inference using the baseline and all twelve shipped checkpoints reproduced
the preserved `Visca Barca.csv` byte for byte. Both complete SHA-256 and binary comparison passed;
the official CSV validator exited 0. Both files were 1,882,093 bytes. Fresh inference took 244.98 s.

Verified SHA-256:

```text
f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef
```

The audit environment was Windows x86-64, CPython 3.14.6, NumPy 2.4.6, pandas 3.0.5,
LightGBM 4.7.0, numba 0.66.0 and PyTorch 2.13.0+cu130. Prediction runs on CPU; CUDA training
is not needed. Installed package versions matched `uv.lock`, and dependency consistency passed.
The original submission's complete creation environment is not independently documented; these
are the verified fresh-reproduction environment and result, not a claim about its original machine.

See [reproduction instructions](docs/reproduction.md), [technical report](docs/technical_report.md),
and [sanitized reproduction evidence](docs/reproduction_verification.json).

## Method

Only authorized tactical ATD tracking and its original XML slot-team metadata are prediction
inputs. The organizers' `games.py` and metric module supply constants and label definitions.
Player gaps of up to ten seconds are interpolated, then non-half positions are masked. LightGBM
and temporal convolutional classifiers provide four P(dead) and eight P(home | live) ensemble
members. A full-timeline three-state Viterbi decoder uses team/live-dead costs of 2.5/20 nats,
followed by the two-second control rule. The writer masks ungraded frames to X.

The baseline fills gaps before masking, uses centered temporal features, and decodes the full
timeline. It does not independently process every half or valid span. Temporal context and
interpolation can cross boundaries; this is an offline method. No fitting occurs during prediction.

## Install and reproduce

Requirements: Python 3.14 and uv. Clone this publication repository with conversion disabled:

```powershell
git -c core.autocrlf=false clone <PUBLIC_REPOSITORY_URL> visca-barca
Set-Location visca-barca
git config core.autocrlf false
uv sync --locked
```

`.gitattributes` disables Git text conversion for all tracked files, preserving exact baseline
and model bytes on Windows. Verify all twelve checksums before inference. Obtain authorized
competition files separately; no match files, predictions or caches are distributed here.
Follow [the explicit-path, fresh-output instructions](docs/reproduction.md). Never overwrite a
validated submission. Do not run training as part of reproduction.

## Tests

These ten synthetic tests use generated arrays and no competition files:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
uv run --no-sync python -B -m pytest -q -p no:cacheprovider tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

The organizers' source checkout is required for imports, even for synthetic tests. Set
`MTAH_COMPETITION_REPO` as described in the reproduction guide. Do not run the full suite
automatically: `tests/test_match_data.py` runs real-match inference. Its historical expected hash
is not the verified preserved-artifact hash; see the guide before interpreting that test.

## Evaluation evidence

Historical README figures were spellF1 0.816/0.792 (Metrica clean/player-dropout) and
0.769/0.752 (DFL clean/player-dropout). These are unverified historical reports. Missing
selection scripts and fold records prevent independent confirmation of held-out isolation,
nested selection, metric invocation or the exact aggregates. Exact CSV reproduction proves
artifact repeatability, not possession accuracy or independent held-out performance.

All shipped models were fitted using the configured external training matches. Scoring those
matches with these checkpoints is an **in-sample diagnostic**, never independent held-out
evaluation. This package reports no new evaluation or competition performance. The
[historical experiment log](docs/experiments.md) is retained with an explicit evidence caveat.

## Provenance and privacy

Training code uses Metrica sample games 1-3 and seven Sportec Open DFL matches. External data
is not included. DFL provenance: Bassek et al. (2025), Scientific Data 12:195,
https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0 as documented by the project.
Metrica sample data: https://github.com/metrica-sports/sample-data. Its provider README requests
attribution and responsible use; this audit does not establish a blanket redistribution license.
Confirm applicable provider terms before downloading or retraining.

Never commit competition files, derived predictions, credentials, caches or private logs.
Use `main` in `HChandramania/FCB-more-than-a-hack` as the official competition submission;
other hosted branches are not the submitted implementation. This package imports no original
branches, tags or historical commits; the baseline identifier above is provenance text only.
