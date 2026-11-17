# Visca Barca — Football Possession Detection

**More Than A Hack 2027 · FC Barcelona × Metrica Sports**

An offline system that assigns a possession state to every tactical-video frame using
automatic player and ball tracking. It combines LightGBM classifiers, temporal convolutional
networks (TCNs), and structured temporal decoding.

**`main` is the official submission implementation.** Other hosted branches contain separate
development work and are not the submitted implementation.

## Evaluator guide

| To review… | Start here |
|---|---|
| Method, training sources, and limitations | [Technical report](docs/technical_report.md) |
| Installation and exact reproduction | [Quick start](#quick-start) and [detailed reproduction guide](docs/reproduction.md) |
| Evidence that the CSV can be regenerated | [Verified reproduction](#verified-reproduction) |
| Production entry point | [`scripts/predict.py`](scripts/predict.py) |
| Ensemble and decoder | [`possession/pipeline.py`](possession/pipeline.py) |
| Pretrained model integrity | [`models/SHA256SUMS`](models/SHA256SUMS) |
| Scope of the performance claims | [Evaluation evidence](#evaluation-evidence) |

## Submission at a glance

| Item | Submitted baseline |
|---|---|
| Prediction inputs | Authorized tactical ATD tracking TXT and original XML slot-team metadata |
| Output | One `H`, `A`, `D`, or `X` label per frame |
| Pretrained models | 4 LightGBM classifiers and 8 TCNs; all included |
| Prediction device | CPU; no GPU or retraining required |
| Verified inference runtime | 244.98 seconds on the audited Windows environment |
| Exact CSV reproduction | **PASS** — complete SHA-256 and binary comparisons |
| Official CSV format validation | **PASS** — exit code 0 |
| Independent held-out accuracy | **Not established by recoverable audit evidence** |

Competition data and the submission CSV are deliberately excluded from this repository.

## How the system works

```text
Tactical tracking TXT + XML team mapping
                  |
                  v
Parse coordinates and official half boundaries
                  |
                  v
Fill player gaps up to 10 s; mask positions outside halves
                  |
          +-------+--------+
          |                |
          v                v
   Player movement,   Player/team geometry,
   team layout        ball motion, touch timing
          |                |
          v                v
   2 LightGBM +       2 LightGBM +
   2 TCN models       6 TCN models
          |                |
          v                v
       P(dead)        P(home given live)
          +-------+--------+
                  |
                  v
Three-state Viterbi: home / away / dead
                  |
                  v
Two-second ownership rule; mask ungraded frames to X
                  |
                  v
              Output CSV
```

### Possession states

| Label | Meaning |
|---|---|
| `H` | Home-team possession |
| `A` | Away-team possession |
| `D` | Dead ball |
| `X` | Ungraded frame or abstention; this writer uses it outside official halves |

### Features and models

- **Features:** player speeds, team shape and advancement, ball-player distances, ball motion,
  candidate-touch timing, and centered temporal summaries. Missing observations remain NaN;
  TCNs also receive presence masks.
- **Dead-ball stage:** four models use player-based signals to reduce reliance on external
  providers' ball conventions.
- **Team-possession stage:** eight models estimate home possession conditional on live play.
  Within each clean/player-dropout group, LightGBM and the three-seed TCN mean receive equal weight.
- **Symmetry:** team-swapped training and averaged swapped predictions enforce probability-level
  team symmetry.
- **Decoding:** full-timeline Viterbi uses costs of 2.5 nats for team changes and 20 nats for
  live/dead changes. Opponent runs shorter than two seconds retain the previous live-play owner.

No fitting, calibration, or tuning occurs during prediction.

## Quick start

These commands describe an evaluator-run reproduction. They require authorized access to the
competition inputs. Use a new output directory so an existing submission is never overwritten.

### 1. Install the pinned environment

Requirements: Git, [uv](https://docs.astral.sh/uv/), and Python 3.14. The verified run used
CPython **3.14.6**. The commands below use PowerShell.

```powershell
git -c core.autocrlf=false clone --branch main --single-branch https://github.com/HChandramania/FCB-more-than-a-hack visca-barca
Set-Location visca-barca
git config core.autocrlf false
uv sync --locked --python 3.14.6
```

Keep [`.gitattributes`](.gitattributes): it disables Git text conversion for tracked files.
Windows line-ending conversion previously changed four LightGBM files and failed their checksums.

### 2. Configure the organizer source and authorized inputs

Clone the organizer source only if it is not already available beside this checkout:

```powershell
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
$env:MTAH_COMPETITION_REPO = (Resolve-Path ../more-than-a-hack-2027).Path
$env:MTAH_DATA_DIR = Read-Host 'Authorized dataset root containing tactical/'
$env:MTAH_MODEL_DIR = (Resolve-Path models).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
```

The dataset root must contain the original tactical ATD TXT and matching XML under `tactical/`.
The organizer's `games.py` and `evaluation/possession_metrics.py` are required for imports.
Video, broadcast assets, Smart Tagging, homography, and competition ground truth are not
prediction inputs. No external training dataset is needed to use the shipped models.

### 3. Verify all twelve checkpoints

This prints a status only, without displaying model contents:

```powershell
@'
from pathlib import Path
import hashlib
entries = [line.split() for line in Path('models/SHA256SUMS').read_text().splitlines() if line.strip()]
assert len(entries) == 12
assert all(hashlib.sha256((Path('models') / name).read_bytes()).hexdigest() == expected for expected, name in entries)
print('MODEL_CHECKSUM PASS')
'@ | uv run --no-sync python -B -
if ($LASTEXITCODE -ne 0) { throw 'MODEL_CHECKSUM FAIL' }
```

### 4. Generate and validate a new CSV

The original production script is unchanged. This procedure isolates both output and caches
outside the repository. Inference and validator logs stay private because the validator prints
prediction summaries.

```powershell
$outputRoot = Join-Path (Split-Path (Get-Location).Path -Parent) ('visca-reproduction-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $outputRoot -ErrorAction Stop | Out-Null
$env:MTAH_CACHE_DIR = Join-Path $outputRoot 'cache'
$env:NUMBA_CACHE_DIR = Join-Path $outputRoot 'numba-cache'
$outputCsv = Join-Path $outputRoot 'Visca Barca.csv'
if (Test-Path -LiteralPath $outputCsv) { throw 'Output already exists' }

uv run --no-sync python -B scripts/predict.py $outputCsv *> (Join-Path $outputRoot 'inference.private.log')
if ($LASTEXITCODE -ne 0) { throw 'INFERENCE FAIL' }

uv run --no-sync python -B ../more-than-a-hack-2027/evaluation/validate_submission.py $outputCsv *> (Join-Path $outputRoot 'validator.private.log')
if ($LASTEXITCODE -ne 0) { throw 'VALIDATOR FAIL' }

$expected = 'f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef'
if ((Get-FileHash -LiteralPath $outputCsv -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'EXACT_REPRODUCTION FAIL'
}
Write-Output 'EXACT_REPRODUCTION PASS'
```

If a hash differs, preserve both files and investigate locally. Do not normalize, correct, or
replace the preserved submission. See the [detailed guide](docs/reproduction.md) for binary
comparison, environment caveats, and the historical test-hash distinction.

## Verified reproduction

The prior gate ran fresh inference from the original baseline with isolated caches and all
twelve verified checkpoints. It did not retrain models or apply temporal-isolation experiments.

| Verification | Recorded result |
|---|---|
| Original versus regenerated SHA-256 | PASS |
| Binary byte-for-byte comparison | PASS |
| File size, each | 1,882,093 bytes |
| Fresh CPU inference | 244.98 seconds |
| Official format validator | PASS; exit 0 |
| Checkpoint hashes before and after | 12/12 PASS |
| Preserved submission unchanged | PASS |

**Verified CSV SHA-256**

```text
f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef
```

The verified environment was Windows x86-64, CPython 3.14.6, NumPy 2.4.6, pandas 3.0.5,
LightGBM 4.7.0, numba 0.66.0, and PyTorch 2.13.0+cu130. Installed package versions matched
`uv.lock`, and dependency consistency passed. Prediction used CPU, despite the CUDA-enabled wheel.

These are the **fresh-reproduction** environment and results. The preserved artifact's full
original creation environment is not independently documented. Byte identity on arbitrary
platforms is not established; the writer uses OS-dependent text-mode line endings.

Evidence: [reproduction verification](docs/reproduction_verification.json),
[baseline file integrity](docs/baseline_integrity.json), and
[publication preparation checks](docs/publication_verification.json).

## Evaluation evidence

**Reproduction proves repeatability. Format validation proves CSV compatibility. Neither proves
possession accuracy or independent held-out performance.**

| Evidence | Status |
|---|---|
| Exact preserved-CSV reproduction | Verified |
| Official CSV format validation | Verified |
| Metrica spellF1: 0.816 clean / 0.792 player-dropout | Unverified historical report |
| DFL spellF1: 0.769 clean / 0.752 player-dropout | Unverified historical report |
| Independently reconstructed held-out performance | Not established |

Historical selection scripts and fold records are missing. Held-out isolation, nested selection,
metric invocation, and exact aggregates therefore cannot be independently confirmed.

All shipped models were fitted using the configured external training matches. Scoring those
same matches with these checkpoints is an **in-sample diagnostic**, not independent held-out
evaluation. This package reports no new evaluation or competition accuracy results.

The [experiment log](docs/experiments.md) is retained as historical development context with an
explicit evidence caveat.

## Limitations

- **Control is inferred:** a candidate touch is a distance-based heuristic, not observed intent
  or guaranteed control. The two-second rule can suppress genuine brief recoveries.
- **Tracking quality matters:** missing observations, duplicate slots, identity changes, and
  interpolated positions can distort proximity and motion.
- **Temporal processing is offline:** centered features, future-touch timing, and TCN context
  use future information. Gap filling happens before masking, and decoding processes the full
  timeline. Half boundaries are not independently isolated in this baseline.
- **Caches need isolation:** cached matches do not validate input provenance. Use fresh caches
  for a different dataset.
- **Historical evaluation is incomplete:** the available records do not establish generalization
  or superiority over simpler baselines.

See the [technical report](docs/technical_report.md) for the complete description and caveats.

## Synthetic tests

The following ten tests use generated arrays and no competition data. Configure the organizer
source as above first; its modules are required for imports even for synthetic tests.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
uv run --no-sync python -B -m pytest -q -p no:cacheprovider tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

Do not run the full suite automatically: `tests/test_match_data.py` invokes real-match inference
and contains a different historical expected CSV hash. It was retained unchanged for provenance;
the [reproduction guide](docs/reproduction.md#historical-test-hash-limited-scope) explains its scope.

## Repository map

```text
possession/       Match loading, features, classifiers, augmentation, and decoding
scripts/
  predict.py     Production inference entry point
  train.py       External-data training; not needed to reproduce with shipped weights
  download_external.py
                 External-data downloader; not needed for pretrained inference
models/          Twelve pretrained checkpoints and SHA256SUMS
tests/           Synthetic tests and caveated historical match tests
docs/            Technical report, reproduction guide, and provenance evidence
pyproject.toml   Python requirement and direct dependencies
uv.lock          Locked dependency resolution
.gitattributes   Exact-byte checkout safeguards
```

## Training provenance and data handling

| Source | Training use | Attribution and terms |
|---|---|---|
| Metrica Sports sample data | Games 1–3; event-derived possession labels | [Provider repository](https://github.com/metrica-sports/sample-data). Its README requests attribution and responsible use; this audit does not establish blanket redistribution rights. |
| Sportec Open DFL | Seven matches; provider possession/status labels | Bassek et al. (2025), *Scientific Data* 12:195; [dataset](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0 as documented by the project. |

External data is not distributed here. Confirm applicable provider terms before downloading or
retraining. Training and external downloading are separate activities, not reproduction steps.

Never commit competition tracking, XML, video, ground truth, prediction CSVs, credentials,
caches, or private logs. Do not send competition inputs, predictions, or raw logs to AI services.

## Baseline source integrity

Verified baseline source commit:

```text
190fbdefcec376f674727a194cd8ac34315a7d70
```

Production Python, scripts, tests, pretrained checkpoints, `pyproject.toml`, and `uv.lock` are
byte-identical to that baseline. Documentation and Git checkout safeguards are the only changes.
The official `main` history was initialized separately; it does not merge original development
histories or experimental temporal-isolation code. The baseline identifier records source provenance.
