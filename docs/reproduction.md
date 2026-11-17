# Exact baseline reproduction

## Verified evidence and environment

Source baseline: `190fbdefcec376f674727a194cd8ac34315a7d70`. The successful gate used the
original production predictor, a fresh cache, the original authorized inputs, and all twelve
manifest-verified checkpoints. No retraining, tuning or temporal-isolation patches were used.

| Check | Verified result |
|---|---|
| Original and regenerated SHA-256 | `f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef` |
| Full SHA-256 comparison | PASS |
| Binary byte comparison | PASS |
| File size, each | 1,882,093 bytes |
| CPU inference runtime | 244.98 seconds |
| Official validator | PASS, exit 0 |
| Checkpoints before and after | 12/12 match `models/SHA256SUMS` |
| Preserved artifact integrity | PASS, unchanged |

Verified host environment: Windows x86-64, CPython 3.14.6; LightGBM 4.7.0, numba 0.66.0,
NumPy 2.4.6, pandas 3.0.5, PyTorch 2.13.0+cu130. All installed package versions matched the
lockfile and `uv pip check` passed. The full package list is in
`reproduction_verification.json`. The lockfile SHA-256 is
`428913284cc6c90f0ccceed7d78cef7d350391ff1c797e5d1762311384549370`.
The original preserved artifact's full creation environment is not independently known.

The audit introduced no thread overrides. LightGBM training has `random_state=0, n_jobs=8`;
TCN training uses seeds 0/1/2, deterministic algorithms, disabled cuDNN benchmarking and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Prediction loads saved states on CPU, uses evaluation mode
and disables gradients. No seed or inference configuration was changed for this package.

## Windows-safe checkout and setup

```powershell
git -c core.autocrlf=false clone <PUBLIC_REPOSITORY_URL> visca-barca
Set-Location visca-barca
git config core.autocrlf false
uv sync --locked
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
$env:MTAH_COMPETITION_REPO = (Resolve-Path ../more-than-a-hack-2027).Path
$env:MTAH_DATA_DIR = Read-Host 'Authorized dataset root containing tactical/'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Reuse an existing organizer checkout instead of cloning again. The baseline requires the
organizer's `games.py` and `evaluation/possession_metrics.py`; it does not vendor them.
The successful gate used the locally installed organizer checkout, whose precise source commit
was not captured by that gate. Do not claim a fully pinned organizer source from this evidence.
Only source modules are needed for synthetic tests. Prediction additionally needs the original
tactical ATD TXT and matching XML. No competition data is supplied or downloaded here.

Keep `.gitattributes`: `* -text` preserves all tracked bytes. A normal Windows checkout of the
old repository converted four LightGBM files to CRLF and failed their hashes. This package
contains the exact committed model bytes. Never normalize or resave checkpoints.

Verify model checksums without displaying their contents:

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

## Authorized local inference: fresh output only

The following is a reproduction procedure, not an instruction to rerun during publication.
It uses the unchanged baseline script. Keep original artifacts outside this repository.

```powershell
$outputRoot = Join-Path (Split-Path (Get-Location).Path -Parent) ('visca-reproduction-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $outputRoot -ErrorAction Stop | Out-Null
$env:MTAH_CACHE_DIR = Join-Path $outputRoot 'cache'
$env:NUMBA_CACHE_DIR = Join-Path $outputRoot 'numba-cache'
$env:MTAH_MODEL_DIR = (Resolve-Path models).Path
$outputCsv = Join-Path $outputRoot 'regenerated.csv'
if (Test-Path -LiteralPath $outputCsv) { throw 'Output already exists' }
uv run --no-sync python -B scripts/predict.py $outputCsv *> (Join-Path $outputRoot 'inference.private.log')
if ($LASTEXITCODE -ne 0) { throw 'INFERENCE FAIL' }
uv run --no-sync python -B ../more-than-a-hack-2027/evaluation/validate_submission.py $outputCsv *> (Join-Path $outputRoot 'validator.private.log')
if ($LASTEXITCODE -ne 0) { throw 'VALIDATOR FAIL' }
$expected = 'f6f402152dfe300986243f5af15cae06ca0fd66f3239fbf02a9787574edfeaef'
if ((Get-FileHash -LiteralPath $outputCsv -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'EXACT_REPRODUCTION FAIL' }
Write-Output 'EXACT_REPRODUCTION PASS'
```

The official validator prints prediction summaries; keep its entire output private. Do not
send CSV contents or raw logs to AI services. For an additional binary comparison, execute
`fc.exe /b` locally with both streams redirected to a private file and inspect only its exit
code (0 equal, 1 different, 2 error). Do not modify either CSV on a mismatch. Different operating
systems can serialize line endings differently; hash equality is asserted only for the audited
artifact/environment, not for every platform. This predictor opens output in text mode.

## Historical test hash: limited scope

`tests/test_match_data.py` is preserved byte-identically and contains historical hash
`2467a1396f08f2e0f4f65bb8ae261da82f44cae9678b8547dcbe4d2b17ac6f8e`.
That value belongs to the original development reference; its exact serialization/environment
was not established by the current gate. It is not the verified preserved Visca Barca hash.
No equivalence between these two historical artifacts has been established. The historical
test may fail despite successful reproduction of the preserved submission. It was not run for
publication preparation, and its constant has not been changed to manufacture a passing test.
The skip marker checks the organizer's default data directory, not `MTAH_DATA_DIR`.

Run only the selected synthetic tests during publication preparation:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
uv run --no-sync python -B -m pytest -q -p no:cacheprovider tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

Caches do not validate input provenance; always isolate them for another dataset. Training and
external downloading are separate activities and are not required to reproduce with shipped
weights. The public package contains no competition files, predictions or trained fold models.
