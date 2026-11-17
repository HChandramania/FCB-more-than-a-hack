# Reproduction and final packaging

These are commands for an authorized person to execute locally. They were not executed during the
documentation audit. Never send match files, predictions or raw local logs to an AI service.
Do not use `run_evaluation.py` against competition ground truth.

## Install and configure

From the solution checkout, install uv separately if needed, then:

```powershell
uv sync --locked
# Clone only if this sibling does not already exist:
# git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
$env:MTAH_COMPETITION_REPO = (Resolve-Path '../more-than-a-hack-2027').Path
$env:MTAH_DATA_DIR = (Read-Host 'Authorized dataset root containing tactical/')
$env:PYTHONDONTWRITEBYTECODE = '1'
```

The dataset must contain the original `tactical/BAR-ATH_tactical_atd.txt` and matching XML. No
download or redistribution of provided data is part of these commands. `MTAH_DATA_DIR` is read by
the organizer module at import time, so set it before starting Python. Preserve the original
`.gitattributes` rule `/models/*.txt -text`: CRLF conversion breaks LightGBM file loading.

Verify model integrity without displaying model contents:

```powershell
@'
from pathlib import Path
import hashlib
for line in Path('models/SHA256SUMS').read_text().splitlines():
    expected, name = line.split()
    if hashlib.sha256((Path('models') / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('MODEL_CHECKSUM FAIL')
print('MODEL_CHECKSUM PASS')
'@ | uv run --no-sync python -B -
if ($LASTEXITCODE -ne 0) { throw 'Model integrity failed' }
```

## Optional exact reproduction check: new output only

Exact reproduction has NOT yet been demonstrated for the audited CSV. This procedure performs an
additional full inference locally; run it only when time permits. It does not modify the original.
Use the same OS, interpreter, lockfile, model bytes and thread configuration as the original run.
Windows normally writes CRLF; Linux writes LF. Both are valid CSVs but not byte-identical files.

```powershell
$originalCsv = (Resolve-Path (Read-Host 'Existing validated CSV path')).Path
$originalHashBefore = (Get-FileHash -LiteralPath $originalCsv -Algorithm SHA256).Hash
$reproDirectory = Join-Path (Split-Path (Get-Location).Path -Parent) ('local-reproduction-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $reproDirectory -ErrorAction Stop | Out-Null
$env:MTAH_CACHE_DIR = Join-Path $reproDirectory 'cache'
$env:NUMBA_CACHE_DIR = Join-Path $reproDirectory 'numba-cache'
$reproCsv = Join-Path $reproDirectory 'submission.csv'
if (Test-Path -LiteralPath $reproCsv) { throw 'Refusing to overwrite output' }
uv run --no-sync python -B scripts/predict.py $reproCsv *> (Join-Path $reproDirectory 'inference.local.log')
if ($LASTEXITCODE -ne 0) { throw 'INFERENCE FAIL; inspect log locally only' }
uv run --no-sync python -B ../more-than-a-hack-2027/evaluation/validate_submission.py $reproCsv *> (Join-Path $reproDirectory 'validation.local.log')
if ($LASTEXITCODE -ne 0) { throw 'VALIDATION FAIL; inspect log locally only' }
$originalHashAfter = (Get-FileHash -LiteralPath $originalCsv -Algorithm SHA256).Hash
$reproHash = (Get-FileHash -LiteralPath $reproCsv -Algorithm SHA256).Hash
if ($originalHashBefore -ne $originalHashAfter) { throw 'ORIGINAL_INTEGRITY FAIL' }
if ($originalHashBefore -ne $reproHash) { throw 'EXACT_REPRODUCTION FAIL' }
Write-Output 'ORIGINAL_INTEGRITY PASS'
Write-Output 'EXACT_REPRODUCTION PASS'
```

Do not normalize, overwrite or silently replace the original if hashes differ. Diagnose only with
local deterministic checks and sanitized results. Match caches are not provenance-checked; the
fresh cache above prevents reuse of another dataset's arrays. Keep local logs private.

The historical `SUBMISSION_SHA256` in `tests/test_match_data.py:15` is an additional historical
reference, not proof that the current file matches it. The match tests were not run in the final
documentation audit. Their skip marker checks the default data directory rather than the
`MTAH_DATA_DIR` override.

## Lightweight tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
uv run --no-sync python -B -m pytest -q -p no:cacheprovider tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

These select ten tests and do not invoke full-match inference. Do not run the full suite
automatically: `tests/test_match_data.py` performs expensive match inference.

## Final delivery

Preserve the validated working CSV. Its current filename is `submission.csv`, not the required
registered-team name, **Visca Barca**. Manually prepare the upload filename `Visca Barca.csv` without
changing its bytes; this audit does not copy or rename it. Recheck the exact upload artifact locally.
Do not place predictions in the public repository.

Markdown is accepted for the technical document by `SUBMISSION.md`, despite its filename example
using PDF. Deliver `docs/technical_report.md` as `Visca Barca.md`, or export it to `Visca Barca.pdf`
locally if desired. Do not submit Markdown bytes under a `.pdf` extension.

Publish the source, lockfile, all twelve models, checksum manifest, documentation and
`.gitattributes` only after a final tracked-file/history/secret review. Public access to the remote
repository and publication of the latest documentation are not established by the local Git remote.
Commit and push are separate user actions; this audit does neither.

The form link is supplied by organizer email. Use the registered team name, upload the CSV and
report, and supply the public repository URL. A Google account is required; one team response may
be edited. The documented deadline is 17 November 2026; a precise cutoff time/timezone is not
specified in the inspected submission instructions, so consult the registration email/form.
