# Paired external decoding validation

## Current authorization and dataset status

Metrica downloads remain blocked at the user's request. DFL-only restoration and one diagnostic
were subsequently authorized; see [DFL-only preparation](dfl_diagnostic.md). No fitting is authorized.
The existing competition submission
must remain untouched. Neither the harness nor its tests calls `load_competition_match`, reads a
competition CSV, invokes the competition prediction CLI, or reads competition ground truth.

DFL: Bassek et al., [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y)
states CC BY 4.0 dataset availability. Attribute the authors/provider and dataset, link the license
and identify transformations. The project downloader pins seven matches and their file hashes.

Metrica: the [pinned provider README](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/README.md)
requests attribution and responsible use, but an explicit license file at that pinned root was not
available. Public accessibility is not a substitute for permission. The user chose to keep downloads
blocked pending resolution. `--metrica-permission-confirmed` records an applicable permission
attestation; it does not itself establish or grant rights. Do not pass it without actual permission.

The unchanged `scripts/download_external.py` downloads both providers, so do not run it while
Metrica permission is unresolved. It does not offer a DFL-only flag. The harness never downloads.
Default all-dataset execution requires all 23 files in `scripts/external_sha256.txt`, hash-verified
before processing. DFL-only diagnostics instead require just the selected DFL match pairs.

## Modes and protected locations

`scripts/evaluate_external.py` requires an explicit mode. `--dataset dfl` now supports DFL-only
DIAGNOSTIC execution with only the selected DFL file pairs, no Metrica permission flag or file access.
DFL-only OUT-OF-FOLD is rejected because the unchanged trainers use the full external corpus.
The default `--dataset all` retains the following modes:

- `DIAGNOSTIC`: shipped checkpoints; every result is labeled
  `in_sample_diagnostic_not_generalization`. No fitting permitted.
- `OUT-OF-FOLD`: requires `--allow-training`; all four LightGBM and eight TCN models are trained
  with the evaluated match excluded. The existing TCN training function calculates normalization
  after filtering that match. No fallback to shipped checkpoints exists. Every result is labeled
  `out_of_fold_fixed_configuration_not_nested_selection`.

The default corpus is the existing three Metrica and seven DFL matches. `--matches` can restrict
evaluation to named configured external matches, but does not change the full training corpus.
All provider permissions/files are still required. Do not pass new or competition match names.

External data and workspace must be dedicated, separate directories outside both repositories.
Paths resolve symlinks before checking. Artifacts, parsed external matches, feature rows, checkpoints,
probabilities and JSON reports are written only under the workspace. Shipped model files are never
overwritten. Run one process per workspace. Incomplete model-fold directories fail explicitly;
use a fresh workspace rather than silently mixing partial and shipped models.

## Controlled comparison

For each match the harness generates or reloads one P(dead)/P(home|live) pair. Both paths receive
the same immutable arrays:

1. Original: decode the full timeline, apply the original control rule globally, then mask outside
   declared halves to X (the original submission writer's order).
2. Patched: call the existing `_decode_halves`, decoding and controlling each span independently.

No features, interpolation, TCN context, member weights, switch costs, or control thresholds change.
Both paths call the official `possession_metrics.evaluate(reference, labels, bootstrap_n=0)`.
Reports contain per-match spellF1/macroF1, patched-minus-original differences, whether valid-half
labels differ, and changed-frame counts. No label or probability sequences are printed.

Reference processing remains exactly the external loader's behavior: DFL provider flags or
Metrica event-derived labels, followed by the two-second control rule. Halves are separated by
1,000 NaN/X frames. Reference frames outside declared halves must be X. The official scorer uses
reference X gaps to separate temporal runs. This is evaluation under processed label conventions,
not independent verification that the two-second rule captures football possession semantics.

Unweighted match means are reported per dataset. A separate summary averages dataset means
equally. Do not interpret that second summary as a pooled-frame metric. Per-match differences
support paired comparisons; no performance or significance claims have been made yet.

## Reproducibility and leakage controls

Each result stores model SHA-256 values, explicit mode/held-out identifier/training match list,
external-file hashes, reference-array hash, half spans, source hashes, package/platform versions,
training device, thread settings, seeds/augmentation seed policy, fixed settings and scorer-file hash.
Source-derived namespaces separate parsed/training caches after data or code changes. Learned
normalization resides in fold-specific checkpoints. Probability caches also validate their arrays'
hashes, shapes, finiteness and ranges and never regenerate a corrupt hit silently.

The historical development process already used these external datasets. Even correct OOF fitting
does not erase historical model-selection bias. This experiment evaluates fixed current settings
and a paired decoder change; it is not a fresh independent test or reconstruction of missing nested
selection. The known feature-context and interpolation isolation issues remain unchanged in both
arms. These limitations must accompany reported scores.

## Commands after separate authorization

Do not execute these yet. First resolve provider permission, separately authorize restoration of
the checksum-pinned external files, and confirm all data is outside both repositories. No provided
competition assets are needed. From the solution directory, once those prerequisites are met:

```powershell
# In-sample diagnostics only; explicit permission flag is a factual attestation, not a license.
.venv/Scripts/python.exe -B scripts/evaluate_external.py --mode DIAGNOSTIC --external-dir ../external-evaluation-data --workspace ../external-decoding-diagnostic --metrica-permission-confirmed
```

```powershell
# Expensive: 10 outer folds x 12 models. Does not overwrite shipped checkpoints.
.venv/Scripts/python.exe -B scripts/evaluate_external.py --mode OUT-OF-FOLD --external-dir ../external-evaluation-data --workspace ../external-decoding-oof --device cpu --allow-training --metrica-permission-confirmed
```

Use `--device cuda` only for explicitly approved GPU training with a compatible local runtime.
Prediction remains the existing CPU path. Start with `--matches Metrica-1` as an explicitly labeled
pilot if desired; a one-match report is not a ten-match dataset result. Reuse the same mode/device/
sources/workspace for cache reuse. The CLI suppresses third-party logs and emits only aggregate
JSON or sanitized error categories. It writes no match-level prediction CSVs.

## Synthetic verification command

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:NUMBA_DISABLE_JIT = '1'
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_external_evaluation.py tests/test_half_decoding.py tests/test_synthetic_temporal_boundaries.py tests/test_control_rule.py tests/test_decoding.py tests/test_fill_gaps.py
```

Tests use generated arrays and temporary synthetic cache/model files, never real external or
competition records. Training calls are mocked to verify exclusions, provenance and cache behavior.
Real-data parsing, full training, numeric performance and end-to-end timing remain unverified.
