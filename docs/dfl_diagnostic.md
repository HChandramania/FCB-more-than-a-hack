# DFL-only diagnostic: Visca Barca

## License and exact file coverage

The official [Figshare article API](https://api.figshare.com/v2/articles/28196177) returned dataset
DOI `10.6084/m9.figshare.28196177.v1`, version 1, CC BY with license URL
<https://creativecommons.org/licenses/by/4.0/>, and no embargo. All 14 exact DFL filenames in
`scripts/external_sha256.txt` appear in that article's file list. This establishes article-level
license coverage of the match-information and raw-observed-position files consumed by the loader;
it is not merely the journal article's copyright license.

Authors: Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert. Title: *An integrated
dataset of spatiotemporal and event data in elite soccer*. Dataset:
<https://doi.org/10.6084/m9.figshare.28196177.v1>. Paper:
<https://doi.org/10.1038/s41597-025-04505-y>. Data provenance: official DFL/Sportec match data.

CC BY 4.0 permits reuse, adaptation and redistribution, including evaluation, with appropriate
credit, license linkage, indication of modifications and no endorsement claim or added legal/technical
restrictions. No extra access, embargo or redistribution conditions were identified in the article
metadata. Rights outside the copyright license are not expanded by CC BY.

The DFL-only downloader writes `DFL_ATTRIBUTION.md` and `DFL_PROVENANCE.json` beside the external
data, outside Git. They retain author/title/DOI/license details, exact source URLs, Figshare file IDs,
sizes, match identifiers and verified SHA-256 values. Downloaded source files are unchanged;
coordinate conversion, derived labels, control-rule processing and model scoring are identified as
evaluation transformations. This document provides the corresponding public acknowledgment.

## Scope and commands

Only `DFL-J03WMX`, the first configured DFL match, is authorized for the initial benchmark. Its pair
is two of the seven-match corpus's fourteen files; the other matches need no download for this
benchmark. Metrica permission remains unresolved. Do not run the combined external downloader.

```powershell
.venv/Scripts/python.exe -B scripts/download_dfl.py --destination ../external-dfl-data --matches J03WMX
.venv/Scripts/python.exe -B scripts/evaluate_external.py --mode DIAGNOSTIC --dataset dfl --matches DFL-J03WMX --external-dir ../external-dfl-data --workspace ../external-dfl-diagnostic-J03WMX
```

No `--metrica-permission-confirmed`, training flag or model fit is involved. The harness checks only
the selected DFL pair, loads only its external-match loader, and uses the shipped models. Metadata
still records that those models were trained on the full historical external corpus. Scores must be
labeled **IN-SAMPLE DIAGNOSTIC, NOT GENERALIZATION**. All probability/reference sequences and
external cache files remain local, outside both repositories. No competition file is needed.

Both decoder paths use one identical cached probability pair and the official scorer with
`bootstrap_n=0`. The original path decodes/controls globally, then masks; the patched path resets
per half. Fresh probability caches are reloaded immediately to verify metadata, array hashes,
shapes and values. Only aggregate metrics, differences, changed-frame counts and sanitized errors
are printed. The CLI reports wall time, process CPU time and peak working-set memory on Windows.

Stop after this one match. More matches, new data restoration, evaluation or training require
separate approval. No competition submission is regenerated or replaced.

## Executed one-match result — 17 November 2026

**IN-SAMPLE DIAGNOSTIC, NOT GENERALIZATION.** Exactly one match, DFL-J03WMX, was evaluated with
the shipped checkpoints. No training, Metrica access or competition-file access occurred.

| Path | spellF1 | macroF1 |
|---|---:|---:|
| Original global decoding/control | 0.8973730297723291 | 0.9688502123535114 |
| Patched half-independent decoding/control | 0.8973730297723291 | 0.9688502123535114 |
| Patched minus original | 0 | 0 |

Changed valid-half frames: 0. Changed scored frames: 0. Probability-cache integrity: PASS, including
reload/hash/shape/range verification of the newly generated shared probabilities. Official scorer:
`bootstrap_n=0`; references use the existing external loader and its possession-control convention.

Harness wall time: 99.55 s (match processing: 98.89 s); process CPU time: 215.55 s; peak process
working set: 1,405.27 MiB. Harness timing excludes interpreter/module-import startup. Exit code: 0.
Two source files were restored and verified, totaling 418,536,763 bytes. The remaining twelve files
for six matches were not downloaded. This single result does not establish behavior on other matches
or prove/disprove the value of the boundary fix.

Artifacts are outside both repositories: `../external-dfl-data` contains source files, attribution
and provenance; `../external-dfl-diagnostic-J03WMX` contains source-keyed caches and aggregate JSON
results. No prediction sequences were displayed. Stop pending approval for any further evaluation.
