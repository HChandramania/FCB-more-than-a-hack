# External evaluation evidence and executable plan

## What the checkout establishes

`README.md` originally reported Metrica spellF1 0.816 and DFL 0.769, plus 0.792 and 0.752 with
10% player-frame removal. These exact four rounded figures occur in the README, not in a per-match
result table. They have not been independently reproduced. Final macroF1 is not measured in the
available records; isolated development comments about class F1 are not final macroF1 results.

`docs/experiments.md:8-10` describes mean per-match spellF1 and equal weighting of the two datasets
for selection. Rows #37 and #41 describe the final ensemble and gap filling; #38 reports 0.8176 /
0.7700 for the preceding v15 robustness reference. Those numbers are not the final README values.
No valid calculation of 0.816 / 0.769 can be reconstructed from the available deltas alone.

The log claims nested leave-one-match-out selection and contains acknowledgments of development
selection biases and subsequent corrections. Its named `scripts/experiments/selection.py`,
`e08_joint_decoding.py`, `possession/validate.py`, per-fold probability records and archived reports
are absent. Actual held-out isolation, inner-fold selection and historical official-metric calls
therefore remain unverified. Production `gbm.fit(excluded=...)` and `tcn.train(excluded=...)` support
match exclusion but the final `scripts/train.py` fits on all external matches, not an evaluation fold.

The official metric implementation is `evaluation/possession_metrics.py:evaluate` in the organizer
checkout. It returns `spell_f1_mean` and `macro_f1`, uses X-defined valid runs, and averages spell F1
across IoU thresholds 0.3/0.4/0.5/0.6/0.7. Its availability does not prove the missing experiments
called it. The historical aggregation is described as unweighted match means within each dataset;
equal dataset weighting applies to selection. Fold records are needed to confirm this calculation.

## Labels and boundaries: inspected behavior, not empirical validation

`external.py:88-91` maps DFL provider status/possession flags to D/H/A, leaving unknowns as X.
`external.py:129-171` approximates Metrica possession from on-ball events and stoppage/restart
intervals, separately per period. Event-derived timing is not human-verified possession truth.
`data.py:63-90,174-177` joins external halves with 1,000 NaN/X frames and applies the two-second
control rule. The scorer's X-defined runs separate halves. Feature windows are bounded by the
separator, although the predictor's decoder is still called across the combined timeline.
No end-to-end external boundary regression or provider-event timing reproduction was executed.

External data is absent locally. The configured set is Metrica-1/2/3 plus DFL-J03WMX/J03WN1/J03WOH/
J03WOY/J03WPY/J03WQQ/J03WR9. Source URLs and hashes are in the tracked downloader and manifest.

## Optional new fixed-configuration leave-one-match-out evaluation

This is a new evaluation, NOT reproduction of the unavailable historical nested selection. It
trains 120 fold-specific models and may take hours. Do not run for the deadline unless necessary.
Do not use pretrained all-external-match models to score these same external matches as held out.
No part of the following plan loads the competition match, predictions or ground truth.

After reviewing provider terms, the user may explicitly download only the permitted external data:

```powershell
$env:MTAH_EXTERNAL_DIR = Join-Path (Split-Path (Get-Location).Path -Parent) 'external-evaluation-data'
$env:MTAH_CACHE_DIR = Join-Path (Split-Path (Get-Location).Path -Parent) ('external-evaluation-cache-' + [guid]::NewGuid().ToString('N'))
uv run --no-sync python -B scripts/download_external.py
if ($LASTEXITCODE -ne 0) { throw 'External download failed' }
```

Save this Python block as a separate local script and execute it from the solution directory with
`uv run --no-sync python -B <script_path>`. It uses fixed current settings, excludes the scored match
from every fit and normalization, keeps fold models in memory, and never overwrites shipped models.

```python
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from possession import data, gbm, tcn, pipeline

records = []
original_gbm_load, original_tcn_load = gbm.load, tcn.load
try:
    for dataset, name, load in data.external_matches():
        excluded = (name,)
        tree_models = {kind: gbm.fit(kind, excluded=excluded).booster_
                       for kind in gbm.KINDS}
        checkpoints = {kind: tcn.train(kind, excluded=excluded, device="cpu",
                                       log=lambda _: None) for kind in tcn.KINDS}
        gbm.load = lambda kind: tree_models[kind]
        tcn.load = lambda kind: checkpoints[kind]
        match = load()
        prediction = pipeline.predict(match)
        score = data.metrics.evaluate(match.labels, prediction, bootstrap_n=0)
        record = dict(dataset=dataset, match=name,
                      spellF1=float(score["spell_f1_mean"]),
                      macroF1=float(score["macro_f1"]))
        records.append(record)
        print(json.dumps(record), flush=True)
finally:
    gbm.load, tcn.load = original_gbm_load, original_tcn_load

means = {}
for dataset in sorted({r["dataset"] for r in records}):
    rows = [r for r in records if r["dataset"] == dataset]
    means[dataset] = {metric: float(np.mean([r[metric] for r in rows]))
                      for metric in ("spellF1", "macroF1")}
print(json.dumps({"per_dataset_match_means": means,
                  "equal_dataset_means": {
                      metric: float(np.mean([v[metric] for v in means.values()]))
                      for metric in ("spellF1", "macroF1")}}))
```

Record source/organizer commit IDs, environment, external-file hashes, fold exclusions, per-match
scores and aggregation. To reproduce historical claims, recover the original nested-selection
scripts and fold records first; repeat inner-fold model fitting without either outer or inner held-out
matches, select settings using only inner scores, then score the untouched outer match. Do not
retune settings from the new outer-fold results or describe fixed-setting LOMO as nested evaluation.
