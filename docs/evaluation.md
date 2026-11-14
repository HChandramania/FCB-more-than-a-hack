# Evaluation

## Official metrics

The organisers score a `frame,label` CSV with `H`, `A`, `D` and `X`.

- **macro-F1** over `H`, `A`, `D` is the per-frame view, averaged unweighted across classes.
  It is the first tie-breaker.
- **spellF1** (mean possession-spell F1 over IoU 0.3/0.4/0.5/0.6/0.7) is the **ranking metric**.
  A possession spell is a maximal run of `H` or `A`; `D` runs are stoppages, not possessions.
- **Transition F1** at ±12 and ±25 frames, with latency, checks whether changes happen at the
  right time. A submission can have high frame accuracy and a systematic delay; these columns
  expose it.
- **Fragmentation** compares the number of predicted segments with the truth. A flickering
  submission can look good per frame while inventing many extra possessions.
- `X` is both "no ground truth" and "abstain". Ground-truth `X` is never scored; a predicted `X`
  on a graded frame counts as a miss.

The official scorer is in the organisers' repository, not in this checkout:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
cd ../more-than-a-hack-2027

python evaluation/validate_submission.py /path/to/submission.csv
python evaluation/run_evaluation.py \
  --results /path/to/folder-containing-submission.csv \
  --out /path/to/output \
  --gt public
```

`--gt public` uses the five-minute window. `--bootstrap 1000` gives the honest noise band for a
final comparison.

## Verified public-window result

The public window has 7,500 graded frames and 17 reference possessions. The shipped submission
scores:

| Metric | Value |
|---|---|
| macro-F1 | **0.9391** |
| spellF1 (mean IoU) | **0.6667** |
| predicted possessions | **10** (reference: **17**) |
| mean possession duration | 23.6 s (reference: 14.2 s) |
| fragmentation | 0.667 (14 predicted segments vs 21 truth segments) |
| transition F1 at ±25 frames | 0.606 |
| mean absolute transition latency | 6.9 frames (0.28 s) |
| abstentions on graded frames | 0 |

This is a **pipeline check**, not a leaderboard estimate. The public window is deliberately small:
17 possessions is far too few to tune on, and the scorer's own guidance puts full-match spellF1
noise at about ±0.04. The full-match (private) score is unknown.

The pattern is clear and consistent with the spell count: frame-level accuracy is strong, but the
model **merges or misses short possessions**. Its average predicted possession is much longer than
the truth's. That is the first thing to investigate with external validation, not by fitting the
public window.

## External validation

Reported in the README from nested leave-one-match-out validation (models never see the scored
match; settings chosen without it). spellF1 is the mean over the same IoU thresholds.

| Validation set | clean spellF1 | 10 % of player-frames removed |
|---|---|---|
| Metrica sample games 1–3 | 0.816 | 0.792 |
| Sportec Open DFL, 7 matches | 0.769 | 0.752 |

These figures are **reported, not reproducible from this checkout**: the external data is absent
and the experiment scripts/raw reports are not shipped. The training and prediction code is here.
See [reproducibility.md](reproducibility.md).

## What is not measured

- The private full-match score.
- Any comparison against the other teams.
- Whether the 2 s control rule is optimal; the experiment log rejects removing it (#14), but it is
  still a modelling choice.

## How to read the result

- **macro-F1 0.939** means the per-frame labels are close to the truth.
- **spellF1 0.667** means only about two-thirds of the possession-spell quality is captured once
  each possession is counted as one object.
- **10 vs 17 possessions** says the error is mostly structural: short spells disappear into
  neighbouring long spells. Reducing the minimum possession length or the decoder's switch costs
  would create more spells, but the external validation in [experiments.md](experiments.md) is the
  only trustworthy place to test that change.
