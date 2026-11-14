# Architecture

## In one paragraph

The competition tracking is parsed into a `Match`: per-frame positions in metres plus a team
slot per player. Player gaps of at most 10 s are interpolated, then two families of per-frame
classifiers estimate `P(dead)` and `P(home | live)`: LightGBM on fixed multi-scale windows and a
temporal convolutional network on the raw signal sequence. Four classifiers vote on dead and eight
on home. A 3-state Viterbi decoder turns the probabilities into runs of home, away and dead, and
the competition control rule folds short runs of the other team back into the previous possession.
`scripts/predict.py` masks everything outside the two halves to `X` and writes the CSV.

## Prediction path

```
data/tactical/*.txt + *.xml
        │
        ▼
data.load_competition_match()          Match(players_m, ball_m, slot_team, pitch_m, halves)
        │
        ▼
augment.fill_player_gaps(match, 10 s)   interior gaps only, one slot = one player
        │
        ▼
data.only_halves(match)                warm-up/halftime positions removed
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
gbm.dead_features / team_features              tcn.step_inputs
multi-scale windows at 25 fps                  25 fps → 5 steps/s
        │                                             │
        ▼                                             ▼
LightGBM: dead_team_sym, team_ball_advance      TCN: tcn_dead, tcn_team_change
(+ _pd variants)                                (+ _pd variants, 3 team seeds)
        │                                             │
        └──────────────┬──────────────────────────────┘
                       ▼
pipeline.dead_probability / home_probability
                       ▼
decoding.possession_states(P(dead), P(home), 2.5, 20 nats)
                       ▼
data.apply_control_rule(labels, 2 s)
                       ▼
scripts/predict.py → frame,label (H/A/D/X)
```

The whole path is CPU-only and takes about two minutes for 199,319 frames.

## Training path

```
external/metrica/* + external/dfl/*
        │
        ▼
external.read_metrica / read_dfl          per-frame H/A/D/X labels
        │
        ▼
data._external_match                      control rule, half separator, metres
        │
        ├── augment.add_live_ball_gaps    copies with 0/15/30 % ball loss
        ├── augment.add_player_dropout    copies with 10 % player loss
        ▼
features.player_signals / team_signals / touch_signals / ball_advance_signals
        │
        ├── gbm.training_rows → LightGBM.fit
        └── tcn.training_copies → TCN.train (GPU if available)
```

Every training row comes from an external match. The competition match and the public ground
truth never enter training or model selection.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `possession/competition.py` | BAR-ATH constants, tracking-to-video offsets, asset paths. Vendored from the organisers' `games.py` at commit `f8f2629…`. |
| `possession/paths.py` | `data/`, `external/`, `cache/`, `models/` roots and their environment overrides. |
| `possession/data.py` | `Match` container, competition/external loaders, half joining, control rule, graded-frame mask, atomic `.npz` cache. |
| `possession/external.py` | DFL XML and Metrica CSV/EPTS/JSON readers; derives Metrica possession labels from events. |
| `possession/augment.py` | Simulated ball loss, simulated player dropout, linear interpolation of short player gaps. |
| `possession/features.py` | Player, team, touch and ball-advance signals; multi-scale window means; team swap. |
| `possession/gbm.py` | Four LightGBM classifiers, training-row construction, symmetric inference. |
| `possession/tcn.py` | Eight TCN classifiers (dilated residual 1-D convolutions), deterministic training, symmetric inference. |
| `possession/decoding.py` | Numba Viterbi over 2 or 3 states with switch costs. |
| `possession/pipeline.py` | The submitted ensembles, decoder settings and control rule. |
| `scripts/download_external.py` | Downloads and checksums the external data. |
| `scripts/train.py` | Trains/saves every classifier kind. |
| `scripts/predict.py` | Loads the match, predicts, writes the submission. |

## Shapes and invariants

| Object | Shape | Meaning |
|---|---|---|
| `players_m` | `(frames, slots, 2)` | Metres from the pitch corner; `NaN` where a player is not tracked. |
| `ball_m` | `(frames, 2)` | Metres; `NaN` where the ball is not tracked. |
| `slot_team` | `(slots,)` | `H` or `A` per tracking slot. |
| `labels` | `(frames,)` | `H/A/D/X` codes; all `X` on the competition match. |
| `halves` | `((start, stop), …)` | Half-open frame ranges; sides change between them. |

Invariants the tests protect:

- **Team symmetry.** Exchanging `H` and `A` mirrors `P(home)` and leaves `P(dead)` unchanged.
  Both classifier families train on swapped copies and average over the swap at inference.
- **No label leakage.** The competition match's `labels` are all `X`; only external labels are
  used to fit anything.
- **No cross-half windows.** External halves are separated by 1,000 `NaN` frames, so no smoothing
  window or TCN crop can span halftime.
- **One slot is one player.** Gaps are filled only inside a slot and only up to 10 s; a slot that
  reappears after a gap has moved at a median 0.2–1.5 m/s on the competition match.

## Runtime layout

| Folder | Written by | Needed for |
|---|---|---|
| `cache/matches/` | `data._cached` | Faster repeat loading. |
| `cache/rows/` | `gbm._cached_rows` | Faster retraining. |
| `cache/steps/` | `tcn._prepared` | Faster retraining. |
| `models/` | `gbm.save`, `tcn.save` | Prediction and tests. Shipped. |

Deleting `cache/` only costs reconstruction time. Deleting `models/` requires retraining; the
shipped files are listed in `models/SHA256SUMS`.
