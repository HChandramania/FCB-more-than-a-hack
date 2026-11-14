# Model

The submitted model is `possession/pipeline.py` (v16). Every setting below was chosen by nested
leave-one-match-out validation on the external matches; the full record is
[experiments.md](experiments.md).

## Signals

All signals are per frame at 25 fps and are designed to mean the same thing under either team,
either provider and either half.

| Group | Count | Examples |
|---|---|---|
| Player movement and layout | 9 | mean/p90 speed, share walking or standing, spread in x/y, players outside the pitch, players in the centre circle, nearest corner distance. |
| Per-team shape (with ball) | 18 | progress and advance in the team's attacking direction, spread, opponent-half share, second-deepest defender, openness, speed, nearest-player ball distance, ball progress. |
| Ball-global | 2 | ball speed, ball seen. |
| Touch timing | 7 | seconds since / until a touch per team (touch = nearest player within 1 m of a seen ball), touch flag, ball acceleration. |
| Ball advance | 2 | ball velocity along each team's attacking direction. |

Team signals are measured **in each team's attacking direction**, estimated per half from which
goal the team defends. Names end in `_H`/`_A`; `features.swap_teams` exchanges the pair.

### Multi-scale windows

`features.multi_scale` applies centred means at **0.5, 1.5, 4.5, 13.5 and 40 s** and past-minus-
future trends at **3 and 9 s** (after Lang et al. 2022, *Scientific Reports* 12:19948). A window
that contains no observation is `NaN`; LightGBM handles it natively.

| Feature set | Base signals | Multi-scale features |
|---|---|---|
| Dead (`gbm.dead_features`) | 9 player + 14 per-team without ball = 23 | 7 × 23 = **161** |
| Team (`gbm.team_features`) | 20 team signals × 7 = 140, + 7 touch raw, + 49 touch scaled, + 14 ball advance | **210** |

## The 12 classifiers

### LightGBM (4 files)

Trained on every 5th frame because neighbouring frames are nearly identical after window means.
Metrica and DFL get equal total sample weight, so the 7 DFL matches do not dominate the 3 Metrica
matches.

| Kind | Target | Training copies |
|---|---|---|
| `dead_team_sym` | `P(dead)` | clean |
| `dead_team_sym_pd` | `P(dead)` | + 10 % player dropout |
| `team_ball_advance` | `P(home \| live)` | ball loss 0 % and 30 % |
| `team_ball_advance_pd` | `P(home \| live)` | the same + 10 % player dropout |

Fixed parameters: 300 trees, learning rate 0.05, 31 leaves, minimum 200 samples per leaf,
subsample 0.8, column sample 0.8, L2 1.0, `random_state=0`.

Each model is trained on every row plus its team-exchanged copy. At inference,
`gbm.symmetric_probability` averages the direct and exchanged predictions; `P(home)` is mirrored
(`1 - p`), `P(dead)` is not.

### TCN (8 files)

A stack of eight dilated residual 1-D convolution blocks, 64 channels, dilations 1–128, dropout
0.1. The receptive field is 511 steps ≈ **102 s** at 5 steps/s. Inputs are standardised with
presence masks; `_H`/`_A` pairs share one mean and standard deviation so the swapped input is the
normalisation of the swapped match.

| Kind | Target | Members |
|---|---|---|
| `tcn_dead`, `tcn_dead_pd` | `P(dead)` | 1 + 1 |
| `tcn_team_change`, `_s1`, `_s2` | `P(home \| live)` | seeds 0, 1, 2 |
| `tcn_team_change_pd`, `_s1_pd`, `_s2_pd` | `P(home \| live)` + player dropout | seeds 0, 1, 2 |

Training: 1,500 steps, batch 16, crops of 1,024 steps, Adam at 1e-3, BCE with per-step weights.
Team kinds see ball-loss copies at 0 % (live placement), 15 % and 30 % (uniform placement). Steps
within 2 s of a true team change weigh **3×**. Training is deterministic
(`torch.use_deterministic_algorithms`, fixed cuBLAS workspace).

## Ensemble

```
P(dead) = mean( mean(dead_team_sym,     tcn_dead),
                mean(dead_team_sym_pd,  tcn_dead_pd) )

P(home) = mean( pair(""), pair("_pd") )
pair(s) = mean( team_ball_advance + s,
                mean(tcn_team_change + s + seed, seed ∈ {"", "_s1", "_s2"}) )
```

Four classifiers contribute to dead and eight to home. The outer clean/dropout average is the v15
compromise: almost no loss on clean tracks and most of the robustness gain at 10 % player dropout,
which is the relevant condition for this tracking (≈ 7.6 % missing player-frames).

## Decoding

`decoding.possession_states` builds a 3-state Viterbi problem:

| State | Frame probability |
|---|---|
| home | `(1 − P(dead)) · P(home)` |
| away | `(1 − P(dead)) · (1 − P(home))` |
| dead | `P(dead)` |

Switch costs are **2.5 nats** between home and away and **20 nats** between live and dead, in both
directions. The cost is added once per state change; per-frame emission cost is `−log p`.

The decoder is deliberately stronger than per-frame argmax: a classifier that is right on 94 % of
frames can still produce many short dead runs, and every short run would split a possession.

## Control rule

`data.apply_control_rule` implements the challenge convention of **controlled possession**:

- inside each live stretch the restarting team owns the ball;
- a run of the other team shorter than **2 s** is treated as a touch or a contest and folded back
  into the owner;
- a run of at least 2 s transfers possession;
- `D` and `X` are never rewritten.

The 2 s value is the smallest reading of "a few seconds" in `CHALLENGE.md`. It is a modelling
assumption, not a rule fixed by the organisers; the experiment log shows that removing it costs
score (#14).

## Missing data

| Stage | Handling |
|---|---|
| Training, ball | Copies with 2–30 s log-uniform ball gaps: GBM 0/30 %, TCN 0/15/30 %. |
| Training, players | Copies with 10 % of player-frames removed in 2–30 s blocks per player (`_pd` kinds). |
| Inference, players | Interior gaps inside one slot of at most **10 s** are linearly interpolated. Leading/trailing gaps stay empty. |
| Inference, signals | GBM accepts `NaN`; TCN replaces missing values with 0 and appends a presence mask. |

The gap fill only applies inside a slot. On the competition match, a slot that reappears after a
gap has moved at a median 0.2–1.5 m/s, so one slot is one player.

## Why the dead classifier has no ball

The external providers encode dead time in the ball itself: DFL freezes the ball on every dead
frame, and the Metrica sample removes it on most dead frames. The competition tracking does not
follow either convention. A ball-aware dead classifier therefore learned the provider convention
and scored well in validation for the wrong reason (#11–#12 in the experiment log). The submitted
dead models use player movement and team layout only. Team classifiers may use the ball because
their target is the team, not alive/dead.

## Deliberately unused inputs

Video, broadcast tracking, homography and Smart Tagging are not read. The challenge allows a
tactical-tracking-only solution, and none of these inputs is needed to reach the shipped score.
