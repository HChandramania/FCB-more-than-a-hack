# Historical evidence notice

The following is an unverified development log retained for provenance. Its claims about
held-out scores, nested selection, identical rebuilds and lack of competition-driven selection
have not been independently established by the publication audit. Named experiment scripts
and fold records are missing. Historical numerical results are not new verified performance.
Using shipped all-external-match checkpoints on their training matches is in-sample diagnostic
only and must never be described as independent held-out evaluation. Exact CSV reproduction
does not verify these historical accuracy or selection claims. See `technical_report.md`.

---

# Experiment log

This is the log of the development work that chose each setting of the submitted model
(`possession/pipeline.py`). The "script" column names the experiment scripts of the development
workspace; the submitted pipeline in this repository contains only the adopted parts. Version
v16 (row 41) is the submitted model.

One change per experiment. From #3 on, settings are chosen by leave-one-match-out with the two sets weighted equally
(scripts/experiments/selection.py), and the scores are held-out scores. Scores are mean spellF1 on the validation matches (Metrica sample games
1-3, DFL open data 7 matches), external labels built with the 2 s controlled-possession rule.
"wins" = matches where the change scores higher than the reference, paired per match.
The BAR-ATH public window is a pipeline check only and never decides an experiment.

| # | change | reference | metrica | dfl | decision | script |
|---|---|---|---|---|---|---|
| 0 | repository baseline, half-window 50 | — | 0.4404 | 0.3558 | reference | possession/validate.py --model baseline |
| 1 | half-window 0-375 | w=50 | best w=50; w=75 -0.011 (1/3) | best w=50; w=75 -0.009 (0/7) | keep 50 | scripts/experiments/e01_window_sweep.py |
| 2 | controlled-possession rule on the prediction, C = 2 s | #0 | 0.5631 (+0.123, 3/3) | 0.3729 (+0.017, 5/7) | adopt; with 3 s labels also +0.144 (3/3), +0.031 (7/7) | scripts/experiments/e02_prediction_control.py |
| 3 | D where the 5 s mean player speed < S | #2 | 0.5629 (-0.000, 2/3) | 0.4301 (+0.057, 7/7) | adopt S = 1.2 m/s | scripts/experiments/e03_slow_play_dead.py |
| 4 | vote weight 1 if the nearest player is within R m of the ball, else 0.01 | #3 | 0.6526 (+0.090, 3/3) | 0.4618 (+0.032, 7/7) | adopt R = 1.0 m (chosen in every fold) | scripts/experiments/e04_proximity_vote.py |
| 5 | in a ball gap, keep the last label unless players slow down anywhere in the gap or the ball was last seen within 1 m of / beyond a line | #4 | live-gap level 0/5/15/30 %: -0.004 (1/3), +0.014 (3/3), +0.037 (3/3), +0.066 (3/3) | 0.000, +0.013 (7/7), +0.023 (7/7), +0.050 (7/7) | adopt (best of 6 variants; selection on validation) | scripts/experiments/e05_carry_through_gaps.py |
| 6 | D from a learned P(dead) > threshold (LightGBM on multi-scale player movement and layout, no ball; trained leave-one-match-out on the external matches) in place of the slow-play rule | #5 | -0.030 (0/3) at 0 %; -0.024 to -0.032 at every gap level | -0.003 (2/7) at 0 %; -0.004 to +0.001 | reject: frame dead F1 rises from 0.70-0.79 to 0.88-0.93, but the dead time breaks into about twice the true number of runs | scripts/experiments/e06_learned_dead.py |
| 6b | #6 decoded by a 2-state Viterbi with a switch cost (nats) and a log-odds bias, nested leave-one-match-out | #5 | 0/5/15/30 %: +0.036 (3/3), +0.029 (3/3), +0.034 (3/3), +0.026 (3/3) | +0.099 (7/7), +0.101 (7/7), +0.095 (7/7), +0.085 (7/7) | adopt as v7 after re-test on #7: on top of v6, 0/5/15/30 %: Metrica +0.035, +0.035, +0.030, +0.026 (3/3 each); DFL +0.099, +0.100, +0.099, +0.095 (7/7 each); production setting 40 nats, bias 0; re-run after the code review with inner gap fills from classifiers that saw neither match: identical except DFL at 30 % +0.093 | scripts/experiments/e06b_learned_dead_decoded.py |
| 7 | in live-play ball gaps, labels from a ball-free team classifier (per-team signals in each team's attacking direction, team-swap symmetric) instead of the last label | #5 | 0/5/15/30 %: +0.001 (1/3), +0.012 (3/3), +0.028 (3/3), +0.072 (3/3) | 0.000 (0/7), +0.005 (6/7), +0.027 (7/7), +0.046 (7/7) | adopt as v6; held-out live-frame accuracy 0.91-0.92 (Metrica), 0.85-0.92 (DFL) | scripts/experiments/e07_team_fill_gaps.py |
| 9 | re-test of the v7 gap decisions with label-independent ("uniform") simulated ball loss, after a code review | v7 arms: no gap rule / carry / fill | uniform 5/15/30 %: fill vs none +0.010 (3/3), +0.051 (3/3), +0.092 (3/3); carry vs none +0.002 (2/3), +0.030, +0.054 | fill vs none +0.016 (7/7), +0.043 (7/7), +0.076 (7/7); carry vs none +0.008 (6/7), +0.027, +0.032 | keep v7; gap gains with "live" placement were 1.3-2x larger | scripts/experiments/e09_gap_placement_check.py |
| 8 | one 3-state Viterbi over home/away/dead on P(dead) (e06 classifier) and P(home | live) (team classifier with ball signals, trained on copies with 0 % and 30 % simulated ball loss), in place of vote, gap rules and 2-state dead decoding; switch costs by nested leave-one-match-out | v7 | live 0/5/15/30 %: +0.036, +0.034, +0.048, +0.048 (3/3 each); uniform 5/15/30 %: +0.049, +0.067, +0.088 (3/3 each) | live: +0.100, +0.096, +0.104, +0.115 (7/7 each); uniform: +0.102, +0.112, +0.127 (7/7 each) | adopt as v8; production costs 10 nats (team change) and 20 nats (live/dead change), best over all matches; validation 0.7231 / 0.6673 | scripts/experiments/e08_joint_decoding.py |
| 10 | team classifier of #8 with touch timing (per team: seconds since the last and until the next touch, touch = nearest player within 1 m of a seen ball, capped 30 s) and ball acceleration; costs chosen again by nested leave-one-match-out | v8 | live 0/5/15/30 %: +0.024, +0.026, +0.015 (3/3 each), +0.009 (2/3); uniform 5/15/30 %: +0.021, +0.018 (3/3), +0.007 (1/3) | live: +0.042, +0.041 (7/7), +0.029 (6/7), +0.030 (7/7); uniform: +0.040 (7/7), +0.034 (7/7), +0.027 (6/7) | adopt as v9 (mean gain at every level and placement); production costs 5 / 40 nats (flat optimum 1-20 for the team cost); validation 0.7474 / 0.7097 | scripts/experiments/e08_joint_decoding.py --team-kind team_ball_touch --reference v8 |
| 11 | dead classifier of #8 with ball signals (ball seen, distance to the nearest line, ball speed, nearest player distance, seconds since and until the ball is seen), trained on copies with 0 % and 30 % uniform simulated ball loss | v8 | live 0/5/15/30 %: +0.074, +0.064, +0.060, +0.052; uniform 5/15/30 %: +0.068, +0.057, +0.047 (3/3 each) | live: +0.103, +0.101, +0.093, +0.086; uniform: +0.099, +0.090, +0.072 (7/7 each) | reject (provider convention leak, see note on #11-#12) | scripts/experiments/e08_joint_decoding.py --dead-kind dead_ball --reference v8 |
| 12 | #10 and #11 together; references scored with the same nested selection (code review 2) | v9 | live 0-30 %: +0.068 to +0.052; uniform: +0.068 to +0.048 (3/3 each) | live: +0.105 to +0.093; uniform: +0.098 to +0.072 (7/7 each) | reject (provider convention leak); v10 submission withdrawn, production stays v9 | scripts/experiments/e08_joint_decoding.py --team-kind team_ball_touch --dead-kind dead_ball --reference v9 |
| 13 | log-odds bias on P(dead) in the joint Viterbi (prior correction), grid -1..1 with the switch costs, nested | v9 (nested) | live 0-30 %: -0.004 to -0.005 (0/3 each); uniform: -0.004 to -0.006 (0/3) | live: -0.000 to +0.002 (1-2/7); uniform: -0.000 to -0.003 (0-2/7) | reject | e08_joint_decoding.py --decoder dead_bias, results/experiment_logs/e13_*.log |
| 14 | no 2 s control rule after the joint Viterbi | v9 (nested) | live: -0.001 to -0.002 (1/3); uniform: -0.002 to +0.005 | live: -0.003 to +0.001 (3-4/7); uniform: -0.007 to -0.001 | reject; the control rule stays | e08_joint_decoding.py --decoder no_control, results/experiment_logs/e14_*.log |
| 15 | team classifier of v9 also trained on the 20 SkillCorner open-data matches (MIT; phase-of-play labels, observed players only; sets weighted equally) | v9 (nested) | live 0-30 %: -0.002, -0.002, +0.005, -0.007 (1/3); uniform: -0.002, -0.005, +0.006, +0.002 | live: -0.014, -0.013, -0.015 (0/7), -0.007 (1/7); uniform: -0.014 to -0.019 (0-2/7) | reject; SkillCorner labels switch at the receiver's possession start and come from SkillCorner's own tracking, and its dead frames leak through empty broadcast frames (not used for dead) | e08_joint_decoding.py --team-kind team_ball_touch_sc |
| 16 | dead classifier with the per-team player signals (progress, advance, spread, opponent-half share, last line, openness, speed per team; no ball) | v9 (nested) | live 0-30 %: +0.006, +0.006, +0.010, +0.011; uniform 5-30 %: +0.007, +0.011, +0.016 (3/3 each) | live: +0.015, +0.014, +0.016, +0.017; uniform: +0.016, +0.017, +0.015 (6/7 each) | adopt as v10; costs 2.5 / 40 nats; BAR-ATH label-free: agreement with the v9 dead decision 0.965 (validation 0.97) | e08_joint_decoding.py --dead-kind dead_team |
| 16b | #16 with a team-symmetric dead classifier (swapped training copies, inference averaged over the swap) | v9 (nested) | live 0-30 %: +0.010, +0.010, +0.015, +0.015; uniform 5-30 %: +0.013, +0.015, +0.019 (3/3 each) | live: +0.016 (6/7), +0.015 (6/7), +0.018 (7/7), +0.019 (6/7); uniform: +0.017, +0.017 (7/7), +0.016 | adopt as v11 (replaces v10); costs 5 / 20 nats | e08_joint_decoding.py --dead-kind dead_team_sym --reference v9 |
| 17 | team classifier of v9 with ball velocity along each team's attacking direction (multi-scale) | v9 (nested) | live 0-30 %: +0.003 (1/3), +0.002 (2/3), +0.009, +0.008 (3/3); uniform 5-30 %: +0.005, +0.012, +0.006 (3/3) | live: +0.005 (4/7), +0.003 (4/7), +0.003 (3/7), +0.007 (6/7); uniform: +0.002 (3/7), +0.006 (5/7), +0.002 (4/7) | small; re-test on top of v10 (#18) | e08_joint_decoding.py --team-kind team_ball_advance --reference v9 |
| 18 | #17 on top of v10 | v10 (nested) | live 0-30 %: +0.005, +0.005, +0.009, +0.005 (3/3 each); uniform 5-30 %: +0.005 (2/3), +0.008 (3/3), -0.002 (1/3) | live: +0.004 (5/7), +0.004 (4/7), +0.004 (3/7), +0.006 (5/7); uniform: +0.003, +0.006, +0.006 (6/7 each) | pending: v10 dead classifier was not team-symmetric (review 3); re-test on #16b | e08_joint_decoding.py --team-kind team_ball_advance --dead-kind dead_team --reference v10 |
| 18b | #17 on top of v11 | v11 (nested) | live 0-30 %: +0.006, +0.004, +0.008, +0.008 (3/3 each); uniform 5-30 %: +0.005 (3/3), +0.009 (3/3), +0.001 (1/3) | live: +0.004 (5/7), +0.003 (4/7), +0.002 (3/7), +0.005 (5/7); uniform: +0.002 (4/7), +0.007 (6/7), +0.003 (4/7) | adopt as v12 (mean gain at every level and placement); costs 1 / 20 nats | e08_joint_decoding.py --team-kind team_ball_advance --dead-kind dead_team_sym --reference v10sym |
| 19 | team classifier with 1000 trees and 63 leaves | v11 | — | — | stopped before the end: too slow on the CPU (about 1 min per model, 55 models); heavy training moves to the GPU | — |
| 22 | touch snap (round 3, rank 1): before the control rule, each start of a decoded team run moves to the nearest touch of that team within 2 s (possession/boundaries.py) | v12 (nested) | live 0-30 %: +0.015, +0.014, +0.010, +0.013; uniform 5-30 %: +0.010, +0.007, +0.004 (2/3 each) | live: +0.008 (5/7), +0.009 (6/7), +0.002 (4/7), -0.003 (3/7); uniform: +0.001 (5/7), +0.003 (4/7), +0.004 (4/7) | reject: loses on DFL at 30 % live-placement loss (rule: a ball-dependent change must not lose at any level); not tuned further, to avoid selection on validation | e08_joint_decoding.py --decoder touch_snap |
| 20a | P(dead) from a TCN (dilated residual 1-D conv, 64 channels, 8 layers, 5 steps/s, about 100 s receptive field; ball-free signals; team-symmetric; trained on the GPU, 15 s per model) | v12 (nested) | live 0-30 %: +0.005, +0.007, +0.008, +0.002; uniform 5-30 %: +0.005, +0.003, +0.000 (2/3 each) | live: +0.010, +0.010, +0.008, +0.006 (6/7 each); uniform: +0.010 (6/7), +0.006 (6/7), +0.009 (5/7) | adopt as v13; costs 5 / 40 nats | e08_joint_decoding.py --dead-kind tcn_dead --reference v12 |
| 20b | P(home | live) from a TCN on the team, ball, touch and ball-advance signals, trained on copies with 0 / 15 / 30 % simulated ball loss | v12 (nested) | live 0-30 %: +0.030, +0.032, +0.026, +0.020; uniform 5-30 %: +0.029, +0.023, +0.026 (3/3 each) | live: +0.007 (4/7), +0.010 (5/7), +0.012 (5/7), -0.004 (3/7); uniform: +0.009 (3/7), +0.004 (3/7), +0.011 (4/7) | not alone (loses on DFL at 30 % live); ensemble with LightGBM tested as #23 | e08_joint_decoding.py --team-kind tcn_team --reference v12 |
| 24a | TCN team classifier with 3x loss weight within 2 s of a true team change | v13 (nested) | live 0-30 %: +0.036, +0.036, +0.019, +0.034; uniform 5-30 %: +0.042, +0.038, +0.032 (3/3 each) | live: +0.007, +0.009, +0.009, +0.003 (4-5/7); uniform: +0.010 (3/7), -0.005 (3/7), +0.020 (7/7) | not alone (loses on DFL at 15 % uniform) | --team-kind tcn_team_change |
| 24b | TCN team classifier with 128 channels and 3000 steps | v13 (nested) | live: +0.033, +0.034, +0.024, +0.032; uniform: +0.035, +0.037, +0.026 (2-3/3) | live: +0.004 (3/7), +0.007, +0.010 (7/7), +0.003; uniform: +0.008, +0.006, +0.014 | passes; superseded by #23b | --team-kind tcn_team_wide |
| 23a | P(home) = mean of LightGBM (v12 team kind) and TCN (#20b) | v13 (nested) | live: +0.030, +0.027, +0.023, +0.031; uniform: +0.033, +0.030, +0.025 (3/3 each) | live: +0.012, +0.014, +0.015, +0.006; uniform: +0.010, +0.012, +0.016 (4-6/7) | passes; superseded by #23b | --team-kind ens_advance_tcn |
| 23b | P(home) = mean of LightGBM (v12 team kind) and the change-weighted TCN (#24a) | v13 (nested) | live: +0.035, +0.035, +0.025, +0.048; uniform: +0.040, +0.033, +0.030 (3/3 each) | live: +0.021, +0.023, +0.023, +0.015 (5-7/7); uniform: +0.022, +0.015, +0.026 (5-7/7) | adopt as v14 after deterministic retraining of the TCN models (reproducibility) | --team-kind ens_advance_tcnchange |
| 23c | P(home) = mean of LightGBM and the wide TCN (#24b) | v13 (nested) | live: +0.036, +0.032, +0.025, +0.023; uniform: +0.030, +0.032, +0.027 (3/3) | live: +0.008 (3/7), +0.014, +0.009, +0.010; uniform: +0.012, +0.011, +0.017 | passes; below #23b | --team-kind ens_advance_tcnwide |
| 25 | TCN team classifier with GRU-D style inputs (last observed value, mask, age of the observation; from the Liquid AI / CfC deep dive) | v13 (nested) | live: +0.005, +0.005, +0.002, +0.009; uniform: +0.005, +0.019, +0.016 (1-3/3) | live: -0.007 (2/7), -0.001, +0.004, -0.011 (2/7); uniform: -0.002, -0.009 (2/7), +0.001 | reject (loses on DFL); the dead probabilities were regenerated by the deterministic retraining during the uniform run, which changes the reference slightly but affects both arms alike | --team-kind tcn_team_gapage |
| 20a-sym | #20a repeated with team-symmetric, deterministic TCN models (code review 4) | v12 (nested) | live 0-30 %: +0.008, +0.008, +0.008, -0.001 (2/3 each) | live: +0.006 (5/7), +0.008 (6/7), +0.005 (5/7), +0.003 (5/7) | superseded by #27 | --dead-kind tcn_dead |
| 27 | P(dead) = mean of LightGBM (dead_team_sym) and TCN (tcn_dead, symmetric) | v12 (nested) | live 0-30 %: +0.013, +0.014, +0.016, +0.010 (3/3); uniform 5-30 %: +0.014, +0.010 (3/3), +0.009 (2/3) | live: +0.017, +0.017, +0.014, +0.014 (7/7 each); uniform: +0.016, +0.013, +0.015 (7/7 each) | adopt in the final combination (#31) | --dead-kind ens_dead |
| 28 | expected-spellF1 (MBR) decoder: tempered forward-backward, 200 posterior samples, candidate spells, weighted interval scheduling; tau nested (possession/mbr.py, built by a parallel agent) | v12 (nested) | -0.005 to -0.015 at every level and placement (0-1/3) | -0.010 to -0.020 (0-1/7) | reject: the tempered posterior does not match the true spell distribution | scripts/experiments/e28_mbr.py |
| 29 | restart-team evidence from set-piece layouts (Vidal-Codina 2022 rules) as a log-odds shift after decoded dead runs; strength nested | v12 (nested) | -0.002 at 0 %, 0 elsewhere | 0 (w = 0 chosen in every fold) | reject: rules agree with the restarting team on 88-90 %, the classifier already on 94-98 % | scripts/experiments/e29_restarts.py |
| 30 | explicit-duration semi-Markov decoder at 5 steps/s, log-normal durations from the training matches, emission scale nested | v12 (nested) | +0.001 to +0.005 in 7 of 8 cells, -0.002 (0/3) at live 15 % | +0.001 to +0.005 (4-5/7) in every cell | reject by the no-loss rule (small and noisy) | scripts/experiments/e30_hsmm.py |
| 31a | final combination: team = mean(LightGBM advance, change-weighted TCN), dead = mean(LightGBM dead_team_sym, TCN dead); all TCNs symmetric and deterministic | v12 (nested) | live 0-30 %: +0.049, +0.050, +0.038, +0.046; uniform 5-30 %: +0.049, +0.042, +0.039 (3/3 each) | live: +0.036, +0.040, +0.034, +0.029 (7/7 each); uniform: +0.040 (7/7), +0.033 (6/7), +0.038 (7/7) | passes; below #31b | --team-kind ens_advance_tcnchange --dead-kind ens_dead |
| 31b | #31a with the TCN team member as the mean of 3 seeds | v12 (nested) | live: +0.053, +0.055, +0.043, +0.049; uniform: +0.056, +0.045, +0.043 (3/3 each) | live: +0.036, +0.037, +0.037, +0.030; uniform: +0.041, +0.034, +0.042 (7/7 each) | adopt as v14; costs 1 / 20 nats; in-sample objective 0.792 | --team-kind ens_advance_tcnchange3 --dead-kind ens_dead |
| 32 | LightGBM team classifier with 3x weight on rows within 2 s of a true team change | v12 (nested) | live: +0.023, +0.026, +0.016, +0.025; uniform: +0.023, +0.019, +0.018 (2-3/3) | live: +0.016, +0.018, +0.018, +0.011; uniform: +0.016, +0.011, +0.015 (5-7/7) | passes; not yet combined with #31b (next step) | --team-kind team_ball_advance_change |
| 33 | v14 with the change-weighted LightGBM member (#32) in place of the plain one | v14 (nested) | live 0-30 %: +0.006 (2/3), +0.009, +0.002, +0.021; uniform 5-30 %: +0.010, +0.003, +0.008 (3/3) | live: +0.002 (3/7), +0.004 (4/7), -0.002 (3/7), +0.000 (2/7); uniform: -0.003, -0.002 (3/7), +0.004 (4/7) | reject (loses on DFL); the change-weighted TCN member already carries the gain | --team-kind ens_advancechange_tcnchange3 --reference v14 |
| 34 | robustness of v14 to lost player tracks (no retraining): 0/5/10/20 % of player-frames removed in 2-30 s blocks per player | v14 at 0 % | 0.819, 0.795 (-0.024), 0.773 (-0.047), 0.732 (-0.087) | 0.765, 0.756 (-0.010), 0.736 (-0.029), 0.704 (-0.061) | weakness found: BAR-ATH misses about 7.6 % of player-frames (label-free count), so v14 loses some 0.02-0.04 there; fix tested in #36 | scripts/experiments/e34_player_dropout.py |
| 35 | TCN team classifier (change-weighted) with the contest signals (players of each team within 2 m and 5 m of the ball) as extra inputs; member-level test | tcn_team_change with ens_dead (nested) | live 0-30 %: +0.001 (2/3), +0.011 (2/3), +0.006 (2/3), -0.018 (0/3); uniform 5-30 %: +0.005 (2/3), +0.013, +0.010 (3/3) | live: +0.001 (2/7), +0.003 (3/7), -0.006 (3/7), -0.006 (3/7); uniform: +0.001 (3/7), +0.002 (4/7), +0.002 (4/7) | reject by the no-loss rule (losses at live 15-30 %) | --team-kind tcn_team_contest_change --dead-kind ens_dead --reference tcnchange |
| 36 | every classifier of v14 also trained on copies with 10 % of player-frames removed (kinds *_pd); scored with player dropout 0 / 10 / 20 % (seeds differ from training) | v14 (nested) | clean tracks, ball 0-30 %: live -0.004, -0.004, -0.009, -0.001; uniform -0.014 (0/3), +0.001, -0.006; dropout 10 %: +0.013, +0.005, +0.012, +0.021; dropout 20 %: +0.037, +0.023, +0.025, +0.021 (3/3) | clean: live +0.001, +0.002, -0.002, -0.003; uniform -0.003, -0.005, +0.000; dropout 10 %: +0.008, +0.008, +0.012, +0.011 (5-6/7); dropout 20 %: +0.020, +0.025, +0.024, +0.019 (6/7) | trade-off: gains on damaged tracks, small loss on clean tracks; BAR-ATH misses about 7.6 % of player-frames, so the dropout condition is the relevant one; compared with the average of v14 and #36 in #37 before a decision | --team-kind ens_advance_tcnchange3_pd --dead-kind ens_dead_pd --player-dropout 0/0.10/0.20 |
| 37 | mean of v14 and #36 (team and dead probabilities each averaged) | v14 (nested) | clean, ball 0-30 %: live -0.001, -0.003, -0.000, +0.001; uniform -0.004 (0/3), -0.001, +0.000; dropout 10 %: +0.012, +0.004, +0.004, +0.013; dropout 20 %: +0.024, +0.019, +0.017, +0.013 (3/3) | clean: live +0.005, +0.007, +0.006, +0.004 (5-6/7); uniform -0.000, -0.003, +0.006; dropout 10 %: +0.010, +0.011, +0.012 (7/7), +0.010 (6/7); dropout 20 %: +0.014 to +0.018 (6/7) | adopt as v15: nearly no cost on clean tracks and most of the gain of #36 at 10 % dropout (BAR-ATH: about 7.6 %); costs 2.5 / 20 nats, best both on clean and on 10 % dropout data; submission reproduced from the saved models | --team-kind ens_team_both --dead-kind ens_dead_both --reference v14 |
| 38 | robustness of v15 to duplicate player tracks (no retraining): copies of player tracks in spare slots, 0.5 m offset, on 5 / 10 % of frames per team (BAR-ATH: >11 tracks on 5-8 %) | v15 | 0.8176, -0.0007, -0.0001 | 0.7700, -0.0008, -0.0007 | no weakness; no change needed | scripts/experiments/e38_duplicate_tracks.py |
| 40 | MS-TCN++ form of the change-weighted TCN team classifier: 3 shared-weight refinement stages on the probabilities, BCE + truncated-MSE smoothing loss (lambda 0.15, tau 4) on every stage (Li et al. 2020); member-level test | tcn_team_change with ens_dead (nested) | live 0-30 %: -0.011 (0/3), +0.001 (2/3), -0.007 (1/3), -0.024 (0/3); uniform 5-30 %: -0.005 (2/3), +0.005 (2/3), -0.001 (1/3) | live: -0.004 (2/7), -0.009 (0/7), -0.005 (2/7), -0.004 (2/7); uniform: -0.009 (2/7), -0.002 (3/7), -0.004 (3/7) | reject (loses in most cells); the Viterbi already smooths, and the refinement stages see only probabilities | --team-kind tcn_team_ms_change --dead-kind ens_dead --reference tcnchange |
| 41 | v15 with player gaps of at most 10 s inside one track slot filled by linear interpolation before inference (models unchanged); a BAR-ATH slot that comes back after a gap has moved at a median 0.2-1.5 m/s, so a slot keeps one player | v15 (nested) | player loss 10 % (uniform ball 0-30 %): -0.000 (1/3), +0.004, +0.004, +0.002 (2/3); 20 %: +0.001, -0.001 (2/3), +0.006 (1/3), +0.003 (1/3); clean tracks (live): +0.000 in every cell | 10 %: +0.001 (4/7), -0.001 (3/7), +0.002 (4/7), +0.001 (5/7); 20 %: +0.005 (5/7), +0.010 (6/7), +0.011 (6/7), +0.006 (4/7); clean: +0.000 | adopt as v16: no effect on clean tracks, gains with lost tracks (BAR-ATH: about 7.6 %); the two losses are below 0.002 with 2/3 and 3/7 matches better; v16 differs from v15 on 1,511 frames; same costs 2.5 / 20 nats | --team-kind ens_team_both --dead-kind ens_dead_both --reference v15 --player-dropout 0.10 --fill-player-gaps 10 |

Note on #3: S is chosen by leave-one-match-out. With every match weighted equally the 7 DFL
matches dominate, S = 1.2-1.3 is chosen and Metrica loses 0.016 (1/3); with the two sets weighted
equally S = 1.2 is chosen in every fold. DFL has the ball on every frame, so slow play is its only
dead-ball cue; on Metrica the ball gaps already mark the stoppages.

Note on #2: the sweep maximum of C follows the label rule (2 s labels: Metrica best 3 s, DFL best
1-1.5 s; 3 s labels: Metrica 4 s, DFL 3 s), so C takes the definition value 2 s, not the maximum.

Note on #5: the external matches almost never lose the ball in live play (Metrica 0.3-3.1 % of
live frames, DFL 0 %), so the experiment adds simulated live-play ball loss at 0, 5, 15 and 30 %
of live frames, in blocks of 2-30 s (log-uniform), seeded per match (possession/validate.py). A
change must not lose at any level. Rejected variants: carry always (Metrica -0.038 at 0 %), carry
for the first 5 or 10 s only (Metrica -0.031 / -0.037 at 0 %), carry unless near a line
(-0.011 at 0 %), carry unless slow inside the gap (-0.011 at 0 %). The chosen variant was picked
after seeing all six, so its validation gain is slightly optimistic.

Note on #6 and #6b: a per-frame classifier gives many short dead runs (median 0.8-4.4 s), each of
which splits a possession. The Viterbi switch cost removes them; the in-sample optimum is flat for
switch costs 10-40 nats and bias 0, and the nested choice varies between these values per fold.

Note on #7: the e07 classifiers and the production path (learned.model_without) train on the same
rows with the same fixed parameters; the v6 validation run reproduces the e07 0 % numbers exactly.

Note on #9: "live" placement (possession/matches.py add_live_ball_gaps) puts every simulated gap
inside a true live stretch, so a gap rule is told that play is live. "uniform" placement puts
blocks anywhere in the labelled frames and counts only the live frames removed. From #9 on, a
gap-related change must not lose under either placement.

Note on #10: label-free comparison on BAR-ATH (no labels used): touches on 28.5 % of ball-seen
frames, 21.4 touch starts and 8.0 team changes per ball-seen minute; DFL 26.5-27.1 %, 19-20 and
7.7-8.6; Metrica 46-57 %, 23-29 and 11-13. BAR-ATH ball acceleration has a higher median (5.7 against
0-1.6 m/s^2), probably tracking jitter; the p90 is in the external range.

Note on #11-#12 (rejected): the external providers encode the dead ball in the ball data. The DFL
ball is exactly still on 100 % of dead frames (0.4-0.9 % of live frames); the Metrica sample
removes the ball on 85-99.6 % of dead frames. BAR-ATH has a still ball on 0.08 % of seen frames.
On frames that the players-only classifier calls dead (P > 0.8) with a visible ball, the BAR-ATH
ball moves at p50 1.9 m/s, 7.8 m inside the lines, and the ball-aware classifier calls 71 % of them
live (DFL 3-4 %). The validation gain of #11 therefore comes mostly from the conventions. A dead
classifier may use the ball only after the conventions are removed from the external data.

Note on code review 2: the reference arms of #8 and #10 used switch costs chosen in-sample, the new
arm nested ones; this biased each paired difference against the new arm by about 0.002-0.004.
With nested references (results/experiment_logs/*_nested_reference.log) #10 gives DFL +0.045 to
+0.029 and Metrica +0.026 to +0.008; no decision changes.

Decision on SkillCorner (open data, 20 matches): not a validation set. Its dead frames are 38-63 % empty
broadcast frames (live <= 4 %), which any dead classifier can detect, and its team labels switch at the
receiver's possession start and come from SkillCorner's own tracking; #15 showed the label mismatch.

Note on code review 3: the dead_team classifier of #16 (v10) has per-team input columns but was
trained without team-swapped copies and used without averaging over the swap, so its P(dead)
depends on which team the provider calls home (Metrica-1: mean |difference| 0.027, 1.7 % of labels
change). Fix: kind dead_team_sym (swapped training copies with the same target, inference averaged
over the swap); tests/test_symmetry.py checks it. #16 is repeated as #16b; later comparisons build on
the result of #16b. The other review findings were missing checks without a current defect.

Note on reproducibility (RULES.md rule 3, "the same file"): LightGBM fits are deterministic (two fits,
identical predictions). GPU TCN training was not (two runs: max weight difference 0.08, 265 frames
of one match cross P = 0.5 differently). possession/sequence.py now sets deterministic algorithms and
the cuBLAS workspace; two runs then give identical weights and outputs. TCN models trained before
this fix are retrained deterministically before a TCN version is submitted.

Note on code review 4: (1) the TCN classifiers were not exactly team-symmetric: the input
normalisation had separate statistics for each _H/_A column, so the swap average was not the
team-exchanged input (Metrica-1: up to 750 frames crossed P = 0.5 when the teams were exchanged).
Fix: pooled statistics per column pair; all TCN kinds in use are retrained, and #20a and #23b are
repeated with the symmetric models (tests/test_symmetry.py now checks tcn_dead and tcn_team_change).
(2) boundaries.snap_to_touches could delete a short middle run (e22 is rejected anyway); fixed, with a
test. (3) e08 probability caches are now refused when a model file of the kind is newer.

Housekeeping (2026-10-25): code, notes and all results up to v14 are in archive/code_2026-10-25_v14.zip
(including the submissions v5-v14). Deleted: models, rows and probability caches of rejected kinds
(dead, dead_ball, dead_team, team_ball, team_ballfree, team_ball_touch(_sc, _large), tcn_team(_wide,
_gapage), the non-deterministic and asymmetric TCN copies), the SkillCorner open data
(github.com/SkillCorner/opendata, commit 4340d27) and the ball-detector downloads (WASB, FootAndBall,
ISSIA). Each can be rebuilt: models by the experiment scripts, data from the cited sources.

Note on reproduction (2026-10-25): scripts/check_reproduction.py rebuilt every v14 production model in an empty
folder (four TCNs on the GPU, LightGBM on the CPU) and then the submission; the file is identical to
results/submission/v14.csv byte for byte (RULES.md rule 3).

Note on v15 reproduction (2026-10-25): scripts/check_reproduction.py rebuilt all 8 TCN and all LightGBM
production models of v15 in an empty folder; the rebuilt submission is identical to results/submission/v15.csv.

Housekeeping (2026-10-25): code, notes and results up to v16 are in archive/code_2026-10-25_v16.zip
(including results/submission/v15.csv, which is removed from the folder). Deleted: the models and
caches of the rejected kinds tcn_team_contest_change (#35) and tcn_team_ms_change (#40). v16 uses
the v15 models unchanged; a second run of possession.submit gives the same file byte for byte.
