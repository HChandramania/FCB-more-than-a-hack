"""The submitted model: classifier ensembles, a 3-state Viterbi decoder and the control rule.

1. Player-track gaps of at most PLAYER_GAP_FILL_SECONDS are filled by linear interpolation.
2. P(dead) averages four classifiers and P(home | live) eight (gbm.KINDS, tcn.KINDS), in the
   nesting of the validated ensembles: the mean of the clean models and of the models trained also
   with lost player tracks. Each classifier is trained on the external matches only.
3. A Viterbi decoder over home, away and dead turns the per-frame probabilities into runs, with
   a cost in nats for each team change and for each live/dead change.
4. The control rule of CHALLENGE.md: a run of the other team shorter than CONTROL_SECONDS is a
   touch or a contest, not a change of possession.
Each setting comes from nested leave-one-match-out validation on the external matches
(docs/experiments.md gives the experiment of each one).
"""
import numpy as np

from . import augment, data, decoding, gbm, tcn

TEAM_SWITCH_COST = 2.5  # nats
DEAD_SWITCH_COST = 20.0  # nats
PLAYER_GAP_FILL_SECONDS = 10.0  # the median gap of a competition track slot; fixed before any validation run


def _mean(values):
    return np.mean(values, axis=0)


def dead_probability(match):
    """P(dead) per frame: the mean over the clean and the dropout-trained pair of (LightGBM, TCN)."""
    names, table = gbm.dead_features(match)
    tcn_inputs = tcn.step_inputs(match, "tcn_dead")

    def member(kind):
        if kind.startswith("tcn_"):
            return _tcn_frames(match, kind, tcn_inputs)
        return gbm.symmetric_probability(gbm.load(kind), names, table, mirrored=False)
    return _mean([_mean([member("dead_team_sym"), member("tcn_dead")]),
                  _mean([member("dead_team_sym_pd"), member("tcn_dead_pd")])])


def home_probability(match):
    """P(home | live) per frame: for clean and dropout-trained models, the mean of LightGBM and a 3-seed TCN mean."""
    names, table = gbm.team_features(match)
    tcn_inputs = tcn.step_inputs(match, "tcn_team_change")

    def pair(suffix):
        lightgbm_member = gbm.symmetric_probability(gbm.load("team_ball_advance" + suffix), names, table, mirrored=True)
        seeds = _mean([_tcn_frames(match, "tcn_team_change" + seed + suffix, tcn_inputs) for seed in ("", "_s1", "_s2")])
        return _mean([lightgbm_member, seeds])
    return _mean([pair(""), pair(gbm.PLAYER_DROPOUT_SUFFIX)])


def _tcn_frames(match, kind, inputs):
    """TCN probability at the full frame count; the frames after the last full step repeat its value."""
    steps = tcn.probability(tcn.load(kind), *inputs, kind)
    full = np.full(len(match.labels), steps[-1] if len(steps) else 0.5)
    full[:len(steps)] = steps
    return full


def predict(match):
    """H/A/D code per frame of a match; frames outside its halves are decoded too and must be masked by the caller."""
    match = data.only_halves(augment.fill_player_gaps(match, PLAYER_GAP_FILL_SECONDS))
    states = decoding.possession_states(dead_probability(match), home_probability(match), TEAM_SWITCH_COST, DEAD_SWITCH_COST)
    labels = np.array([data.H, data.A, data.D], dtype=np.int8)[states]
    return data.apply_control_rule(labels)
