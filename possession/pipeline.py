"""The submitted model: classifier ensembles, a 3-state Viterbi decoder and the control rule.

1. Player-track gaps of at most PLAYER_GAP_FILL_SECONDS are filled by linear interpolation.
2. P(dead) averages four classifiers and P(home | live) eight (gbm.KINDS, tcn.KINDS), in the
   nesting of the validated ensembles: the mean of the clean models and of the models trained also
   with lost player tracks. Each classifier is trained on the external matches only.
3. Within each half independently, a Viterbi decoder over home, away and dead turns the per-frame
   probabilities into runs, with a cost in nats for each team change and for each live/dead change.
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
    steps = tcn.probability(tcn.load(kind), *inputs, kind, match.halves or None)
    full = np.full(len(match.labels), 0.5)
    offset = 0
    for start, stop in (match.halves or ((0, len(match.labels)),)):
        count = (stop - start) // tcn.STEP_FRAMES
        values = steps[offset:offset + count * tcn.STEP_FRAMES]
        offset += count * tcn.STEP_FRAMES
        if len(values):
            full[start:start + len(values)] = values
            full[start + len(values):stop] = values[-1]
    return full


def _validated_halves(halves, n_frames):
    """Ordered, non-overlapping integer [start, stop) spans within 0..n_frames.

    The container must be a tuple or list. Zero-length spans are allowed and skipped by decoding;
    an empty container means no valid frames. Invalid spans raise ValueError, never clip or sort.
    """
    if not isinstance(halves, (tuple, list)):
        raise ValueError("halves must be a tuple or list of frame spans")
    result = []
    previous_stop = 0
    for span in halves:
        try:
            start, stop = span
        except (TypeError, ValueError):
            raise ValueError("each half must contain exactly two integer bounds") from None
        if any(isinstance(bound, (bool, np.bool_)) or not isinstance(bound, (int, np.integer))
               for bound in (start, stop)):
            raise ValueError("half bounds must be integers")
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= n_frames:
            raise ValueError("half bounds must satisfy 0 <= start <= stop <= frame count")
        if start < previous_stop:
            raise ValueError("halves must be ordered and non-overlapping")
        result.append((start, stop))
        previous_stop = stop
    return tuple(result)


def _decode_halves(p_dead, p_home, halves, n_frames):
    """Decode/control each valid half independently; return int8 X everywhere else."""
    halves = _validated_halves(halves, n_frames)
    p_dead, p_home = np.asarray(p_dead), np.asarray(p_home)
    if p_dead.shape != (n_frames,) or p_home.shape != (n_frames,):
        raise ValueError("probabilities must have one value per match frame")
    labels = np.full(n_frames, data.X, dtype=np.int8)
    state_codes = np.array([data.H, data.A, data.D], dtype=np.int8)
    for start, stop in halves:
        if start == stop:
            continue
        states = decoding.possession_states(p_dead[start:stop], p_home[start:stop],
                                           TEAM_SWITCH_COST, DEAD_SWITCH_COST)
        labels[start:stop] = data.apply_control_rule(state_codes[states])
    return labels


def predict(match):
    """H/A/D independently decoded and controlled per half; X outside valid half spans.

    Half-span errors raise ValueError before model calls. No nonempty spans returns all X.
    Probability generation and decoding/control reset independently at each half.
    """
    n_frames = len(match.labels)
    halves = _validated_halves(match.halves, n_frames)
    if not any(start < stop for start, stop in halves):
        return np.full(n_frames, data.X, dtype=np.int8)
    match = augment.fill_player_gaps(data.only_halves(match), PLAYER_GAP_FILL_SECONDS)
    return _decode_halves(dead_probability(match), home_probability(match), halves, n_frames)
