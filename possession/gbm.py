"""Gradient-boosted per-frame classifiers (LightGBM) on multi-scale window features.

Two classifiers, each also in a variant trained with lost player tracks (suffix _pd):
- dead_team_sym: P(dead) from player movement and per-team layout. It has no ball input, because
  the external providers encode the dead ball in the ball data (DFL freezes it on every dead frame,
  the Metrica sample removes it), which the automatic competition tracking does not do.
- team_ball_advance: P(home | live) from per-team signals with the ball, touch timing and ball
  advance, trained also on copies with simulated ball loss.
Both see each training row also with the teams exchanged and average over the exchange at
inference, so their output is exactly team-symmetric.
"""
import zlib

import lightgbm
import numpy as np

from . import augment, data, features, paths

KINDS = ("dead_team_sym", "dead_team_sym_pd", "team_ball_advance", "team_ball_advance_pd")
TRAIN_STRIDE_FRAMES = 5  # neighbouring frames are almost equal after the window means; every 5th frame is enough
# Fixed before any run, not tuned: a common LightGBM setting for some 10^5 rows and some 10^2 features.
MODEL_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=0, n_jobs=8, verbose=-1)
# The external matches almost never lose the ball in live play; copies with simulated loss teach the
# with-ball classifier to fall back on the players where the ball is missing.
TRAIN_LIVE_GAP_SHARES = (0.0, 0.3)
PLAYER_DROPOUT_SUFFIX = "_pd"
TRAIN_PLAYER_DROPOUT = 0.10  # the competition tracking misses about 7.6 % of its 22 players per graded frame


def base_kind(kind):
    """The kind without the player-dropout suffix."""
    return kind[:-len(PLAYER_DROPOUT_SUFFIX)] if kind.endswith(PLAYER_DROPOUT_SUFFIX) else kind


def dropout_copy(match):
    """The training copy of a match with TRAIN_PLAYER_DROPOUT of its player-frames removed; the seed depends on the name."""
    return augment.add_player_dropout(match, TRAIN_PLAYER_DROPOUT, zlib.crc32(match.name.encode()) + 19)


def dead_features(match):
    """(names, (frames, features)) of dead_team_sym: pooled player signals and per-team signals without the ball."""
    names, table = features.player_signals(match)
    team_names, team_table = features.team_signals(match, with_ball=False)
    return features.multi_scale(names + team_names, np.concatenate((table, team_table), axis=1), match.halves or None)


def team_features(match):
    """(names, (frames, features)) of team_ball_advance: per-team signals, touch timing and ball advance.

    The touch timing signals enter raw and at multiple scales.
    """
    names, table = features.multi_scale(*features.team_signals(match, with_ball=True), segments=match.halves or None)
    touch_names, touch_table = features.touch_signals(match)
    scaled_names, scaled_table = features.multi_scale(touch_names, touch_table, match.halves or None)
    advance_names, advance_table = features.multi_scale(*features.ball_advance_signals(match), segments=match.halves or None)
    return (names + touch_names + scaled_names + advance_names,
            np.concatenate((table, touch_table.astype(np.float32), scaled_table, advance_table), axis=1))


def _dead_rows(match, set_name):
    """(features, is_dead, set name) on every TRAIN_STRIDE_FRAMES-th labelled frame, also with the teams exchanged.

    The dead label does not depend on which team is home, so the exchanged copy keeps the target.
    """
    names, table = dead_features(match)
    labelled = np.flatnonzero(match.labels != data.X)[::TRAIN_STRIDE_FRAMES]
    rows, is_dead = table[labelled], match.labels[labelled] == data.D
    return (np.concatenate((rows, features.swap_teams(names, rows))), np.concatenate((is_dead, is_dead)),
            np.full(2 * len(labelled), set_name))


def _team_rows(match, set_name):
    """(features, is_home, set name) on live frames of each ball-loss copy, also with the teams exchanged."""
    parts = []
    for share in TRAIN_LIVE_GAP_SHARES:
        gapped = augment.add_live_ball_gaps(match, share, zlib.crc32(match.name.encode()) + 7)
        names, table = team_features(gapped)
        live = np.flatnonzero((gapped.labels == data.H) | (gapped.labels == data.A))[::TRAIN_STRIDE_FRAMES]
        rows, is_home = table[live], gapped.labels[live] == data.H
        parts.append((np.concatenate((rows, features.swap_teams(names, rows))), np.concatenate((is_home, ~is_home)),
                      np.full(2 * len(live), set_name)))
    return tuple(np.concatenate(columns) for columns in zip(*parts))


def _cached_rows(kind, set_name, match_name, load):
    """Training rows of one external match and one base kind, cached; the match is loaded only on a cache miss."""
    path = paths.CACHE_DIR / "rows" / "{}-{}.npz".format(kind, match_name)
    if path.is_file():
        stored = np.load(path)
        return stored["table"], stored["target"], stored["sets"]
    build = _dead_rows if kind.startswith("dead_team_sym") else _team_rows
    match = load() if not kind.endswith(PLAYER_DROPOUT_SUFFIX + "only") else dropout_copy(load())
    rows = build(match, set_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_npz_atomic(path, table=rows[0], target=rows[1], sets=rows[2])
    return rows


def training_rows(kind, set_name, match_name, load):
    """Rows of one external match; a _pd kind has the rows of its base kind, then those of the dropout copy."""
    base = base_kind(kind)
    parts = [_cached_rows(base, set_name, match_name, load)]
    if kind.endswith(PLAYER_DROPOUT_SUFFIX):
        parts.append(_cached_rows(base + PLAYER_DROPOUT_SUFFIX + "only", set_name, match_name, load))
    return tuple(np.concatenate(columns) for columns in zip(*parts))


def fit(kind, excluded=()):
    """A LightGBM classifier of one kind on every external match except the excluded names.

    Each set of matches (Metrica, DFL) gets the same total weight, so the 7 DFL matches do not
    outweigh the 3 Metrica matches.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    rows = [training_rows(kind, set_name, match_name, load) for set_name, match_name, load in data.external_matches()
            if match_name not in excluded]
    table = np.concatenate([r[0] for r in rows])
    target = np.concatenate([r[1] for r in rows])
    set_names = np.concatenate([r[2] for r in rows])
    weights = np.ones(len(target))
    n_sets = len(np.unique(set_names))
    for name in np.unique(set_names):
        in_set = set_names == name
        weights[in_set] = len(target) / (n_sets * float(in_set.sum()))
    model = lightgbm.LGBMClassifier(**MODEL_PARAMS)
    model.fit(table, target, sample_weight=weights)
    return model


def model_path(kind):
    return paths.MODEL_DIR / "{}.txt".format(kind)


def save(model, kind):
    path = model_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    model.booster_.save_model(str(temporary))
    # A partial model file makes LightGBM abort the whole process on load, so only complete files get the final name.
    temporary.replace(path)


def load(kind):
    path = model_path(kind)
    if not path.is_file():
        raise FileNotFoundError("{} is missing; run scripts/train.py".format(path))
    return lightgbm.Booster(model_file=str(path))


def symmetric_probability(booster, names, table, mirrored):
    """P(class 1) averaged over the table and its team-exchanged copy.

    mirrored: the class flips when the teams are exchanged (P(home)); otherwise it stays (P(dead)).
    """
    direct = booster.predict(table)
    exchanged = booster.predict(features.swap_teams(names, table))
    if mirrored:
        return 0.5 * (direct + 1.0 - exchanged)
    return 0.5 * (direct + exchanged)
