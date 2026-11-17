"""Per-frame signals, computed the same way on every match and every tracking provider."""
import warnings

import numpy as np

from .data import FPS, A, H

SPEED_STRIDE_FRAMES = 2  # speed from positions 2 frames before and after: less sensitive to tracking jitter
WALK_M_S = 2.0
STILL_M_S = 0.7
OUTSIDE_MARGIN_M = 1.0  # a player this far beyond a line stands outside the pitch (throw-in, corner taker)
CENTRE_CIRCLE_M = 9.15  # Laws of the Game, law 1
# Multi-scale windows after Lang et al. 2022 (doi:10.1038/s41598-022-19948-1), who use +-0.4 s to +-80 s.
CENTRED_WINDOWS_S = (0.5, 1.5, 4.5, 13.5, 40.0)
TREND_WINDOWS_S = (3.0, 9.0)
DISTANCE_CHUNK_FRAMES = 5000  # float64 at 46 slots: some 170 MB for the (frames, slots, slots, 2) temporary
BALL_SIGNALS = ("ball_distance", "ball_progress", "ball_speed", "ball_seen")
TOUCH_RADIUS_M = 1.0  # a player this near the ball controls it
TOUCH_HORIZON_S = 30.0  # times since and until a touch are capped here; longer means "not recently"


def window_mean(values, before_frames, after_frames):
    """Mean over frames t-before_frames .. t+after_frames that skips NaN; NaN where the window holds no value.

    values is (frames,) or (frames, columns).
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    zero_row = np.zeros((1,) + values.shape[1:])
    total = np.concatenate((zero_row, np.cumsum(np.where(valid, values, 0.0), axis=0)))
    count = np.concatenate((zero_row, np.cumsum(valid, axis=0, dtype=np.float64)))
    index = np.arange(len(values))
    low = np.maximum(0, index - before_frames)
    high = np.minimum(len(values), index + after_frames + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total[high] - total[low]) / (count[high] - count[low])


def _centred_difference(values, stride, segments=None):
    """Centered difference within each segment, leaving segment edges unavailable."""
    values = np.asarray(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    segments = ((0, len(values)),) if segments is None else segments
    for start, stop in segments:
        if stop - start > 2 * stride:
            result[start + stride:stop - stride] = values[start + 2 * stride:stop] - values[start:stop - 2 * stride]
    return result


def player_speeds(match):
    """(frames, slots) speed in m/s from positions SPEED_STRIDE_FRAMES before and after; NaN at track edges."""
    stride = SPEED_STRIDE_FRAMES
    step = _centred_difference(match.players_m, stride, match.halves or None)
    speed = np.sqrt((step ** 2).sum(axis=2)) * FPS / (2 * stride)
    return speed


def player_signals(match):
    """(names, (frames, signals)) movement and layout of all tracked players, teams pooled.

    The signals ignore team identity and the ball, so they mean the same on every match and every provider.
    Counts of tracks and the centroid motion are left out: automatic tracking starts and ends tracks at
    random, which changes both without any change in play.
    """
    speed = player_speeds(match)
    x, y = match.players_m[:, :, 0], match.players_m[:, :, 1]
    length, width = match.pitch_m
    tracked = np.isfinite(x)
    moving = np.isfinite(speed)
    n_moving = moving.sum(axis=1).astype(np.float64)
    outside = tracked & ((np.where(tracked, np.minimum(x, length - x), 0.0) < -OUTSIDE_MARGIN_M)
                         | (np.where(tracked, np.minimum(y, width - y), 0.0) < -OUTSIDE_MARGIN_M))
    centre_distance = np.hypot(x - length / 2, y - width / 2)
    in_circle = tracked & (np.where(tracked, centre_distance, np.inf) < CENTRE_CIRCLE_M)
    corners = np.array([(0.0, 0.0), (0.0, width), (length, 0.0), (length, width)])
    corner_distance = np.min([np.hypot(x - cx, y - cy) for cx, cy in corners], axis=0)
    with warnings.catch_warnings():
        # Frames without any tracked player give NaN on purpose; numpy warns about each such frame.
        warnings.simplefilter("ignore", RuntimeWarning)
        signals = {
            "speed_mean": np.nanmean(speed, axis=1),
            "speed_p90": np.nanpercentile(speed, 90, axis=1),
            "walk_share": np.where(n_moving > 0, (moving & (np.where(moving, speed, 0.0) < WALK_M_S)).sum(axis=1) / n_moving, np.nan),
            "still_share": np.where(n_moving > 0, (moving & (np.where(moving, speed, 0.0) < STILL_M_S)).sum(axis=1) / n_moving, np.nan),
            "spread_x": np.nanstd(x, axis=1),
            "spread_y": np.nanstd(y, axis=1),
            "n_outside": outside.sum(axis=1).astype(np.float64),
            "n_centre_circle": in_circle.sum(axis=1).astype(np.float64),
            "corner_distance_min": np.nanmin(np.where(tracked, corner_distance, np.nan), axis=1),
        }
    return list(signals), np.column_stack(list(signals.values()))


def multi_scale(names, signals, segments=None):
    """(names, (frames, features)) centred means at CENTRED_WINDOWS_S and past-minus-future trends at TREND_WINDOWS_S."""
    columns, out_names = [], []
    segments = ((0, len(signals)),) if segments is None else segments
    for seconds in CENTRED_WINDOWS_S:
        half = int(round(seconds * FPS / 2))
        column = np.full_like(signals, np.nan, dtype=np.float64)
        for start, stop in segments:
            column[start:stop] = window_mean(signals[start:stop], half, half)
        columns.append(column)
        out_names += ["c{}s_{}".format(seconds, name) for name in names]
    for seconds in TREND_WINDOWS_S:
        frames = int(round(seconds * FPS))
        column = np.full_like(signals, np.nan, dtype=np.float64)
        for start, stop in segments:
            column[start:stop] = (window_mean(signals[start:stop], frames, -1)
                                   - window_mean(signals[start:stop], -1, frames))
        columns.append(column)
        out_names += ["trend{}s_{}".format(seconds, name) for name in names]
    return out_names, np.concatenate(columns, axis=1).astype(np.float32)


def home_attacks_up(match):
    """Per half: True if the home team attacks towards x = pitch length.

    The team whose players stand nearer to x = 0 over the whole half defends that goal. The median
    over a half of the team mean x is robust to single frames and to track fragments.
    """
    x = match.players_m[:, :, 0]
    out = []
    for start, stop in match.halves:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home = np.nanmedian(np.nanmean(x[start:stop, match.slot_team == H], axis=1))
            away = np.nanmedian(np.nanmean(x[start:stop, match.slot_team == A], axis=1))
        out.append(bool(home < away))
    return out


def half_of_frame(match):
    """Per frame: index of the half; a frame between halves goes to the half whose range is nearer."""
    (_, first_stop), (second_start, _) = match.halves
    boundary = (first_stop + second_start) // 2
    return (np.arange(len(match.players_m)) >= boundary).astype(np.int8)


def team_signals(match, with_ball=True):
    """(names, (frames, signals)) per-team signals, each team measured in its own attacking direction.

    Names end in "_H" or "_A"; swap_teams exchanges the two, so a classifier can be made symmetric.
    The progress, line and ball_progress values are metres from the team's own goal line.
    """
    length, _ = match.pitch_m
    up = np.array(home_attacks_up(match))[half_of_frame(match)]
    x, y = match.players_m[:, :, 0], match.players_m[:, :, 1]
    slot_up = np.where(match.slot_team[None, :] == H, up[:, None], ~up[:, None])
    oriented_x = np.where(slot_up, x, length - x)
    stride = SPEED_STRIDE_FRAMES
    velocity_x = _centred_difference(oriented_x, stride, match.halves or None) * FPS / (2 * stride)
    speed = player_speeds(match)
    ball_x = match.ball_m[:, 0]
    ball_step = np.hypot(*_centred_difference(match.ball_m, stride, match.halves or None).T) * FPS / (2 * stride)
    nearest_opponent = _nearest_opponent_distance(match)
    ball_distance = np.hypot(x - match.ball_m[:, None, 0], y - match.ball_m[:, None, 1])
    signals = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for code, suffix in ((H, "H"), (A, "A")):
            mine = match.slot_team == code
            team_up = up if code == H else ~up
            tracked = np.isfinite(x[:, mine])
            count = tracked.sum(axis=1)
            ordered = np.sort(oriented_x[:, mine], axis=1)  # NaN sorts last
            signals["progress_" + suffix] = np.nanmean(oriented_x[:, mine], axis=1)
            signals["advance_" + suffix] = np.nanmean(velocity_x[:, mine], axis=1)
            signals["spread_" + suffix] = np.nanstd(x[:, mine], axis=1) * np.nanstd(y[:, mine], axis=1)
            signals["opponent_half_share_" + suffix] = np.where(
                count > 0, (tracked & (np.where(tracked, oriented_x[:, mine], 0.0) > length / 2)).sum(axis=1) / np.maximum(count, 1), np.nan)
            # The second-deepest player is the defensive line; the deepest one is usually the goalkeeper.
            signals["last_line_" + suffix] = np.where(count >= 2, ordered[:, 1], np.nan)
            signals["openness_" + suffix] = np.nanmedian(nearest_opponent[:, mine], axis=1)
            signals["speed_" + suffix] = np.nanmean(speed[:, mine], axis=1)
            signals["ball_distance_" + suffix] = np.nanmin(ball_distance[:, mine], axis=1)
            signals["ball_progress_" + suffix] = np.where(team_up, ball_x, length - ball_x)
        signals["ball_speed"] = ball_step
        signals["ball_seen"] = np.isfinite(ball_x).astype(np.float64)
    if not with_ball:
        signals = {name: value for name, value in signals.items() if not name.startswith(BALL_SIGNALS)}
    return list(signals), np.column_stack(list(signals.values()))


def _nearest_opponent_distance(match):
    """(frames, slots) distance in m from each player to the nearest tracked opponent; NaN if none."""
    out = np.full(match.players_m.shape[:2], np.nan)
    teams = match.slot_team
    for start in range(0, len(out), DISTANCE_CHUNK_FRAMES):
        block = match.players_m[start:start + DISTANCE_CHUNK_FRAMES]
        distance = np.sqrt(((block[:, :, None, :] - block[:, None, :, :]) ** 2).sum(axis=3))
        distance = np.where(teams[None, :, None] != teams[None, None, :], distance, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[start:start + DISTANCE_CHUNK_FRAMES] = np.nanmin(distance, axis=2)
    return out


def swap_order(names):
    """Column order that exchanges each "_H" column with its "_A" partner; other columns keep their place."""
    index = {name: i for i, name in enumerate(names)}
    return np.array([index[name[:-2] + ("_A" if name.endswith("_H") else "_H")] if name[-2:] in ("_H", "_A")
                     else index[name] for name in names])


def swap_teams(names, table):
    """The same table with the "_H" and "_A" columns exchanged, as if the teams changed roles."""
    return table[:, swap_order(names)]


def _frames_since(mask, segments=None):
    """Per frame: frames since the last True at or before it; a large value where there is none."""
    if segments is not None:
        result = np.full(len(mask), np.nan)
        for start, stop in segments:
            result[start:stop] = _frames_since(mask[start:stop])
        return result
    index = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, index, -10 ** 9))
    return index - last


def _frames_until(mask, segments=None):
    """Per frame: frames until the next True at or after it; a large value where there is none."""
    if segments is not None:
        result = np.full(len(mask), np.nan)
        for start, stop in segments:
            result[start:stop] = _frames_until(mask[start:stop])
        return result
    return _frames_since(mask[::-1])[::-1]


def ball_advance_signals(match):
    """(names, (frames, signals)) per team: ball velocity in m/s along the team's attacking direction.

    The team in possession usually moves the ball towards the opponent goal.
    """
    up = np.array(home_attacks_up(match))[half_of_frame(match)]
    stride = SPEED_STRIDE_FRAMES
    velocity_x = _centred_difference(match.ball_m[:, 0], stride, match.halves or None) * FPS / (2 * stride)
    home_advance = np.where(up, velocity_x, -velocity_x)
    return ["ball_advance_H", "ball_advance_A"], np.column_stack((home_advance, -home_advance))


def touch_signals(match):
    """(names, (frames, signals)) per team: seconds since its last and until its next touch, capped.

    A touch is a frame where the team's nearest player is within TOUCH_RADIUS_M of a seen ball. Play is
    labelled offline, so the next touch is known; during a pass it names the receiving team. Ball
    acceleration (m/s^2, stride differences) marks kicks and deflections.
    """
    x, y = match.players_m[:, :, 0], match.players_m[:, :, 1]
    ball_distance = np.hypot(x - match.ball_m[:, None, 0], y - match.ball_m[:, None, 1])
    signals = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for code, suffix in ((H, "H"), (A, "A")):
            nearest = np.nanmin(ball_distance[:, match.slot_team == code], axis=1)
            touch = np.isfinite(nearest) & (np.where(np.isfinite(nearest), nearest, np.inf) <= TOUCH_RADIUS_M)
            segments = match.halves or None
            signals["since_touch_" + suffix] = np.minimum(_frames_since(touch, segments) / FPS, TOUCH_HORIZON_S)
            signals["until_touch_" + suffix] = np.minimum(_frames_until(touch, segments) / FPS, TOUCH_HORIZON_S)
            signals["touch_" + suffix] = np.where(np.isfinite(match.ball_m[:, 0]), touch.astype(np.float64), np.nan)
    stride = SPEED_STRIDE_FRAMES
    velocity = _centred_difference(match.ball_m, stride, match.halves or None) * FPS / (2 * stride)
    acceleration = np.hypot(*_centred_difference(velocity, stride, match.halves or None).T) * FPS / (2 * stride)
    signals["ball_acceleration"] = acceleration
    if match.halves:
        inside = np.zeros(len(match.players_m), dtype=bool)
        for start, stop in match.halves:
            inside[start:stop] = True
        for name in signals:
            signals[name] = np.where(inside, signals[name], np.nan)
    return list(signals), np.column_stack(list(signals.values()))
