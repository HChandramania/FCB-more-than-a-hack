"""Changes to the tracking of a match: simulated tracking loss for training, and the fill of short player gaps.

The external matches track the ball and all 22 players almost without a gap, while the automatic
tracking of the competition match loses the ball on 41 % of graded frames and about 7.6 % of the
player-frames. Training copies with simulated loss teach the classifiers to work with gaps.
"""
import dataclasses

import numpy as np

from .data import FPS, A, H, X

LIVE_GAP_SECONDS = (2.0, 30.0)  # log-uniform gap length; a generic range, not measured on any match
PLAYER_GAP_SECONDS = (2.0, 30.0)  # log-uniform length of a lost player track; a generic range, as for the ball
GAP_PLACEMENTS = ("live", "uniform")


def add_live_ball_gaps(match, share, seed, placement="live"):
    """Copy of an external match with the ball removed on about `share` of its live frames, in blocks.

    placement "live": blocks lie inside live stretches only. placement "uniform": blocks lie anywhere
    in the labelled frames, so their position does not depend on the labels; only the count of
    removed live frames does. Positions are random but fixed by the seed.
    """
    if share <= 0:
        return match
    if placement not in GAP_PLACEMENTS:
        raise ValueError(placement)
    rng = np.random.default_rng(seed)
    alive = (match.labels == H) | (match.labels == A)
    region = alive if placement == "live" else match.labels != X
    edges = np.diff(np.concatenate(([0], region.astype(np.int8), [0])))
    stretches = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    lengths = np.array([b - a for a, b in stretches], dtype=np.float64)
    target = share * alive.sum()
    reachable = int((np.isfinite(match.ball_m[:, 0]) & alive).sum())
    if target > reachable:
        raise ValueError("share {:.3f} needs {:.0f} live frames with a ball; the match has {}".format(share, target, reachable))
    ball = match.ball_m.copy()
    removed = 0
    low, high = np.log(LIVE_GAP_SECONDS[0] * FPS), np.log(LIVE_GAP_SECONDS[1] * FPS)
    while removed < target:
        stretch = rng.choice(len(stretches), p=lengths / lengths.sum())
        start, stop = stretches[stretch]
        length = min(int(np.exp(rng.uniform(low, high))), stop - start)
        begin = start + int(rng.integers(0, stop - start - length + 1))
        removed += int((np.isfinite(ball[begin:begin + length, 0]) & alive[begin:begin + length]).sum())
        ball[begin:begin + length] = np.nan
    return dataclasses.replace(match, ball_m=ball)


def add_player_dropout(match, share, seed):
    """Copy of the match with about `share` of the tracked player-frames removed, in blocks per player."""
    if share <= 0:
        return match
    rng = np.random.default_rng(seed)
    players = match.players_m.copy()
    tracked = np.isfinite(players[:, :, 0])
    target = share * tracked.sum()
    removed = 0
    low, high = np.log(PLAYER_GAP_SECONDS[0] * FPS), np.log(PLAYER_GAP_SECONDS[1] * FPS)
    slots_with_tracks = np.flatnonzero(tracked.any(axis=0))
    while removed < target:
        slot = int(rng.choice(slots_with_tracks))
        length = int(np.exp(rng.uniform(low, high)))
        begin = int(rng.integers(0, max(1, len(players) - length)))
        removed += int(np.isfinite(players[begin:begin + length, slot, 0]).sum())
        players[begin:begin + length, slot] = np.nan
    return dataclasses.replace(match, players_m=players)


def fill_player_gaps(match, max_seconds):
    """Copy of the match where each gap of at most max_seconds inside one track slot is filled by linear interpolation.

    A slot keeps one player: on the competition match a slot that comes back after a gap has moved at
    a median 0.2-1.5 m/s. Gaps at the start or end of a slot stay empty. Half time is longer than any
    max_seconds in use, so no fill crosses it.
    """
    if max_seconds <= 0:
        return match
    players = match.players_m.copy()
    longest = int(round(max_seconds * FPS))
    frames = np.arange(len(players))
    for slot in range(players.shape[1]):
        tracked = np.isfinite(players[:, slot, 0])
        if not tracked.any():
            continue
        change = np.diff(tracked.astype(np.int8))
        gap_starts = np.flatnonzero(change == -1) + 1
        gap_stops = np.flatnonzero(change == 1) + 1
        # A slot that starts untracked has a first gap_stop without a gap_start; drop it.
        gap_stops = gap_stops[gap_stops > gap_starts[0]] if len(gap_starts) > 0 else gap_stops[:0]
        for start, stop in zip(gap_starts, gap_stops):
            if stop - start > longest:
                continue
            fraction = (frames[start:stop] - (start - 1)) / (stop - (start - 1))
            players[start:stop, slot] = players[start - 1, slot] + fraction[:, None] * (players[stop, slot] - players[start - 1, slot])
    return dataclasses.replace(match, players_m=players)
