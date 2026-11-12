"""The player-gap fill: tracked values stay, and only interior gaps up to the limit are filled, linearly."""
import numpy as np

from possession import augment
from possession.data import FPS, A, H, X, Match


def match_with_gaps():
    players = np.zeros((40 * FPS, 2, 2))
    players[:, 0, 0] = np.arange(len(players), dtype=np.float64)
    players[:, 1, 0] = 5.0
    players[:2 * FPS, 0] = np.nan  # leading gap: stays
    players[5 * FPS:8 * FPS, 0] = np.nan  # 3 s interior gap: filled
    players[10 * FPS:25 * FPS, 0] = np.nan  # 15 s interior gap: stays
    players[38 * FPS:, 1] = np.nan  # trailing gap: stays
    return Match("test", players, np.full((len(players), 2), np.nan), np.array([H, A], dtype=np.int8),
                 np.full(len(players), X, dtype=np.int8), (105.0, 68.0), ((0, len(players)),))


def test_fill_player_gaps():
    match = match_with_gaps()
    filled = augment.fill_player_gaps(match, 10.0).players_m
    tracked = np.isfinite(match.players_m)
    assert np.array_equal(filled[tracked], match.players_m[tracked])
    assert np.allclose(filled[5 * FPS:8 * FPS, 0, 0], np.arange(5 * FPS, 8 * FPS))  # the slot moves 1 m per frame
    assert np.isnan(filled[:2 * FPS, 0]).all()
    assert np.isnan(filled[10 * FPS:25 * FPS, 0]).all()
    assert np.isnan(filled[38 * FPS:, 1]).all()


def test_no_fill_without_a_limit():
    match = match_with_gaps()
    assert augment.fill_player_gaps(match, 0.0) is match
