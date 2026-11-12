"""The control rule of CHALLENGE.md on hand-made label sequences."""
import numpy as np
import pytest

from possession.data import FPS, A, D, H, X, apply_control_rule


def sequence(*parts):
    """Codes from (code, seconds) pairs."""
    return np.concatenate([np.full(int(round(seconds * FPS)), code, dtype=np.int8) for code, seconds in parts])


@pytest.mark.parametrize("given, expected", [
    # a short touch by the other team stays with the owner
    (sequence((H, 5), (A, 1), (H, 5)), sequence((H, 11))),
    # a run of at least 2 s transfers possession
    (sequence((H, 5), (A, 2), (H, 5)), sequence((H, 5), (A, 2), (H, 5))),
    # contested flicker stays with the last team, then a long run transfers
    (sequence((H, 4), (A, 1), (H, 0.4), (A, 6)), sequence((H, 5.4), (A, 6))),
    # the restarting team owns the ball even for a short first run
    (sequence((D, 3), (A, 1), (H, 6)), sequence((D, 3), (A, 1), (H, 6))),
    # a dead ball resets the owner
    (sequence((H, 5), (D, 3), (A, 1), (H, 0.4), (X, 1)), sequence((H, 5), (D, 3), (A, 1.4), (X, 1))),
    # a short run before a stoppage stays with the owner
    (sequence((H, 5), (A, 1), (D, 3)), sequence((H, 6), (D, 3))),
])
def test_control_rule(given, expected):
    assert np.array_equal(apply_control_rule(given, 2.0), expected)
