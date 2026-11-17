"""Synthetic regressions for half-local decoding; no match assets or shipped models."""
import dataclasses

import numpy as np
import pytest

from possession import data, decoding, pipeline


def decode(home, halves, dead=None):
    home = np.asarray(home, dtype=np.float64)
    dead = np.zeros(len(home)) if dead is None else np.asarray(dead, dtype=np.float64)
    return pipeline._decode_halves(dead, home, halves, len(home))


def previous_global_decode(home, dead=None):
    home = np.asarray(home, dtype=np.float64)
    dead = np.zeros(len(home)) if dead is None else np.asarray(dead, dtype=np.float64)
    states = decoding.possession_states(dead, home, pipeline.TEAM_SWITCH_COST,
                                       pipeline.DEAD_SWITCH_COST)
    return data.apply_control_rule(np.array([data.H, data.A, data.D], dtype=np.int8)[states])


@pytest.mark.parametrize("direction", ["second_to_first", "first_to_second"])
def test_old_global_coupling_removed(direction):
    halves = ((0, 1), (1001, 1002))
    if direction == "second_to_first":
        a, b, index = np.r_[0.6, np.full(1000, 0.5), 0.999], np.r_[0.6, np.full(1000, 0.5), 0.001], 0
    else:
        a, b, index = np.r_[0.999, np.full(1000, 0.5), 0.6], np.r_[0.001, np.full(1000, 0.5), 0.6], -1
    assert previous_global_decode(a)[index] != previous_global_decode(b)[index]
    assert decode(a, halves)[index] == decode(b, halves)[index]


def test_identical_to_previous_algorithm_run_separately_on_each_half():
    rng = np.random.default_rng(42)
    home, dead = rng.uniform(0.01, 0.99, (2, 320))
    halves = ((7, 137), (160, 311))
    actual = decode(home, halves, dead)
    expected = np.full(320, data.X, dtype=np.int8)
    for start, stop in halves:
        expected[start:stop] = previous_global_decode(home[start:stop], dead[start:stop])
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (320,) and actual.dtype == np.int8
    assert set(actual.tolist()) <= {data.H, data.A, data.D, data.X}


def test_single_full_span_is_identical_to_old_global_behavior():
    home = np.r_[np.full(60, 0.999999), np.full(55, 0.000001)]
    np.testing.assert_array_equal(decode(home, ((0, len(home)),)), previous_global_decode(home))


def test_outside_probabilities_do_not_affect_halves_and_output_is_x():
    home = np.full(250, 0.5)
    dead = np.zeros(250)
    halves = ((20, 80), (130, 220))
    baseline = decode(home, halves, dead)
    outside = np.ones(250, dtype=bool)
    for start, stop in halves:
        outside[start:stop] = False
    home[outside], dead[outside] = 0.999999, 0.999999
    changed = decode(home, halves, dead)
    np.testing.assert_array_equal(baseline, changed)
    assert np.all(changed[outside] == data.X)


def test_adjacent_halves_reset_control_without_an_inserted_gap():
    home = np.r_[np.full(60, 0.999999), np.full(25, 0.000001)]
    assert np.all(previous_global_decode(home)[60:] == data.H)
    assert np.all(decode(home, ((0, 60), (60, 85)))[60:] == data.A)


@pytest.mark.parametrize("length", [49, 50])
def test_control_threshold_is_unchanged_within_each_half(length):
    half = np.r_[np.full(60, 0.999999), np.full(length, 0.000001), np.full(60, 0.999999)]
    n = len(half)
    result = decode(np.r_[half, half], ((0, n), (n, 2 * n)))
    for start in (0, n):
        assert np.all(result[start + 60:start + 60 + length] == (data.H if length == 49 else data.A))


@pytest.mark.parametrize("halves", [(), [], ((0, 0), (4, 4))])
def test_no_nonempty_spans_returns_all_x(halves):
    np.testing.assert_array_equal(decode(np.full(4, 0.9), halves), np.full(4, data.X))


def test_zero_frame_match_and_numpy_integer_bounds():
    assert decode([], ()).shape == (0,)
    assert decode([0.9], ((np.int64(0), np.int64(1)),)).tolist() == [data.H]


@pytest.mark.parametrize("halves", [
    None, "01", ((0,),), ((0, 1, 2),), (None,), ((0.0, 1),), ((False, 1),),
    ((0, True),), (("0", 1),), ((-1, 2),), ((0, 5),), ((3, 2),),
    ((0, 3), (2, 4)), ((2, 4), (0, 1)), ((0, 3), (1, 1)),
])
def test_invalid_spans_raise_value_error(halves):
    with pytest.raises(ValueError):
        decode(np.full(4, 0.9), halves)


@pytest.mark.parametrize("dead,home", [
    (np.zeros(3), np.zeros(4)), (np.zeros(4), np.zeros(3)),
    (np.zeros((4, 1)), np.zeros(4)), (np.zeros(4), np.zeros((4, 1))),
])
def test_probability_shape_errors_are_explicit(dead, home):
    with pytest.raises(ValueError, match="one value per match frame"):
        pipeline._decode_halves(dead, home, ((0, 4),), 4)


def synthetic_match(halves):
    return data.Match("synthetic", np.zeros((4, 2, 2)), np.zeros((4, 2)),
                      np.array([data.H, data.A]), np.full(4, data.X, dtype=np.int8),
                      (105.0, 68.0), halves)


def test_predict_rejects_invalid_spans_before_preprocessing_or_models(monkeypatch):
    def unexpected(*args):
        pytest.fail("invalid spans reached probability generation or preprocessing")
    monkeypatch.setattr(pipeline.augment, "fill_player_gaps", unexpected)
    monkeypatch.setattr(pipeline, "dead_probability", unexpected)
    monkeypatch.setattr(pipeline, "home_probability", unexpected)
    with pytest.raises(ValueError):
        pipeline.predict(synthetic_match(((0, 3), (2, 4))))


def test_predict_no_spans_skips_probability_generation(monkeypatch):
    def unexpected(*args):
        pytest.fail("empty spans reached model calls")
    monkeypatch.setattr(pipeline, "dead_probability", unexpected)
    monkeypatch.setattr(pipeline, "home_probability", unexpected)
    for halves in ((), ((0, 0), (4, 4))):
        assert pipeline.predict(synthetic_match(halves)).tolist() == [data.X] * 4
    empty = synthetic_match(())
    empty = dataclasses.replace(empty, labels=np.empty(0, dtype=np.int8))
    assert pipeline.predict(empty).shape == (0,)


def test_probability_generation_still_runs_once_on_full_timeline(monkeypatch):
    calls = []
    def probability(match, name, value):
        calls.append((name, match.players_m.shape, match.halves))
        return np.full(4, value)
    monkeypatch.setattr(pipeline, "dead_probability", lambda m: probability(m, "dead", 0.0))
    monkeypatch.setattr(pipeline, "home_probability", lambda m: probability(m, "home", 0.9))
    halves = ((0, 1), (3, 4))
    result = pipeline.predict(synthetic_match(halves))
    assert calls == [("dead", (4, 2, 2), halves), ("home", (4, 2, 2), halves)]
    assert result.tolist() == [data.H, data.X, data.X, data.H]
