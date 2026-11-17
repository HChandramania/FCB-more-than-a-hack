"""Synthetic-only boundary audit. No data loaders, shipped models, or CSV access.

Strict xfails state desired half independence where current code lacks a boundary reset.
Passing evidence tests lock down the counterexamples without modifying production behavior.
"""
import dataclasses

import numpy as np
import pytest
import torch

from possession import augment, data, decoding, features, pipeline, tcn


def synthetic_match(gap=1000, half_length=50, second_ball=10.0):
    n = 2 * half_length + gap
    start = half_length + gap
    players = np.full((n, 2, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for a, b in ((0, half_length), (start, n)):
        players[a:b, 0] = (0.0, 0.0)
        players[a:b, 1] = (20.0, 0.0)
        ball[a:b] = (10.0, 0.0)
    ball[start:] = (second_ball, 0.0)
    return data.Match("synthetic", players, ball, np.array([data.H, data.A]),
                      np.full(n, data.X, dtype=np.int8), (105.0, 68.0),
                      ((0, half_length), (start, n)))


def decoded(first, second, gap=1000):
    home = np.r_[first, np.full(gap, 0.5), second]
    return decoding.possession_states(np.zeros(len(home)), home,
                                     pipeline.TEAM_SWITCH_COST, pipeline.DEAD_SWITCH_COST)


@pytest.mark.parametrize("direction", ["second_to_first", "first_to_second"])
def test_viterbi_counterexample(direction):
    if direction == "second_to_first":
        before, after = decoded(0.6, 0.999), decoded(0.6, 0.001)
        assert (before[0], after[0]) == (0, 1)
    else:
        before, after = decoded(0.999, 0.6), decoded(0.001, 0.6)
        assert (before[-1], after[-1]) == (0, 1)


@pytest.mark.parametrize("direction", ["second_to_first", "first_to_second"])
def test_pipeline_half_predictions_are_independent(monkeypatch, direction):
    match = synthetic_match(half_length=1)
    monkeypatch.setattr(pipeline, "dead_probability", lambda m: np.zeros(len(m.labels)))

    def run(first, second):
        monkeypatch.setattr(pipeline, "home_probability",
                            lambda m: np.r_[first, np.full(1000, 0.5), second])
        return pipeline.predict(match)

    if direction == "second_to_first":
        assert run(0.6, 0.999)[0] == run(0.6, 0.001)[0]
    else:
        assert run(0.999, 0.6)[-1] == run(0.001, 0.6)[-1]


def test_separately_decoded_halves_are_independent():
    def alone(p):
        return decoding.possession_states(np.zeros(1), np.array([p]), 2.5, 20.0)
    assert alone(0.6).tolist() == [0]
    assert alone(0.999).tolist() == [0]
    assert alone(0.001).tolist() == [1]


def test_raw_windows_respect_sufficiently_long_nan_separator():
    left = np.r_[np.ones(50), np.full(1000, np.nan), np.ones(50)]
    right = left.copy()
    right[-50:] = 100.0
    _, a = features.multi_scale(["synthetic"], left[:, None])
    _, b = features.multi_scale(["synthetic"], right[:, None])
    np.testing.assert_allclose(a[:50], b[:50], equal_nan=True)


def test_short_nan_interval_does_not_reset_window_mean():
    result = features.window_mean(np.array([1.0, np.nan, 9.0]), 2, 2)
    assert result[0] == 5.0


def touch_window(match):
    names, values = features.multi_scale(*features.touch_signals(data.only_halves(match)))
    return values[49, names.index("c40.0s_until_touch_H")]


def test_touch_window_resets_at_external_separator():
    assert touch_window(synthetic_match(second_ball=0.0)) == touch_window(
        synthetic_match(second_ball=10.0))


def test_touch_window_is_independent_of_second_half():
    assert touch_window(synthetic_match(second_ball=0.0)) == touch_window(
        synthetic_match(second_ball=10.0))


def synthetic_network():
    # Deterministic hand-set test weights, not shipped model artifacts or fitted weights.
    net = tcn.TCN(2, channels=1).eval()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.inputs.weight[0, 0, 0] = 1.0
        net.output.weight.fill_(1.0)
        for block in net.blocks:
            block.dilated.weight.fill_(1.0)
            block.mix.weight.fill_(1.0)
    return net


@pytest.mark.parametrize("direction", ["second_to_first", "first_to_second"])
def test_tcn_receptive_field_counterexample(direction):
    # 200 steps = the external separator's 1000 frames; radius is 255 steps.
    signal = torch.zeros((1, 2, 600))
    source, target = (400, 199) if direction == "second_to_first" else (199, 400)
    net = synthetic_network()
    with torch.no_grad():
        before = net(signal)[0, target].item()
        signal[0, 0, source] = 1.0
        after = net(signal)[0, target].item()
    assert before == 0.0 and after > 0.0


def test_tcn_does_not_reach_across_separator_longer_than_its_radius():
    signal = torch.zeros((1, 2, 1000))
    net = synthetic_network()
    with torch.no_grad():
        before = net(signal)[0, 199].item()
        signal[0, 0, 800] = 1.0
        after = net(signal)[0, 199].item()
    assert before == after


def test_interpolation_preserves_external_long_separator():
    match = synthetic_match()
    filled = augment.fill_player_gaps(match, 10.0)
    assert np.isnan(filled.players_m[50:1050]).all()


def test_interpolation_does_not_bridge_short_half_boundary():
    match = synthetic_match(gap=2, half_length=3)
    players = match.players_m.copy()
    players[2:5, 0] = np.nan
    other = players.copy()
    other[5:, 0] = (40.0, 0.0)
    a = augment.fill_player_gaps(dataclasses.replace(match, players_m=players), 10.0)
    b = augment.fill_player_gaps(dataclasses.replace(match, players_m=other), 10.0)
    np.testing.assert_equal(a.players_m[2, 0], b.players_m[2, 0])


def test_interpolation_before_masking_uses_ungraded_anchor():
    match = synthetic_match(half_length=3)
    players = match.players_m.copy()
    players[2, 0] = np.nan
    players[4, 0] = (8.0, 0.0)  # Synthetic excluded-frame observation.
    result = data.only_halves(augment.fill_player_gaps(
        dataclasses.replace(match, players_m=players), 10.0))
    assert np.isnan(result.players_m[2, 0]).all()
    assert np.isnan(result.players_m[4, 0]).all()


def test_five_frame_groups_do_not_cross_half_boundaries():
    values = np.arange(10.0)[:, None]
    grouped = tcn.to_steps(values, ((0, 3), (5, 10)))
    np.testing.assert_array_equal(grouped[:, 0], [7.0])


@pytest.mark.parametrize("direction", ["second_to_first", "first_to_second"])
def test_segmented_tcn_independence_with_short_separator_and_partial_halves(direction):
    net = synthetic_network()
    saved = {"state": net.state_dict(), "names": ["synthetic"], "mean": np.zeros(1),
             "std": np.ones(1), "channels": 1}
    segments = ((0, 23), (26, 53))
    baseline = np.zeros((9, 1), dtype=np.float32)
    changed = baseline.copy()
    first_slice, second_slice = slice(0, 20), slice(20, None)
    if direction == "second_to_first":
        changed[4:, 0] = 1.0
    else:
        changed[:4, 0] = 1.0
    before = tcn.probability(saved, ["synthetic"], baseline, "tcn_dead", segments)
    after = tcn.probability(saved, ["synthetic"], changed, "tcn_dead", segments)
    if direction == "second_to_first":
        np.testing.assert_array_equal(before[first_slice], after[first_slice])
        assert not np.array_equal(before[second_slice], after[second_slice])
    else:
        np.testing.assert_array_equal(before[second_slice], after[second_slice])
        assert not np.array_equal(before[first_slice], after[first_slice])


def codes(*runs):
    return np.concatenate([np.full(n, label, dtype=np.int8) for label, n in runs])


@pytest.mark.parametrize("length", [1, 49, 50])
def test_control_threshold(length):
    result = data.apply_control_rule(codes((data.H, 50), (data.A, length), (data.H, 50)))
    assert np.all(result[50:50 + length] == (data.H if length < 50 else data.A))


def test_control_keeps_short_initial_owner_and_resets_on_dead_or_x():
    for separator in (data.D, data.X):
        before = codes((data.H, 50), (separator, 1), (data.A, 1))
        np.testing.assert_array_equal(data.apply_control_rule(before), before)
    assert data.apply_control_rule(np.array([data.A], dtype=np.int8)).tolist() == [data.A]


def test_control_rule_is_causal_with_respect_to_second_half_when_x_separated():
    a = data.apply_control_rule(codes((data.H, 50), (data.X, 1000), (data.A, 25)))
    b = data.apply_control_rule(codes((data.H, 50), (data.X, 1000), (data.H, 25)))
    np.testing.assert_array_equal(a[:50], b[:50])
    assert np.all(a[-25:] == data.A)


def test_control_without_x_separator_inherits_previous_half_owner():
    merged = data.apply_control_rule(codes((data.H, 1050), (data.A, 25)))
    separated = data.apply_control_rule(codes((data.H, 50), (data.X, 1000), (data.A, 25)))
    assert np.all(merged[-25:] == data.H)
    assert np.all(separated[-25:] == data.A)


def test_pipeline_control_resets_at_half_boundary(monkeypatch):
    match = synthetic_match(half_length=25)
    monkeypatch.setattr(pipeline, "dead_probability", lambda m: np.zeros(len(m.labels)))
    monkeypatch.setattr(pipeline, "home_probability",
                        lambda m: np.r_[np.ones(1025), np.zeros(25)])
    result = pipeline.predict(match)
    assert np.all(result[-25:] == data.A)
    assert np.all(result[25:1025] == data.X)
    assert np.all(data.apply_control_rule(np.full(25, data.A)) == data.A)


def test_control_can_change_earlier_frames_when_a_run_reaches_threshold():
    short = data.apply_control_rule(codes((data.H, 50), (data.A, 49)))
    long = data.apply_control_rule(codes((data.H, 50), (data.A, 50)))
    assert np.all(short[50:] == data.H)
    assert np.all(long[50:99] == data.A)


def test_reference_x_splits_official_possession_spells():
    reference = np.array([data.H, data.X, data.H], dtype=np.int8)
    assert len(data.metrics.possession_spells(reference, data.metrics.valid_runs(reference))) == 2
    continuous = np.array([data.H, data.H, data.H], dtype=np.int8)
    assert len(data.metrics.possession_spells(continuous, data.metrics.valid_runs(continuous))) == 1


def test_control_processed_references_reward_same_postprocessing():
    raw = codes((data.H, 50), (data.A, 25), (data.H, 50))
    processed = data.apply_control_rule(raw)
    matching = data.metrics.evaluate(processed, processed)
    unprocessed = data.metrics.evaluate(processed, raw)
    assert matching["spell_f1_mean"] == 1.0
    assert unprocessed["spell_f1_mean"] < matching["spell_f1_mean"]


def test_empty_and_missing_sequences_are_not_implicitly_live():
    assert data.apply_control_rule(np.empty(0, dtype=np.int8)).size == 0
    unknown = np.full(20, data.X, dtype=np.int8)
    np.testing.assert_array_equal(data.apply_control_rule(unknown), unknown)
    assert data.metrics.possession_spells(unknown, data.metrics.valid_runs(unknown)) == []
