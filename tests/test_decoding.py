"""The Viterbi decoder against an exhaustive search over all paths."""
import itertools

import numpy as np

from possession import decoding


def path_cost(probability, switch_cost, path):
    cost = -np.log(np.clip(probability[np.arange(len(path)), path], decoding.PROBABILITY_FLOOR, 1.0)).sum()
    return cost + sum(switch_cost[a, b] for a, b in zip(path[:-1], path[1:]))


def test_viterbi_equals_exhaustive_search():
    rng = np.random.default_rng(0)
    for _ in range(200):
        states = int(rng.integers(2, 4))
        length = int(rng.integers(1, 8))
        probability = rng.dirichlet(np.ones(states), size=length)
        switch_cost = rng.uniform(0, 3, size=(states, states))
        np.fill_diagonal(switch_cost, 0.0)
        best = min(path_cost(probability, switch_cost, np.array(p)) for p in itertools.product(range(states), repeat=length))
        assert np.isclose(path_cost(probability, switch_cost, decoding.viterbi(probability, switch_cost)), best)


def test_switch_cost_removes_a_short_dip():
    p_home = np.array([0.9, 0.9, 0.4, 0.9, 0.9])
    p_dead = np.zeros(5)
    assert (decoding.possession_states(p_dead, p_home, 3.0, 20.0) == 0).all()
    assert decoding.possession_states(p_dead, p_home, 0.0, 20.0)[2] == 1
