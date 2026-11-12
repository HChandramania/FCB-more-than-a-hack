"""Segment decoding of per-frame class probabilities: fewer, longer runs than a per-frame argmax."""
import numba
import numpy as np

PROBABILITY_FLOOR = 1e-6  # keeps log(0) out of the costs


@numba.njit(cache=True)
def _viterbi(cost, switch_cost):
    n, states = cost.shape
    total = cost[0].copy()
    back = np.zeros((n, states), dtype=np.int8)
    for t in range(1, n):
        new = np.empty(states)
        for state in range(states):
            best, best_from = np.inf, 0
            for previous in range(states):
                value = total[previous] + switch_cost[previous, state]
                if value < best:
                    best, best_from = value, previous
            new[state] = best + cost[t, state]
            back[t, state] = best_from
        total = new
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmin(total)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi(probability, switch_cost):
    """Most probable state path: per-frame cost -log p plus switch_cost[i, j] (nats) for each change i -> j.

    probability is (frames, states); a frame with NaN probability costs the same in every state.
    """
    probability = np.asarray(probability, dtype=np.float64)
    if len(probability) == 0:
        return np.empty(0, dtype=np.int8)
    cost = -np.log(np.clip(probability, PROBABILITY_FLOOR, 1.0))
    cost[~np.isfinite(cost)] = 0.0
    return _viterbi(cost, np.asarray(switch_cost, dtype=np.float64))


def possession_states(p_dead, p_home, team_switch_cost, dead_switch_cost):
    """Index path over (home, away, dead) of a 3-state Viterbi on P(dead) and P(home | live).

    Frame probabilities: home (1 - P(dead)) P(home), away (1 - P(dead)) (1 - P(home)), dead P(dead).
    A change between the teams costs team_switch_cost; a change between live and dead costs dead_switch_cost.
    """
    probability = np.column_stack(((1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead))
    switch = np.array([[0.0, team_switch_cost, dead_switch_cost],
                       [team_switch_cost, 0.0, dead_switch_cost],
                       [dead_switch_cost, dead_switch_cost, 0.0]])
    return viterbi(probability, switch)
