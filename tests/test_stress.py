"""Stress/regression tests for boundary states and the Bellman logic."""

from itertools import product

import numpy as np
import pytest

from airline_dp.model import solve_dp, solve_mean_fare_dp
from airline_dp.policies import (
    accept_all_policy,
    accept_if_fare_at_least,
    fixed_threshold_policy,
)
from airline_dp.simulation import (
    compare_policies,
    paired_revenue_difference,
    simulate_policy,
)


def test_one_seat_constant_fare_has_closed_form_value_and_thresholds():
    fare, lam, horizon = 250.0, 0.3, 7
    sol = solve_dp(1, horizon, lam, [fare], [1.0])
    for t in range(horizon + 1):
        expected = fare * (1 - (1 - lam) ** t)
        assert sol.values[t, 1] == pytest.approx(expected)
        assert sol.values[t, 0] == 0
    for t in range(1, horizon + 1):
        expected_threshold = fare * (1 - (1 - lam) ** (t - 1))
        assert sol.thresholds[t, 1] == pytest.approx(expected_threshold)


def test_dp_matches_exhaustive_deterministic_markov_policy_enumeration():
    """Tiny model: enumerate every accept/reject action for each (t,x,fare)."""
    capacity, horizon = 2, 2
    lam = 0.65
    fares = (10.0, 35.0)
    probs = (0.4, 0.6)
    states = [
        (t, x, i)
        for t in range(1, horizon + 1)
        for x in range(1, capacity + 1)
        for i in range(len(fares))
    ]
    best_value = -np.inf
    for decisions in product((False, True), repeat=len(states)):
        action = dict(zip(states, decisions))
        value = np.zeros((horizon + 1, capacity + 1))
        for t in range(1, horizon + 1):
            for x in range(1, capacity + 1):
                no_arrival = value[t - 1, x]
                with_arrival = 0.0
                for i, (fare, prob) in enumerate(zip(fares, probs)):
                    if action[(t, x, i)]:
                        outcome = fare + value[t - 1, x - 1]
                    else:
                        outcome = value[t - 1, x]
                    with_arrival += prob * outcome
                value[t, x] = (1 - lam) * no_arrival + lam * with_arrival
        best_value = max(best_value, value[horizon, capacity])
    exact = solve_dp(capacity, horizon, lam, fares, probs).expected_revenue
    assert exact == pytest.approx(best_value)


def test_expected_value_is_monotone_in_arrival_probability():
    vals = [
        solve_dp(4, 9, lam, [0, 10, 50], [0.2, 0.3, 0.5]).expected_revenue
        for lam in np.linspace(0, 1, 11)
    ]
    assert np.all(np.diff(vals) >= -1e-10)


@pytest.mark.parametrize(
    "capacity,horizon,lam",
    [(0, 5, 0.5), (5, 0, 0.5), (0, 0, 1.0), (3, 4, 0.0)],
)
def test_degenerate_dp_states_are_finite_and_correct(capacity, horizon, lam):
    sol = solve_dp(capacity, horizon, lam, [0, 10, 50], [0.2, 0.3, 0.5])
    assert np.all(np.isfinite(sol.values))
    assert np.all(sol.values >= 0)
    if capacity == 0 or horizon == 0 or lam == 0:
        assert sol.expected_revenue == 0
    if capacity == 0:
        assert np.all(np.isinf(sol.thresholds))


def test_extreme_finite_probability_weights_are_normalized_without_overflow():
    sol = solve_dp(1, 1, 0.5, [10, 20], [1e308, 1e308])
    assert sol.probabilities == pytest.approx([0.5, 0.5])
    assert sol.expected_revenue == pytest.approx(7.5)


def test_numeric_overflow_is_reported_instead_of_returning_infinite_values():
    with pytest.raises(ValueError, match="overflowed"):
        solve_dp(2, 4, 1.0, [1e308], [1.0])


@pytest.mark.parametrize("bad_lambda", ["0.5", True, None, np.nan, np.inf, -0.01, 1.01])
def test_bad_arrival_probability_has_consistent_value_error(bad_lambda):
    with pytest.raises(ValueError):
        solve_dp(2, 2, bad_lambda, [10], [1])


@pytest.mark.parametrize(
    "fares,probs",
    [
        ([[10, 20]], [1, 1]), ([10], [[1]]), ([np.nan], [1]), ([np.inf], [1]),
        ([10], [np.nan]), ([10], [np.inf]), ([10], [-1]), ([10, 20], [0, 0]),
    ],
)
def test_invalid_fare_distributions_are_rejected(fares, probs):
    with pytest.raises(ValueError):
        solve_dp(2, 3, 0.5, fares, probs)


def test_zero_probability_fare_class_is_harmless():
    sol_a = solve_dp(3, 6, 0.5, [10, 20], [1, 0])
    sol_b = solve_dp(3, 6, 0.5, [10], [1])
    assert sol_a.values == pytest.approx(sol_b.values)


def test_equal_fares_and_probability_weights_are_consistent():
    a = solve_dp(3, 5, 0.4, [20, 20, 20], [0.1, 0.2, 0.7])
    b = solve_dp(3, 5, 0.4, [20], [1])
    assert a.values == pytest.approx(b.values)


def test_optimal_policy_respects_boundary_and_state_space():
    sol = solve_dp(3, 4, 0.5, [10, 20], [0.5, 0.5])
    policy = accept_if_fare_at_least(sol)
    assert policy(100, 0, 4) is False
    assert policy(100, 2, 0) is False
    with pytest.raises(ValueError):
        policy(10, -1, 2)
    with pytest.raises(ValueError):
        policy(10, 2, -1)
    with pytest.raises(ValueError):
        policy(10, 4, 2)
    with pytest.raises(ValueError):
        policy(10, 2, 5)


@pytest.mark.parametrize("fare", [-1, np.nan, np.inf, -np.inf, True])
def test_policies_reject_invalid_fares(fare):
    with pytest.raises(ValueError):
        accept_all_policy(fare, 1, 1)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, True])
def test_fixed_policy_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        fixed_threshold_policy(threshold)


@pytest.mark.parametrize(
    "replications,seed",
    [(0, 1), (-1, 1), (1.5, 1), (True, 1), (2, -1), (2, 1.5), (2, True)],
)
def test_simulation_run_settings_are_strict(replications, seed):
    sol = solve_dp(2, 2, 0.5, [10], [1])
    with pytest.raises(ValueError):
        simulate_policy(accept_all_policy, sol, replications=replications, seed=seed)


def test_comparison_rejects_duplicate_empty_names_and_noncallable_policies():
    sol = solve_dp(2, 2, 0.5, [10], [1])
    with pytest.raises(ValueError, match="duplicate"):
        compare_policies([("same", accept_all_policy), ("same", accept_all_policy)], sol)
    with pytest.raises(ValueError, match="non-empty"):
        compare_policies([(" ", accept_all_policy)], sol)
    with pytest.raises(ValueError, match="callable"):
        compare_policies([("broken", None)], sol)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        compare_policies([], sol)


def test_single_replication_standard_error_is_undefined_not_zero():
    sol = solve_dp(1, 1, 0.5, [10], [1])
    result = simulate_policy(accept_all_policy, sol, replications=1, seed=1)
    assert np.isnan(result.standard_error)


def test_paired_difference_uses_replicationwise_differences():
    sol = solve_dp(3, 5, 0.6, [10, 30], [0.5, 0.5])
    results = compare_policies(
        [("optimal", accept_if_fare_at_least(sol)), ("all", accept_all_policy)],
        sol, replications=2000, seed=77,
    )
    paired = paired_revenue_difference(results["optimal"], results["all"], "optimal", "all")
    raw_difference = results["optimal"].revenues - results["all"].revenues
    assert paired.mean_difference == pytest.approx(raw_difference.mean())
    assert paired.standard_error == pytest.approx(
        raw_difference.std(ddof=1) / np.sqrt(len(raw_difference))
    )
    assert paired.ci95_low <= paired.mean_difference <= paired.ci95_high
    assert paired.replications == 2000


def test_paired_difference_requires_compatible_samples():
    sol = solve_dp(2, 2, 0.5, [10], [1])
    a = simulate_policy(accept_all_policy, sol, replications=20, seed=1)
    b = simulate_policy(accept_all_policy, sol, replications=21, seed=1)
    with pytest.raises(ValueError, match="equal lengths"):
        paired_revenue_difference(a, b)


def test_exact_dp_policy_monte_carlo_agrees_over_many_seeds():
    sol = solve_dp(5, 12, 0.7, [10, 35, 90], [0.4, 0.35, 0.25])
    policy = accept_if_fare_at_least(sol)
    for seed in (0, 17, 1234):
        estimate = simulate_policy(policy, sol, replications=20_000, seed=seed)
        assert abs(estimate.mean_revenue - sol.expected_revenue) <= 5 * estimate.standard_error


def test_mean_fare_policy_solution_has_matching_state_dimensions():
    full = solve_dp(4, 8, 0.55, [5, 25, 80], [0.2, 0.3, 0.5])
    ce = solve_mean_fare_dp(4, 8, 0.55, [5, 25, 80], [0.2, 0.3, 0.5])
    assert ce.capacity == full.capacity
    assert ce.horizon == full.horizon
    assert ce.fares[0] == pytest.approx(np.dot(full.fares, full.probabilities))
    assert np.isfinite(ce.expected_revenue)
