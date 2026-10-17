import numpy as np
import pytest
from airline_dp.model import solve_dp, solve_mean_fare_dp

def test_boundary_conditions():
    sol = solve_dp(3, 4, 0.5, [10, 20], [0.5, 0.5])
    assert np.all(sol.values[0, :] == 0)
    assert np.all(sol.values[:, 0] == 0)

def test_zero_arrival_probability_gives_zero_value():
    assert solve_dp(4, 8, 0.0, [10, 20], [0.5, 0.5]).expected_revenue == 0

def test_zero_capacity_gives_zero_value():
    sol = solve_dp(0, 8, 0.8, [10, 20], [0.5, 0.5])
    assert sol.values.shape == (9, 1)
    assert sol.expected_revenue == 0

def test_one_period_value():
    fares, probs = [10, 20, 40], [0.2, 0.3, 0.5]
    sol = solve_dp(1, 1, 0.4, fares, probs)
    assert sol.values[1, 1] == pytest.approx(0.4 * np.dot(fares, probs))
    assert sol.thresholds[1, 1] == 0

def test_value_monotonicity():
    sol = solve_dp(8, 12, 0.55, [10, 20, 40], [0.5, 0.3, 0.2])
    assert np.all(np.diff(sol.values, axis=0) >= -1e-10)
    assert np.all(np.diff(sol.values, axis=1) >= -1e-10)

def test_marginal_values_nonincreasing_in_capacity():
    sol = solve_dp(10, 15, 0.6, [10, 30, 70], [0.5, 0.3, 0.2])
    for t in range(1, sol.horizon + 1):
        assert np.all(np.diff(sol.thresholds[t, 1:]) <= 1e-9)

def test_probabilities_are_normalized():
    sol = solve_dp(2, 2, 0.5, [10, 20], [2, 2])
    assert sol.probabilities.sum() == pytest.approx(1.0)

@pytest.mark.parametrize("args", [
    (-1, 2, 0.5, [10], [1]), (2, -1, 0.5, [10], [1]),
    (2, 2, 1.1, [10], [1]), (2, 2, 0.5, [], []),
    (2, 2, 0.5, [10, 20], [1]), (2, 2, 0.5, [10], [0]),
    (2, 2, 0.5, [-1], [1]),
])
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        solve_dp(*args)

def test_mean_fare_dp_uses_weighted_mean():
    sol = solve_mean_fare_dp(2, 3, 0.5, [10, 20], [0.25, 0.75])
    assert sol.fares[0] == pytest.approx(17.5)
    assert sol.probabilities.tolist() == [1.0]
