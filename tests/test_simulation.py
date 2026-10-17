import pytest
from airline_dp.model import solve_dp
from airline_dp.policies import accept_all_policy, accept_if_fare_at_least
from airline_dp.simulation import compare_policies, simulate_policy

def make_solution():
    return solve_dp(4, 8, 0.6, [100, 250, 500], [0.5, 0.3, 0.2])

def test_simulation_is_reproducible():
    sol = make_solution()
    policy = accept_if_fare_at_least(sol)
    a = simulate_policy(policy, sol, replications=500, seed=7)
    b = simulate_policy(policy, sol, replications=500, seed=7)
    assert a.mean_revenue == b.mean_revenue
    assert (a.revenues == b.revenues).all()

def test_optimal_policy_monte_carlo_close_to_exact_value():
    sol = make_solution()
    result = simulate_policy(accept_if_fare_at_least(sol), sol, replications=30000, seed=9)
    assert abs(result.mean_revenue - sol.expected_revenue) < 5 * max(result.standard_error, 1.0)

def test_compare_policies_uses_common_sample_size():
    sol = make_solution()
    results = compare_policies([("optimal", accept_if_fare_at_least(sol)), ("all", accept_all_policy)],
                               sol, replications=1000, seed=12)
    assert set(results) == {"optimal", "all"}
    assert results["optimal"].replications == results["all"].replications == 1000

def test_simulation_rejects_invalid_replication_count():
    with pytest.raises(ValueError):
        simulate_policy(accept_all_policy, make_solution(), replications=0)
