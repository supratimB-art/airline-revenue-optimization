"""Airline seat inventory control via dynamic programming."""

from .model import DpSolution, solve_dp, solve_mean_fare_dp
from .policies import accept_all_policy, accept_if_fare_at_least, fixed_threshold_policy
from .simulation import (
    PairedDifference,
    SimulationResult,
    compare_policies,
    paired_revenue_difference,
    simulate_policy,
)

__all__ = [
    "DpSolution",
    "PairedDifference",
    "SimulationResult",
    "accept_all_policy",
    "accept_if_fare_at_least",
    "compare_policies",
    "fixed_threshold_policy",
    "paired_revenue_difference",
    "simulate_policy",
    "solve_dp",
    "solve_mean_fare_dp",
]
