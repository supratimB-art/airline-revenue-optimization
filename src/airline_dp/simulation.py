"""Monte Carlo policy evaluation with reproducible, shared demand paths."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .model import DpSolution
from .policies import Policy
FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class SimulationResult:
    """Monte Carlo summary for a single policy."""
    mean_revenue: float
    standard_error: float
    mean_seats_sold: float
    fill_rate: float
    replications: int
    revenues: FloatArray

def _generate_paths(horizon, arrival_probability, fares, probabilities, replications, seed):
    if replications < 1:
        raise ValueError("replications must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    arrivals = rng.random((replications, horizon)) < arrival_probability
    fare_indices = rng.choice(len(fares), size=(replications, horizon), p=probabilities)
    return arrivals, fares[fare_indices]

def _evaluate_paths(policy: Policy, capacity, horizon, arrivals, offered_fares):
    replications = arrivals.shape[0]
    revenue = np.zeros(replications, dtype=float)
    seats_sold = np.zeros(replications, dtype=float)
    for r in range(replications):
        x = capacity
        for period_index in range(horizon):
            if x == 0:
                break
            if not arrivals[r, period_index]:
                continue
            t = horizon - period_index
            fare = float(offered_fares[r, period_index])
            if policy(fare, x, t):
                revenue[r] += fare
                seats_sold[r] += 1
                x -= 1
    return revenue, seats_sold

def _summarize(revenues, seats_sold, capacity):
    n = len(revenues)
    standard_error = float(revenues.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    fill_rate = float(seats_sold.mean() / capacity) if capacity > 0 else 0.0
    return SimulationResult(float(revenues.mean()), standard_error, float(seats_sold.mean()),
                            fill_rate, n, revenues)

def simulate_policy(policy: Policy, solution: DpSolution, replications: int = 10_000, seed: int = 12345) -> SimulationResult:
    """Evaluate a policy under the fare distribution stored in solution."""
    arrivals, offered_fares = _generate_paths(solution.horizon, solution.arrival_probability,
                                               solution.fares, solution.probabilities, replications, seed)
    revenue, seats_sold = _evaluate_paths(policy, solution.capacity, solution.horizon, arrivals, offered_fares)
    return _summarize(revenue, seats_sold, solution.capacity)

def compare_policies(policies: Sequence[tuple[str, Policy]], solution: DpSolution,
                     replications: int = 10_000, seed: int = 12345) -> dict[str, SimulationResult]:
    """Evaluate policies on common random numbers to reduce comparison noise."""
    arrivals, offered_fares = _generate_paths(solution.horizon, solution.arrival_probability,
                                               solution.fares, solution.probabilities, replications, seed)
    results = {}
    for name, policy in policies:
        revenue, seats_sold = _evaluate_paths(policy, solution.capacity, solution.horizon, arrivals, offered_fares)
        results[name] = _summarize(revenue, seats_sold, solution.capacity)
    return results
