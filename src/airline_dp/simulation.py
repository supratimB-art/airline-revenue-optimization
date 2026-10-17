"""Monte Carlo policy evaluation with reproducible, shared demand paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .model import DpSolution
from .policies import Policy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    """Monte Carlo summary for a single policy.

    revenues stores one outcome per replication. Standard error is NaN for a
    single replication because sampling variability cannot be estimated.
    """

    mean_revenue: float
    standard_error: float
    mean_seats_sold: float
    fill_rate: float
    replications: int
    revenues: FloatArray


@dataclass(frozen=True)
class PairedDifference:
    """Monte Carlo estimate of candidate minus reference revenue on shared paths."""

    candidate_name: str
    reference_name: str
    mean_difference: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    replications: int


def _validate_run_settings(replications: int, seed: int) -> tuple[int, int]:
    if (isinstance(replications, (bool, np.bool_))
            or not isinstance(replications, Integral) or replications < 1):
        raise ValueError("replications must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(replications), int(seed)


def _generate_paths(
    horizon: int,
    arrival_probability: float,
    fares: FloatArray,
    probabilities: FloatArray,
    replications: int,
    seed: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    replications, seed = _validate_run_settings(replications, seed)
    rng = np.random.default_rng(seed)
    arrivals = rng.random((replications, horizon)) < arrival_probability
    fare_indices = rng.choice(len(fares), size=(replications, horizon), p=probabilities)
    return arrivals, fares[fare_indices]


def _evaluate_paths(
    policy: Policy,
    capacity: int,
    horizon: int,
    arrivals: NDArray[np.bool_],
    offered_fares: NDArray[np.float64],
) -> tuple[FloatArray, FloatArray]:
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
    if not np.all(np.isfinite(revenue)):
        raise ValueError("simulated revenue overflowed floating-point range")
    return revenue, seats_sold


def _summarize(revenues: FloatArray, seats_sold: FloatArray, capacity: int) -> SimulationResult:
    n = int(len(revenues))
    standard_error = float(revenues.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    fill_rate = float(seats_sold.mean() / capacity) if capacity > 0 else 0.0
    return SimulationResult(
        mean_revenue=float(revenues.mean()),
        standard_error=standard_error,
        mean_seats_sold=float(seats_sold.mean()),
        fill_rate=fill_rate,
        replications=n,
        revenues=revenues,
    )


def simulate_policy(
    policy: Policy,
    solution: DpSolution,
    replications: int = 10_000,
    seed: int = 12345,
) -> SimulationResult:
    """Evaluate one policy under the fare distribution stored in solution."""
    if not callable(policy):
        raise ValueError("policy must be callable")
    arrivals, offered_fares = _generate_paths(
        solution.horizon, solution.arrival_probability, solution.fares,
        solution.probabilities, replications, seed
    )
    revenue, seats_sold = _evaluate_paths(
        policy, solution.capacity, solution.horizon, arrivals, offered_fares
    )
    return _summarize(revenue, seats_sold, solution.capacity)


def compare_policies(
    policies: Sequence[tuple[str, Policy]],
    solution: DpSolution,
    replications: int = 10_000,
    seed: int = 12345,
) -> dict[str, SimulationResult]:
    """Evaluate named policies on common random numbers.

    Names must be unique and non-empty so results cannot be silently overwritten.
    The returned revenue vectors share the same generated demand paths; use
    paired_revenue_difference when estimating differences between them.
    """
    try:
        named_policies = tuple(policies)
    except TypeError as exc:
        raise ValueError("policies must be a sequence of (name, callable) pairs") from exc
    if not named_policies:
        raise ValueError("at least one policy is required")

    names: set[str] = set()
    validated: list[tuple[str, Policy]] = []
    for item in named_policies:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each policy entry must be a (name, callable) pair")
        name, policy = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError("policy names must be non-empty strings")
        if name in names:
            raise ValueError(f"duplicate policy name: {name!r}")
        if not callable(policy):
            raise ValueError(f"policy {name!r} must be callable")
        names.add(name)
        validated.append((name, policy))

    arrivals, offered_fares = _generate_paths(
        solution.horizon, solution.arrival_probability, solution.fares,
        solution.probabilities, replications, seed
    )
    results: dict[str, SimulationResult] = {}
    for name, policy in validated:
        revenue, seats_sold = _evaluate_paths(
            policy, solution.capacity, solution.horizon, arrivals, offered_fares
        )
        results[name] = _summarize(revenue, seats_sold, solution.capacity)
    return results


def paired_revenue_difference(
    candidate: SimulationResult,
    reference: SimulationResult,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
) -> PairedDifference:
    """Estimate candidate-minus-reference revenue using paired replication outcomes.

    Only use this helper for outputs from the same compare_policies call,
    because its uncertainty calculation assumes common random demand paths.
    The confidence interval is a normal-approximation 95% interval.
    """
    if not isinstance(candidate, SimulationResult) or not isinstance(reference, SimulationResult):
        raise ValueError("candidate and reference must be SimulationResult instances")
    candidate_revenue = np.asarray(candidate.revenues, dtype=float)
    reference_revenue = np.asarray(reference.revenues, dtype=float)
    if (candidate_revenue.ndim != 1 or reference_revenue.ndim != 1
            or candidate_revenue.shape != reference_revenue.shape):
        raise ValueError("revenue samples must be one-dimensional and have equal lengths")
    if (candidate_revenue.size < 1
            or not np.all(np.isfinite(candidate_revenue))
            or not np.all(np.isfinite(reference_revenue))):
        raise ValueError("revenue samples must be non-empty and finite")

    differences = candidate_revenue - reference_revenue
    mean_difference = float(differences.mean())
    standard_error = (
        float(differences.std(ddof=1) / np.sqrt(differences.size))
        if differences.size > 1 else float("nan")
    )
    margin = 1.96 * standard_error
    return PairedDifference(
        candidate_name=candidate_name,
        reference_name=reference_name,
        mean_difference=mean_difference,
        standard_error=standard_error,
        ci95_low=mean_difference - margin,
        ci95_high=mean_difference + margin,
        replications=int(differences.size),
    )
