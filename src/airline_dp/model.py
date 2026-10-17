"""Bellman solver for a finite-horizon airline seat-inventory model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DpSolution:
    """Optimal value table and marginal seat values.

    Arrays use [periods_remaining, seats_remaining] indexing.
    thresholds[t, x] is the minimum fare accepted in state (x, t).
    The threshold at zero inventory is infinity because no seat is available.
    """

    values: FloatArray
    thresholds: FloatArray
    fares: FloatArray
    probabilities: FloatArray
    arrival_probability: float

    @property
    def horizon(self) -> int:
        return self.values.shape[0] - 1

    @property
    def capacity(self) -> int:
        return self.values.shape[1] - 1

    @property
    def expected_revenue(self) -> float:
        """Optimal expected revenue from the full-capacity, full-horizon state."""
        return float(self.values[self.horizon, self.capacity])


def _finite_real_scalar(value: object, name: str) -> float:
    """Validate one finite real number without accidentally accepting booleans/strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _validate_inputs(
    capacity: int,
    horizon: int,
    arrival_probability: float,
    fares: list[float] | tuple[float, ...] | FloatArray,
    probabilities: list[float] | tuple[float, ...] | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer)) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")

    lam = _finite_real_scalar(arrival_probability, "arrival_probability")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("arrival_probability must be between 0 and 1")

    try:
        fare_array = np.asarray(fares, dtype=float)
        prob_array = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fares and probabilities must be numeric arrays") from exc

    if fare_array.ndim != 1 or prob_array.ndim != 1 or fare_array.size == 0:
        raise ValueError("fares and probabilities must be non-empty one-dimensional arrays")
    if fare_array.size != prob_array.size:
        raise ValueError("fares and probabilities must have the same length")
    if not np.all(np.isfinite(fare_array)) or np.any(fare_array < 0):
        raise ValueError("fares must be finite and non-negative")
    if not np.all(np.isfinite(prob_array)) or np.any(prob_array < 0):
        raise ValueError("probabilities must be finite and non-negative")

    # Scaling before summing avoids overflow for otherwise finite probability weights.
    largest_weight = float(prob_array.max())
    if largest_weight <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    scaled_probabilities = prob_array / largest_weight
    probability_sum = float(scaled_probabilities.sum())
    if not np.isfinite(probability_sum) or probability_sum <= 0.0:
        raise ValueError("probabilities could not be normalized")
    prob_array = scaled_probabilities / probability_sum
    return fare_array, prob_array


def solve_dp(
    capacity: int,
    horizon: int,
    arrival_probability: float,
    fares: list[float] | tuple[float, ...] | FloatArray,
    probabilities: list[float] | tuple[float, ...] | FloatArray,
) -> DpSolution:
    """Solve the stochastic fare-class model by backward induction.

    Complexity is O(horizon * capacity * number_of_fare_classes) time and
    O(horizon * capacity) memory. Probability weights are normalized after
    validation. If the requested scale overflows floating-point arithmetic,
    a ValueError is raised rather than returning infinite value estimates.
    """
    fare_array, prob_array = _validate_inputs(
        capacity, horizon, arrival_probability, fares, probabilities
    )
    values = np.zeros((horizon + 1, capacity + 1), dtype=float)
    thresholds = np.full((horizon + 1, capacity + 1), np.inf, dtype=float)
    thresholds[:, 1:] = 0.0

    lam = float(arrival_probability)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, horizon + 1):
            for x in range(1, capacity + 1):
                reject_value = values[t - 1, x]
                accept_values = fare_array + values[t - 1, x - 1]
                arrival_value = float(np.dot(prob_array, np.maximum(accept_values, reject_value)))
                values[t, x] = (1.0 - lam) * reject_value + lam * arrival_value
                if not np.isfinite(values[t, x]):
                    raise ValueError(
                        "expected revenue overflowed floating-point range; reduce fares or horizon"
                    )

    # A state with t periods remaining compares the current fare to the value
    # difference after one period has elapsed, V[t-1, x] - V[t-1, x-1].
    for t in range(1, horizon + 1):
        thresholds[t, 1:] = values[t - 1, 1:] - values[t - 1, :-1]

    return DpSolution(values, thresholds, fare_array, prob_array, lam)


def solve_mean_fare_dp(
    capacity: int,
    horizon: int,
    arrival_probability: float,
    fares: list[float] | tuple[float, ...] | FloatArray,
    probabilities: list[float] | tuple[float, ...] | FloatArray,
) -> DpSolution:
    """Solve the certainty-equivalence approximation using a constant mean fare.

    Use this solution to derive a policy only. Evaluate that policy under the
    original fare distribution before judging its performance against the full DP.
    """
    fare_array, prob_array = _validate_inputs(
        capacity, horizon, arrival_probability, fares, probabilities
    )
    mean_fare = float(np.dot(fare_array, prob_array))
    if not np.isfinite(mean_fare):
        raise ValueError("mean fare overflowed floating-point range")
    return solve_dp(capacity, horizon, arrival_probability, [mean_fare], [1.0])
