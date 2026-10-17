"""Bellman solver for a finite-horizon airline seat-inventory model."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class DpSolution:
    """Optimal value table and marginal seat values; arrays index [periods, seats]."""
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
        return float(self.values[self.horizon, self.capacity])

def _validate_inputs(capacity, horizon, arrival_probability, fares, probabilities):
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if not np.isfinite(arrival_probability) or not 0.0 <= arrival_probability <= 1.0:
        raise ValueError("arrival_probability must be between 0 and 1")
    fare_array = np.asarray(fares, dtype=float)
    prob_array = np.asarray(probabilities, dtype=float)
    if fare_array.ndim != 1 or prob_array.ndim != 1 or len(fare_array) == 0:
        raise ValueError("fares and probabilities must be non-empty one-dimensional arrays")
    if len(fare_array) != len(prob_array):
        raise ValueError("fares and probabilities must have the same length")
    if not np.all(np.isfinite(fare_array)) or np.any(fare_array < 0):
        raise ValueError("fares must be finite and non-negative")
    if not np.all(np.isfinite(prob_array)) or np.any(prob_array < 0):
        raise ValueError("probabilities must be finite and non-negative")
    probability_sum = float(prob_array.sum())
    if probability_sum <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return fare_array, prob_array / probability_sum

def solve_dp(capacity: int, horizon: int, arrival_probability: float, fares, probabilities) -> DpSolution:
    """Solve the stochastic fare-class model by backward induction in O(T*C*m)."""
    fare_array, prob_array = _validate_inputs(capacity, horizon, arrival_probability, fares, probabilities)
    values = np.zeros((horizon + 1, capacity + 1), dtype=float)
    thresholds = np.full((horizon + 1, capacity + 1), np.inf, dtype=float)
    thresholds[:, 1:] = 0.0
    lam = float(arrival_probability)
    for t in range(1, horizon + 1):
        for x in range(1, capacity + 1):
            reject_value = values[t - 1, x]
            accept_values = fare_array + values[t - 1, x - 1]
            arrival_value = float(np.dot(prob_array, np.maximum(accept_values, reject_value)))
            values[t, x] = (1.0 - lam) * reject_value + lam * arrival_value
    for t in range(1, horizon + 1):
        thresholds[t, 1:] = values[t - 1, 1:] - values[t - 1, :-1]
    return DpSolution(values, thresholds, fare_array, prob_array, lam)

def solve_mean_fare_dp(capacity: int, horizon: int, arrival_probability: float, fares, probabilities) -> DpSolution:
    """Solve a certainty-equivalence approximation with fare fixed at its mean."""
    fare_array, prob_array = _validate_inputs(capacity, horizon, arrival_probability, fares, probabilities)
    mean_fare = float(np.dot(fare_array, prob_array))
    return solve_dp(capacity, horizon, arrival_probability, [mean_fare], [1.0])
