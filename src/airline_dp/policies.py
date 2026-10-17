"""Decision rules for the airline inventory model."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np

from .model import DpSolution

Policy = Callable[[float, int, int], bool]


def _validate_decision_inputs(
    fare: float, seats_remaining: int, periods_remaining: int
) -> tuple[float, int, int]:
    if isinstance(fare, (bool, np.bool_)) or not isinstance(fare, Real):
        raise ValueError("fare must be a finite, non-negative real number")
    fare_value = float(fare)
    if not np.isfinite(fare_value) or fare_value < 0:
        raise ValueError("fare must be a finite, non-negative real number")

    for value, name in (
        (seats_remaining, "seats_remaining"),
        (periods_remaining, "periods_remaining"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return fare_value, int(seats_remaining), int(periods_remaining)


def accept_if_fare_at_least(solution: DpSolution) -> Policy:
    """Return the threshold policy associated with a dynamic-programming solution."""
    def policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
        fare_value, x, t = _validate_decision_inputs(fare, seats_remaining, periods_remaining)
        if x > solution.capacity:
            raise ValueError("seats_remaining is outside the solution state space")
        if t > solution.horizon:
            raise ValueError("periods_remaining is outside the solution state space")
        if x == 0 or t == 0:
            return False
        return fare_value >= solution.thresholds[t, x]
    return policy


def fixed_threshold_policy(threshold: float) -> Policy:
    """Return a benchmark that accepts fares at or above a fixed threshold."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite, non-negative real number")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError("threshold must be a finite, non-negative real number")

    def policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
        fare_value, x, t = _validate_decision_inputs(fare, seats_remaining, periods_remaining)
        return x > 0 and t > 0 and fare_value >= threshold_value
    return policy


def accept_all_policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
    """Accept any non-negative fare while inventory and time remain."""
    _fare_value, x, t = _validate_decision_inputs(fare, seats_remaining, periods_remaining)
    return x > 0 and t > 0
