"""Decision rules for the airline inventory model."""
from __future__ import annotations
from collections.abc import Callable
from .model import DpSolution
Policy = Callable[[float, int, int], bool]

def accept_if_fare_at_least(solution: DpSolution) -> Policy:
    """Return the optimal threshold policy associated with a DP solution."""
    def policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
        if seats_remaining <= 0 or periods_remaining <= 0:
            return False
        if not 0 <= seats_remaining <= solution.capacity:
            raise ValueError("seats_remaining is outside the solution state space")
        if not 0 <= periods_remaining <= solution.horizon:
            raise ValueError("periods_remaining is outside the solution state space")
        return fare >= solution.thresholds[periods_remaining, seats_remaining]
    return policy

def fixed_threshold_policy(threshold: float) -> Policy:
    """Accept fares at or above a fixed benchmark threshold."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    def policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
        return seats_remaining > 0 and periods_remaining > 0 and fare >= threshold
    return policy

def accept_all_policy(fare: float, seats_remaining: int, periods_remaining: int) -> bool:
    """Accept any fare whenever inventory and time remain."""
    return seats_remaining > 0 and periods_remaining > 0
