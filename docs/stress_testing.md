# Stress-test record

This record documents reliability checks for the finite-horizon seat-inventory solver and policy evaluator. It is not a claim of validation against airline data.

## Changes made

- Hardened input validation for model parameters, fares, probabilities, policy state values, fixed thresholds, replication counts, and random seeds.
- Normalized finite probability weights after scaling by their maximum, avoiding overflow when otherwise valid weights are extremely large.
- Added an explicit error when expected-revenue arithmetic overflows rather than allowing infinite estimates to propagate.
- Reject malformed, non-callable, empty, or duplicate-named policy comparisons.
- A one-replication standard error is reported as NaN because sampling variability cannot be estimated from a single observation.
- Added a paired revenue-difference estimator with a standard error and approximate 95% interval for policies evaluated on common random paths.
- The dashboard reports paired gaps and uncertainty, and warns when empirical standard error is zero although stochastic outcomes remain possible.

## Automated checks

Run from the repository root:

    python -m compileall -q app src tests
    python -m pytest -q

The suite includes a closed-form one-seat model, exhaustive enumeration of all state-dependent accept/reject policies for a tiny instance, zero-capacity/horizon and arrival-probability boundaries, malformed/extreme inputs, monotonicity checks, paired-interval arithmetic, a default-path dashboard smoke test, and comparison of DP values to simulation over multiple seeds.

On the validation runtime (Python 3.13, NumPy 2.3, pandas 2.2, Plotly 6.5), all 69 tests passed. An additional 500 randomly generated model instances passed checks for finite, non-negative values; value monotonicity in time and capacity; and nonincreasing marginal value as capacity increases.

## Interpretation caveats

- Monte Carlo standard errors summarize variation observed in the chosen sample. If all replications happen to produce the same revenue, an empirical standard error of zero does not prove the underlying revenue is deterministic. Increase replication counts and use several seeds when outcomes may be rare.
- The paired confidence interval assumes result vectors came from one call to compare_policies, which ensures that they share simulated demand paths. Do not use the paired helper on independently simulated samples.
- The dashboard smoke test executes the default path with stubbed Streamlit widgets but real Plotly chart construction, catching stale chart-column references. A live Streamlit server was not launched because Streamlit was not installed in the offline validation runtime; it remains a declared project dependency.
- The source workbook for the separate flight-price notebook was not supplied, so its historical metrics could not be reproduced.
