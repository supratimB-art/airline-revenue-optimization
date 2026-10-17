# Summer Learning Project: Airline Revenue Optimization with Dynamic Programming

An independent summer 2026 learning project exploring stochastic dynamic programming through a small airline seat-inventory control problem. The model and interactive dashboard connect mathematical study with implementation, testing, and computational experiments.

## Background and academic guidance

This project grew out of a self-directed study of dynamic programming and stochastic optimization. After I asked **Prof. Wang Zizhuo** for a learning path, he recommended Dimitri P. Bertsekas's *Dynamic Programming and Optimal Control*, shared lecture slides, and suggested hands-on problems such as shortest paths and airline capacity control. I explored a shortest-path problem and then developed this airline revenue-management model and dashboard. Prof. Wang offered brief feedback on my progress and report.

This is an **independent learning project informed by academic guidance**, not a co-authored paper or a claim of formal research supervision. The model uses illustrative synthetic fares and is not calibrated to real airline data.

## Model

There are C seats and T booking periods. Each period has an arrival with probability λ; conditional on arrival, the fare is drawn from a discrete distribution {(f_i,p_i)}. At most one customer arrives per period. Accepting a booking earns the fare and consumes one seat. The baseline excludes cancellations, refunds, overbooking, variable costs, and strategic customer responses.

Let V_t(x) be maximum expected revenue with x seats and t periods remaining. Boundary conditions are V_0(x)=0 and V_t(0)=0. The Bellman recursion is

V_t(x) = (1−λ)V_{t−1}(x) + λ Σ_i p_i max{f_i + V_{t−1}(x−1), V_{t−1}(x)}.

The bid-price threshold is Δ_t(x)=V_{t−1}(x)−V_{t−1}(x−1). Accept fare f when f ≥ Δ_t(x); ties are accepted. Backward induction takes O(T C m) time and O(T C) memory.

## Quick start

Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app/dashboard.py
```

Run tests with `pytest`.

## Project structure

- `app/dashboard.py` — interactive Streamlit dashboard
- `src/airline_dp/model.py` — Bellman solver and thresholds
- `src/airline_dp/policies.py` — optimal threshold and benchmark policies
- `src/airline_dp/simulation.py` — reproducible Monte Carlo policy evaluation
- `tests/` — model and simulation tests
- `docs/project_context.md` — learning path and attribution
- `docs/research_notes.md` — assumptions, validation, and experiment ideas

## Dashboard

Explore the value function, bid-price thresholds, accept/reject decisions, and policy performance. The policy comparison uses common simulated demand paths and reports Monte Carlo uncertainty. A certainty-equivalence comparison evaluates the policy from a mean-fare approximation under the true fare distribution rather than comparing values from mismatched models.

## Baseline parameters

Default example: C=10 seats, T=30 periods, λ=0.5, fares ₹1,000/₹2,500/₹5,000, probabilities 0.50/0.30/0.20. These are illustrative, not real airline data.

## Research cautions

- This is a normative control model, not a market-price prediction model.
- Bernoulli arrivals allow at most one arrival per period and are not automatically equivalent to a continuous-time Poisson process.
- Monte Carlo estimates are noisy; report seed, replication count, and uncertainty.
- Do not describe synthetic simulations as empirical validation. Real-data work requires defensible demand and fare calibration.

## Possible next experiments

1. Vary fare dispersion while holding mean fare fixed.
2. Sweep capacity, horizon, and arrival probability.
3. Compare policies using paired outcomes on common random numbers.
4. Add multiple fare classes, network effects, or cancellations only with an explicitly revised model.

## License and attribution

Choose a license before public release. Cite external datasets, papers, and code used in future extensions. The current baseline uses synthetic, user-specified fare distributions.
