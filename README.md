# Summer Learning Project: Airline Revenue Optimization with Dynamic Programming

An independent summer 2026 learning project exploring stochastic dynamic programming through a small airline seat-inventory control problem. The repository documents the work from worked shortest-path exercises to an interactive model.

## Background and academic guidance

This project grew out of a self-directed study of dynamic programming and stochastic optimization. After I asked **Prof. Wang Zizhuo** for a learning path, he recommended Dimitri P. Bertsekas's *Dynamic Programming and Optimal Control*, shared lecture slides, and suggested hands-on exercises such as shortest paths and airline capacity control. I worked through the exercises and then developed an airline seat-inventory model and dashboard. Prof. Wang offered brief feedback on my progress and report.

This is an **independent learning project informed by academic guidance**, not a co-authored paper or a claim of formal research supervision. The airline model uses illustrative synthetic fares and is not calibrated to real airline data.

## What I worked through

- **Deterministic shortest path:** formulated the state/value function and used backward induction; the worked route A → B → F → H has total cost 1063.
- **Traveling-salesperson exercise:** compared four-city tours; two routes tie for the minimum cost of 13.
- **Stochastic shortest path:** made one route edge random, computed route cost distributions, and considered when replacing random costs with their means preserves the optimal non-adaptive route.
- **Adaptive decision-making:** added an observed delay state and examined how information can change the best onward action.
- **Airline seat-inventory control:** implemented the Bellman recursion for fare acceptance, with bid-price thresholds, simulation comparisons, and an interactive dashboard.
- **Separate flight-price prediction notebook:** explored supervised regression on a scraped airfare dataset. This estimates price from features; it is not the same as deciding whether to accept a booking with limited inventory.

More detailed notes are in [the learning journey](docs/learning_journey.md), [flight-price prediction notes](docs/flight_price_prediction_notes.md), and [source-material inventory](docs/source_material_inventory.md). The academic references are listed in [references](docs/references.md).

## Airline model

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
- `src/airline_dp/model.py` — Bellman solver and bid-price thresholds
- `src/airline_dp/policies.py` — optimal threshold and benchmark policies
- `src/airline_dp/simulation.py` — reproducible Monte Carlo policy evaluation
- `tests/` — model and simulation tests
- `docs/learning_journey.md` — worked examples and progression of the study
- `docs/flight_price_prediction_notes.md` — separate regression notebook and caveats
- `docs/references.md` — attribution and references
- `docs/research_notes.md` — proposed experiments and extensions

## Baseline parameters

Default example: C=10 seats, T=30 periods, λ=0.5, fares ₹1,000/₹2,500/₹5,000, probabilities 0.50/0.30/0.20. Under the current solver, optimal expected revenue is about ₹27,435.31 and the initial bid price is about ₹1,197.58. These are calculations under synthetic assumptions, not real airline results.

## Research cautions and reproducibility

- This is a normative accept/reject control model, not a market-price prediction model.
- Bernoulli arrivals allow at most one arrival per period and are not automatically equivalent to a continuous-time Poisson process.
- Monte Carlo estimates are noisy; report seed, replication count, and uncertainty.
- The flight-price notebook expects a local `Flight_Prices.xlsx` workbook that was not supplied, so its stored metrics are documented as historical notebook outputs, not results reproduced by these tests.
- The stochastic shortest-path report contains one global value-of-information statement that appears inconsistent with its displayed node-level calculation; this is flagged in the learning log instead of being presented as verified.
- The academic paper and lecture slides are cited rather than redistributed, as permission to republish the full PDFs was not established.

## Possible next experiments

1. Vary fare dispersion while holding mean fare fixed.
2. Sweep capacity, horizon, and arrival probability.
3. Compare policies using paired outcomes on common random numbers.
4. Add multiple fare classes, network effects, or cancellations only with an explicitly revised model.

## License

No license has been added yet. Choose a license before inviting reuse of the code. Cite external datasets, papers, and code used in future extensions.
