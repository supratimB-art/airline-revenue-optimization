# My summer learning journey: dynamic programming

This page records the work I completed and the ideas I worked through during my summer 2026 study. It is a learning log, not a claim that every experiment is a new research result or has been independently validated.

## 1. Learning the DP framework

The introductory lecture notes by Prof. Zizhuo Wang explain dynamic programming through recursion: define the state, available actions, immediate cost or reward, value function, Bellman recursion, and terminal condition. The central idea is to reuse solved smaller subproblems rather than enumerate every complete sequence.

I used this framework first on shortest paths and then carried it over to a finite-horizon seat-inventory problem. Prof. Wang suggested the reading and implementation exercises; the practical work and code here are my own learning exercises.

## 2. Deterministic shortest path

The worked graph has four stages, from A through B/C/D, then E/F/G, to H. I compared all nine possible routes and implemented backward induction:

\[
V(H)=0,\qquad V(i)=\min_{j:(i,j)\in E}\{d_{ij}+V(j)\}.
\]

The lowest-cost path is **A → B → F → H**, with total cost **1063**. Enumeration, hand calculation, and the DP implementation in the practical agree.

## 3. A small traveling-salesperson exercise

I also worked through a four-city tour that starts and ends at A. The two minimum tours have the same total cost, **13**:

- A → B → D → C → A
- A → C → D → B → A

The DP implementation returns one of the tied optimal tours. This small example helped illustrate how the state must keep track of which cities remain to be visited, rather than only the current city.

## 4. Stochastic shortest path and certainty equivalence

The written practical then makes the edge F → H random: cost 450 with probability 0.7 and cost 900 with probability 0.3. At the stated base probability, the recorded fixed-route results are:

| Fixed route | Recorded expected cost | Recorded standard deviation |
|---|---:|---:|
| A → B → F → H | 1198 | about 204 |
| A → C → F → H | 1271 | about 204 |

Replacing the random edge cost by its mean, 585, gives the same best full route A → B → F → H for this non-adaptive expected-cost problem. The best onward decision at D changes from F in the deterministic example to E in the stochastic example. These examples illustrate that a local decision can change even when the globally best starting route remains the same.

The practical's stated conditions for exact certainty equivalence are: additive costs, independent edge costs, and a risk-neutral linear expected-cost objective. These conditions matter; taking the mean is not a universal substitute for a random variable.

## 5. Adaptive decisions and the value of information

The practical next asks what happens if the traveller learns the delay status at F before choosing the onward route. The direct edge costs 450 when clear and 900 when delayed; the alternative route costs 600.

At F, the adaptive expected cost is

\[
V_{\mathrm{adaptive}}(F)=450+150p,
\]

whereas the best decision made without observing the status has expected cost

\[
V_{\mathrm{nonadaptive}}(F)=\min\{450+450p,600\}.
\]

The report calculates a value of information of 90 at p = 0.3 and 75 at p = 0.5 at node F. The general lesson is that information matters when it can change the preferred action.

**A point to recheck before reusing the report's results:** its later discussion says the global value of information from A reaches 200 at p = 1. That statement appears inconsistent with the displayed node-F benchmark: at p = 1, both the adaptive and best non-adaptive choice at F cost 600, so the local value of information is zero. The original recorded claim is not silently treated here as a verified result.

## 6. Moving to airline seat-inventory control

The airline prototype uses a finite booking horizon. The state is (x,t), where x seats and t booking periods remain. In each period, a customer arrives with probability λ; an arriving customer offers a fare drawn from a discrete distribution. The decision is to accept the fare and use a seat, or reject it and preserve capacity.

\[
V_t(x)=(1-\lambda)V_{t-1}(x)+\lambda\sum_i p_i\max\{f_i+V_{t-1}(x-1),V_{t-1}(x)\}.
\]

The optimal decision compares the fare with the marginal value of a seat, \(V_{t-1}(x)-V_{t-1}(x-1)\). This is a different control problem from predicting a ticket's market price: it decides whether to accept an offered fare given scarce capacity and future demand.

The illustrative default is C = 10 seats, T = 30 periods, λ = 0.5, with fares ₹1,000 / ₹2,500 / ₹5,000 and probabilities 0.50 / 0.30 / 0.20. Under the current repository solver, its optimal expected revenue is about **₹27,435.31** and its initial bid price is about **₹1,197.58**. These are model outputs under synthetic assumptions, not empirical estimates for an airline.

The original interactive Matplotlib prototype is the starting point for the repository's current modular solver, policy helpers, simulation code, Streamlit dashboard, and tests. The current code is a reimplementation/refactor rather than a claim of line-by-line reproduction.

## 7. Separate experiment: flight-price prediction

I also kept a separate notebook on predicting ticket prices from a scraped dataset. It describes a dataset of 5,303 rows collected from Yatra with Selenium, exploratory plots, feature engineering, encoding/scaling, and multiple regression models. It is included in the project's history because it is related to airfare, but it is **not** the same problem as airline revenue management or dynamic programming.

The notebook's own recorded results and caveats are summarized separately in [flight-price prediction notes](flight_price_prediction_notes.md). The Excel dataset is not included here, so those figures are historical notebook outputs and have not been reproduced by this repository's test suite.

## What this repository currently demonstrates

- Writing down a Bellman recursion and boundary conditions.
- Solving small deterministic and stochastic examples by backward induction.
- Connecting a fare-acceptance decision to the marginal value of inventory.
- Comparing a model-derived policy with simple benchmarks using repeatable simulations.
- Recording assumptions, provenance, limitations, and questions that need rechecking.
