# M0 Specification: Passenger Flight-Purchasing Decision Engine

**Status:** implementation specification  
**Initial audience:** long-haul international travellers booking months ahead  
**Initial example route:** London (LON-area airports) to San Francisco (SFO-area), economy  
**Scope:** formal state, discretization, transition generators, decision objectives, benchmark protocol, hand-computed test cases, and the parallel fare-logging workstream.

This document specifies a small, testable research model. It does not claim that the synthetic generators represent real airfare dynamics. Live fare access and historical-data coverage must be validated independently.

## 1. Decision problem and scope

At each observation time, choose between:

- **BUY:** purchase the least-cost currently available itinerary that satisfies all hard user constraints.
- **WAIT:** observe the market again at the next decision step.
- **FAIL:** at the deadline, if no acceptable itinerary is available, incur an explicit failure penalty.

The initial model uses hard constraints only. Examples include cabin, maximum number of stops, departure/arrival window, maximum total price, and required baggage. Soft preferences and a generalized-cost penalty (Q(j,u)) are deferred until preference elicitation is separately designed and tested.

One decision step is one scheduled observation interval, not necessarily one calendar day. The initial experiments should state the interval explicitly (for example, one day) and run sensitivity checks on it.

### Out of scope for M0

- Live purchasing or autonomous booking.
- Claims of guaranteed cheapest fares.
- Calendar integration.
- Refund/rebooking optimization.
- A trained real-world forecast model.
- Soft-preference utility weights.
- Unvalidated claims that synthetic performance transfers to real markets.

## 2. State and discretization

### 2.1 Core state

Use the compact state

\[
s=(z,a),
\]

where:

- (z) is the current fare-band state.
- (a) is the acceptable-availability state.

The user's trip constraints (u), route, cabin, currency, and other fixed trip details are **model parameters**, not state variables. Quote history is not copied into the DP state. It is used by an estimator to infer transition probabilities and initialize the current state; the DP receives only the resulting compact state and transition model.

For a single itinerary archetype, (z) is one fare bucket and (a\in\{0,1\}) indicates whether an acceptable offer is currently available. For date flexibility or multiple itinerary archetypes, (z) becomes a small vector of fare buckets and (a) becomes an availability bitmask. This is a finite-state approximation, and its limits must be documented.

### 2.2 Fare buckets

Start with three interpretable buckets for a one-way economy example:

| Bucket | Representative fare | Meaning |
|---|---:|---|
| L | $400 | low |
| M | $600 | middle |
| H | $900 | high |

These values are illustrative only. Bucket boundaries and representative values must later be learned from, or calibrated against, real comparable offers. The fare objective uses representative prices, so discretization error must be measured by repeating experiments with finer buckets and alternative representatives.

For the 5-state random-walk generator, use normalized bands (z\in\{1,2,3,4,5\}) with representative prices ($350,$450,$550,$700,$900). Keep the 3-state airline-bucket generator separate so each matrix remains easy to inspect and hand-check.

### 2.3 Availability state

For the first multi-option extension, define two itinerary archetypes:

- (P): preferred acceptable itinerary class (for example, a nonstop itinerary).
- (B): backup acceptable itinerary class (for example, a one-stop itinerary satisfying all hard constraints).

The availability state is a bitmask:

| (a) | Interpretation | Available classes |
|---:|---|---|
| 0 | none | ∅ |
| 1 | preferred only | ({P}) |
| 2 | backup only | ({B}) |
| 3 | both | ({P,B}) |

Only itineraries satisfying hard constraints belong in the acceptable set. A cheap itinerary that violates a hard constraint is excluded before optimization; it is not assigned a large soft penalty.

For multiple classes, store a fare bucket per class, (z=(z_P,z_B)), and an availability bitmask (a). A buy action selects the least-priced available acceptable class. If prices are represented in different currencies, normalize them using a timestamped exchange-rate convention before comparing them; never compare raw amounts across currencies.

### 2.4 Quote history and estimation

Quote history belongs to the data/estimation layer, not the DP state. At decision time (t), the estimator may use only observations timestamped at or before (t) to produce:

- the current bucket(s) (z_t);
- the current availability state (a_t);
- an estimated transition matrix or scenario distribution;
- a data-quality indicator (sufficient / sparse / unavailable).

The initial M0 experiments use known synthetic matrices. Later backtests must re-estimate matrices from training cohorts only and freeze them before evaluating held-out trips.

## 3. Objective and Bellman recursion

Let (t) be the number of future observation opportunities remaining before the purchase deadline. Let (A(s)) be the set of currently available itineraries satisfying all hard constraints, and let (p_j(z)) be the representative total price for itinerary (j) at its current fare bucket.

For a nonterminal state, define the cost of buying now as

\[
B(s)=
\begin{cases}
\min_{j\in A(s)} p_j(z), & A(s)\ne\varnothing,\\
+\infty, & A(s)=\varnothing.
\end{cases}
\]

The (+\infty) value means BUY is not an available action when no acceptable offer exists now; it does **not** mean the user pays an infinite fare.

At the deadline, the terminal cost is

\[
V_0(s)=
\begin{cases}
\min_{j\in A(s)}p_j(z), & A(s)\ne\varnothing,\\
L, & A(s)=\varnothing,
\end{cases}
\]

where (L) is the explicit cost of failing to secure an acceptable itinerary by the deadline. (L) is a modelling parameter, not a universal truth. Test a range of plausible values and report sensitivity. A high penalty should reflect the consequences of the defined failure event without silently asserting that every missed purchase has the same real-world cost.

For (t\ge 1), the risk-neutral recursion is

\[
V_t(s)=\min\left\{B(s),\;
\mathbb{E}\left[V_{t-1}(S')\mid s,\text{WAIT}\right]\right\}.
\]

**Tie-breaking:** BUY wins exact ties. Implement with a documented numerical tolerance for floating-point comparisons; the tolerance must be fixed in tests and not tuned per experiment. If (B(s)=+\infty), WAIT is chosen whenever another observation is allowed.

There is no (c_{wait}) term in M0. Waiting has no separately charged cost unless a concrete, measurable cost is introduced later. The failure penalty already captures the terminal no-acceptable-option outcome.

### 3.1 Risk-sensitive CVaR variant

Implement a second, nested risk-sensitive solver. For upper-tail level (α\) (initially (α=0.90)), define the conditional CVaR of next-step cost (X) by

\[
\operatorname{CVaR}_{\alpha}(X)
=\min_{\eta\in\mathbb R}
\left[\eta+\frac{1}{1-\alpha}\mathbb E[(X-\eta)^+]\right].
\]

Then use the nested recursion

\[
V^{\mathrm{CVaR}}_0(s)=V_0(s),\qquad
V^{\mathrm{CVaR}}_t(s)=
\min\left\{B(s),\;
\operatorname{CVaR}_{\alpha}\left(
V^{\mathrm{CVaR}}_{t-1}(S')\mid s,\text{WAIT}
\right)\right\}.
\]

This is a nested conditional risk measure, not merely a report of the final outcome's CVaR. Compare it with the risk-neutral solver at the same (L), transition model and scenarios. State clearly that this version is more conservative and may pay more on average to reduce upper-tail cost. The α parameter and (L) must be sensitivity-tested, not selected on the test set.

## 4. Transition generator A: fare-bucket random walk

This generator represents local price persistence with adjacent-band movement. It is intentionally simple and does not claim to describe the real airfare market.

Use five ordered fare bands (1,…,5). Rows denote the current band and columns the next band. With state order ((1,2,3,4,5)), set

\[
P_{\mathrm{RW}}=
\begin{bmatrix}
0.70&0.30&0&0&0\\
0.20&0.55&0.25&0&0\\
0&0.25&0.50&0.25&0\\
0&0&0.25&0.55&0.20\\
0&0&0&0.30&0.70
\end{bmatrix}.
\]

Each row sums to one. The matrix reflects at the boundaries rather than allowing a transition outside the defined price range. This is a **scenario generator**, not an empirically fitted transition matrix.

For an initial three-bucket toy example, use the reduced matrix

\[
P_{\mathrm{RW},3}=
\begin{bmatrix}
0.70&0.30&0\\
0.20&0.60&0.20\\
0&0.30&0.70
\end{bmatrix},
\]

with buckets (L,M,H). This reduced matrix is useful for hand calculations and unit tests.

## 5. Transition generator B: airline fare-bucket adaptation

The existing `airline_dp` model has a finite fare-class distribution and an arrival/no-arrival event. Its original semantics are airline-side customer arrivals and seat acceptance, **not** a time-series model of market airfare. We will reuse its discrete fare-class structure as a second synthetic generator, with an explicit reinterpretation for the passenger experiment.

For the three representative fare buckets ((L,M,H)), set quote-event probability (lambda=0.5) and bucket probabilities

\[
p=(0.50,0.30,0.20).
\]

In each step:

1. With probability (1-\lambda), no new quote-state event occurs and the current fare bucket persists.
2. With probability (\lambda), a new bucket is sampled from (p).

Thus

\[
P_{\mathrm{AIR}}=(1-\lambda)I+\lambda\mathbf{1}p^\top
=
\begin{bmatrix}
0.75&0.15&0.10\\
0.25&0.65&0.10\\
0.25&0.15&0.60
\end{bmatrix}.
\]

Rows are current bucket; columns are next bucket. Every row sums to one. This is an adaptation inspired by the existing airline model's event-plus-fare-class structure; it is not a claim that the airline model itself estimates these probabilities or that this generator is realistic.

### 5.1 Availability transitions

Use the following explicit availability matrix for initial multi-option experiments. State order is (a=(0,1,2,3)), as defined above. Rows are current state, columns next state:

\[
P_A=
\begin{bmatrix}
0.80&0.10&0.10&0\\
0.15&0.65&0&0.20\\
0.15&0&0.65&0.20\\
0.05&0.20&0.20&0.55
\end{bmatrix}.
\]

Each row sums to one. This matrix allows an option to appear or disappear and permits no acceptable option to remain unavailable. These are illustrative probabilities, to be stress-tested; they must not be presented as estimated market rates.

For M0, use the factorized transition approximation

\[
\Pr(z',a'\mid z,a)=P_Z[z,z']P_A[a,a'].
\]

This assumes fare-band and availability transitions are conditionally independent. That is a known simplification. The M2 experiment must test it and include a coupled stress scenario where high fares and scarce availability are more likely to co-occur.

For the multi-itinerary state (z=(z_P,z_B)), the simplest version applies a fare transition to each available class independently, while (P_A) governs which classes are available. This independence is only a baseline generator; do not treat it as an established market fact.

## 6. Main misspecification experiment

The headline experiment is a **cross-generator transfer test**, not merely a set of random stress tests.

1. Solve the DP under (P_{\mathrm{RW}}), then evaluate its resulting policy on held-out paths generated by (P_{\mathrm{AIR}}).
2. Solve the DP under (P_{\mathrm{AIR}}), then evaluate its policy on held-out paths generated by (P_{\mathrm{RW}}).
3. Evaluate both policies on matched paths from their own generator.
4. Compare against tuned simple baselines and a hindsight oracle on every evaluation path.
5. Repeat across multiple horizons, failure penalties (L), fare volatility levels, and availability transition settings.
6. Include the factorized availability generator and a coupled high-fare/low-availability stress generator.

Report policy value under the assumed model separately from realized performance under the evaluation generator. A policy can be optimal under its own model and poor under misspecification. That gap is a central research result.

For a reproducible first benchmark, use fixed published seeds, at least 100,000 synthetic paths per generator/configuration for Monte Carlo precision, and a separate set of seeds for tuning versus final evaluation. Monte Carlo path count is not a substitute for independent real-trip count in historical evaluation.

## 7. Baselines and oracle protocol

The DP policy is not meaningfully validated by beating an arbitrary fixed threshold. Use the following protocol.

### 7.1 Baselines

- **Buy now:** purchase the best acceptable current offer.
- **Fixed wait:** wait a fixed number of decision steps, then buy the best available option; if none exists, continue only according to a pre-specified rule until the deadline.
- **Fixed price threshold:** buy when the best acceptable current price is at or below threshold (q); if no such price occurs, use the pre-specified deadline fallback.
- **DP:** risk-neutral and nested-CVaR variants.
- **Hindsight oracle:** for each realized path, choose the lowest-cost acceptable offer that was available at any decision time on that path; if none was available, pay (L). This oracle sees the complete path and is unattainable online. It is a lower bound on cost under the same observed path and availability rules, not a deployable policy.

For multiple itinerary classes, all policies operate on the same acceptable set and hard constraints. Baselines may not receive additional information unavailable to DP.

### 7.2 Tune baselines only on training paths

For each training configuration, search the fixed threshold over all bucket representatives and relevant within-bucket price thresholds. Search fixed waiting time over all permissible wait lengths from zero to the deadline. Select each baseline's parameters by **minimum mean training cost**, with ties resolved toward the earlier purchase rule. If tuning continuous thresholds on continuous fares later, define the candidate grid from training data only.

Freeze the selected parameters before test evaluation. Do not re-tune baselines separately on each test path, test cohort or test result. The oracle is computed per path only as an explicitly labelled hindsight reference.

Report test cost versus:
1. buy now;
2. tuned fixed threshold;
3. tuned fixed wait;
4. hindsight oracle.

The primary claim is whether DP improves over the best tuned online baseline, not whether it beats an arbitrary rule.

### 7.3 Metrics

Report at least:

- mean and median realized cost;
- 90th and 95th percentile cost;
- failure rate (no acceptable option by deadline);
- mean regret to the hindsight oracle;
- paired cost difference versus each baseline;
- nested CVaR at the chosen level;
- policy action rates (buy/wait) and average purchase lead time.

Define regret per trip as (C_{\mathrm{policy}}-C_{\mathrm{oracle}}\), using the same path and failure penalty. Do not call oracle regret “savings” because the oracle uses future information.

## 8. Historical evaluation and confidence intervals

The independent unit is a **trip**, not an individual quote. Quotes from the same trip are repeated measurements and must never be counted as independent samples.

Use a chronological train/validation/test split by trip cohort (for example, departure month or route-departure cohort). Estimate transition matrices and tune baseline parameters on training trips only; use validation for model selection; freeze everything before the final test.

For confidence intervals, use a paired cohort bootstrap:

1. Group trips into pre-defined route/departure cohorts.
2. Resample cohorts with replacement, preserving all quote observations and paired policy outcomes for every trip in a sampled cohort.
3. Recompute mean paired cost differences and regret for each bootstrap replicate.
4. Report percentile or bias-corrected 95% intervals, documenting the bootstrap method and number of replicates (at least 2,000).
5. If there are too few independent cohorts for a defensible cluster bootstrap, say so; do not substitute quote-level resampling.

### 8.1 Sample-size planning

Before collecting the final holdout, choose the smallest practically meaningful paired improvement δ in the currency used for evaluation. Estimate the standard deviation σ_D of per-trip paired cost differences from a separate pilot/training cohort. For an approximate two-sided 5% test with 80% power, use

\[
n\approx\left(\frac{(1.96+0.84)\sigma_D}{\delta}\right)^2
=7.84\left(\frac{\sigma_D}{\delta}\right)^2.
\]

For illustration only, if σ_D=$100 and the smallest meaningful improvement is δ=$25, then (n\approx126\) independent trips; allowing 15% attrition gives about 149 trips. This is a planning approximation, not a guaranteed power calculation. Cohort clustering, multiple comparisons and heavy-tailed outcomes can increase the required sample. Set a provisional minimum of 200 held-out trips, then use the pilot variance and the desired effect size to increase that number when needed. If the available trip count is below the required size, report the result as exploratory rather than conclusive.

## 9. Hand-computed toy cases

All prices and penalties below are illustrative. These are exact finite-state checks, not empirical results.

### Case 1 — Risk-neutral solver waits for a favourable expected outcome

One acceptable itinerary is available now for $100. There is one future observation. At the next observation, the fare is $50 with probability 0.9 and $200 with probability 0.1. The offer is available in both outcomes; set (L=$300) although failure cannot occur in this case.

At the deadline:

\[
V_0(L)=50,\qquad V_0(H)=200.
\]

Waiting has expected cost

\[
0.9(50)+0.1(200)=45+20=65.
\]

Buying now costs $100, so

\[
V_1=\min(100,65)=65.
\]

**Expected answer:** WAIT; value $65.

### Case 2 — Failure penalty makes availability risk matter

No acceptable offer is available now. There is one future observation. At the next observation, an acceptable $90 offer exists with probability 0.8; otherwise no acceptable offer exists and the terminal penalty is (L=$300).

\[
V_0(\text{available})=90,\qquad
V_0(\text{none})=300.
\]

Therefore,

\[
V_1=0.8(90)+0.2(300)=72+60=132.
\]

BUY is currently infeasible, so **WAIT**; expected cost is $132. The calculation must treat “query failed/unknown” as an observation-quality state in real data, not automatically as the “none available” state. This toy case assumes true unavailability is known.

### Case 3 — Exact tie buys now

An acceptable itinerary costs $100 now and will deterministically cost $100 at the next observation. There is one future observation.

\[
V_0=100,\qquad
V_1=\min(100,100)=100.
\]

**Expected answer:** BUY, by the explicit tie-breaking rule. This test guards against unstable floating-point or branch-order behaviour.

### Case 4 — Nested CVaR changes the action

An acceptable offer costs $100 now. Waiting leads to cost $50 with probability 0.9 and $200 with probability 0.1. For risk-neutral DP, Case 1 gives expected waiting cost $65, so it waits.

For upper-tail α=0.90, the worst 10% of waiting outcomes cost $200, hence

\[
\operatorname{CVaR}_{0.90}(C_{\mathrm{wait}})=200.
\]

The CVaR policy compares $100 with $200 and **BUYS now**. This is the intended risk-sensitive behaviour; it does not imply CVaR always lowers average cost.

### Case 5 — Hard constraints override a cheaper invalid option

Two offers are displayed: $70 for an itinerary with an unacceptable overnight layover, and $120 for an itinerary satisfying all hard constraints. The $70 offer is excluded from (A(s)) before optimization.

\[
B(s)=\min_{j\in A(s)}p_j=120.
\]

**Expected answer:** BUY the $120 acceptable itinerary if buying is optimal versus waiting; never select the $70 itinerary. This test belongs in the offer-filtering layer and the DP integration tests.

### Case 6 — Flexibility across departure dates creates option value

The traveller can depart on either of two dates, and both satisfy their hard constraints. At the current observation, date A has an acceptable $100 offer and date B has an acceptable $70 offer. There are no other constraints distinguishing them.

For the flexible-date choice,

\[
B(s)=\min(100,70)=70,
\]

so the immediate best acceptable offer is date B at $70. A fixed-date-only version restricted to date A pays $100. The difference is $30 in this constructed example. M2 must extend this beyond the trivial immediate choice to scenarios where each date's fare and availability evolve differently, because that is where flexibility changes the value of waiting.

### Case 7 — No acceptable offer at the deadline

At (t=0), (A(s)=\varnothing), and (L=$300.

\[
V_0(s)=L=300.
\]

**Expected answer:** explicit failure outcome with cost $300. The interface must not invent an itinerary or silently return a null cost.

## 10. Quote logging starts in parallel with M0

Do not wait for the DP implementation before collecting data. Start a minimal append-only quote logger while M0 is being specified and coded. Its first purpose is to learn whether data sources provide stable, comparable observations and to accumulate history over calendar time.

Minimum record fields:

- query timestamp (UTC) and source;
- origin/destination and airport/region semantics;
- intended departure date and allowed date-flexibility window;
- passenger count, cabin and fare family where known;
- itinerary legs, local departure/arrival times, duration and stops;
- total price, currency, taxes/fees and baggage inclusion when available;
- refund/change conditions when available;
- availability result: available / explicitly unavailable / unknown / query error;
- source response identifier and verified booking deep link when available;
- parser/schema version and raw-response reference where permitted by source terms.

Log failures and no-result responses as separate observations. Do not store secrets or unnecessary personal data. Respect source terms, rate limits and privacy requirements; do not assume scraping is permitted. A missing quote is not proof of unavailability. A displayed quote is not proof that checkout will honour it, so retain the observation timestamp and verification status.

**Logger acceptance criteria:** append-only records, UTC timestamps, schema version, explicit query status, no silent conversion of errors into unavailable fares, deduplication rules documented, and a daily health summary. Begin with one route family and a small set of future departure dates. Record the source and legal/terms review before scaling queries.

## 11. Acceptance gates for M0

M0 is complete when:

1. State and action definitions are documented and implemented without quote history embedded in the DP state.
2. Both fare transition matrices and the availability matrix are validated as stochastic (finite, non-negative entries and rows summing to one).
3. The risk-neutral and nested-CVaR solvers pass all hand-computed cases above.
4. BUY wins exact ties under a documented tolerance.
5. Hard-constraint filtering occurs before optimization.
6. The cross-generator experiment is runnable from a recorded configuration and seed.
7. Fixed-threshold and fixed-wait baselines are tuned on training paths only; the hindsight oracle is labelled as non-deployable.
8. Test results include regret, tail cost and failure rate—not just mean price.
9. The quote logger is running in parallel and treats unavailable, unknown and query-error outcomes separately.
10. The README links to this specification and clearly distinguishes the passenger model from the existing airline-side revenue model.

## 12. Required reporting language

Allowed: “On the specified synthetic generators, the policy reduced mean cost versus the tuned threshold baseline by X, with confidence interval Y; cross-generator performance changed by Z.”

Not allowed without appropriate evidence: “The model predicts the cheapest flight,” “guaranteed savings,” or “beats existing flight-purchasing services.”

A positive synthetic result demonstrates that the implementation behaves as expected under stated assumptions. A positive historical result is still limited to the covered routes, periods, fare products, data sources and evaluation design.
