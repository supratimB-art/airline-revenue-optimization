# Learning notes and proposed experiments

These notes document exploratory questions from a self-directed summer 2026 project on dynamic programming and airline capacity control. The project followed reading and implementation suggestions from Prof. Wang Zizhuo; it is a learning exercise rather than a claim of original published research.

## Research question

How do inventory scarcity, booking-horizon length, arrival intensity, and fare dispersion affect optimal bid prices and the revenue cost of simple booking policies?

## Experiment A: fare dispersion with fixed mean

Construct distributions with a fixed mean fare but varying dispersion. For every distribution, solve the exact DP and the mean-fare approximation. Then evaluate both resulting policies under the same true fare distribution using common random numbers. Report exact optimal expected revenue, Monte Carlo revenue of each policy, paired differences, uncertainty, model parameters, and random seed.

Do not claim that a difference between two value functions calculated under different fare models is itself the realized policy regret.

## Experiment B: capacity and booking horizon

Sweep a grid of capacity and horizon values. Plot initial bid price, exact expected revenue, expected seats sold, and policy gap. State whether the goal is exploratory or inferential and preserve parameter choices.

## Experiment C: demand intensity

Vary arrival probability while holding fares and other parameters fixed. Explain how arrival opportunity and inventory scarcity interact. Include boundary checks at arrival probabilities 0 and 1.

## Validation plan

- Test boundary conditions and the one-period case analytically.
- Verify monotonicity of value in capacity and horizon.
- Check diminishing marginal value of capacity numerically.
- Compare exact DP values with Monte Carlo estimates and report Monte Carlo error.
- Use fixed seeds and common random numbers for policy comparisons.

## Extensions

1. More than three fare classes or arbitrary discrete fare support.
2. Nonstationary arrival probabilities or time-dependent fare distributions.
3. Multiple flight legs or network revenue management.
4. Cancellations, no-shows, and overbooking with a revised state model.
5. Demand learning or parameter uncertainty, where the state may need to include a posterior or belief state.

A real-data extension requires a defensible source, data-use compliance, and explicit calibration methods. Price prediction alone does not solve the accept/reject control problem.
