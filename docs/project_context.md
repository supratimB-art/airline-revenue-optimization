# Project context

## What this project is

An independent summer 2026 learning project on dynamic programming, stochastic optimization, and airline seat-inventory control. It documents a path from basic recursive formulations and shortest-path exercises to an interactive finite-horizon airline revenue-management model.

## Learning path recorded from the supplied materials

1. Studied introductory material on the dynamic-programming framework: states, actions, value functions, Bellman recursions, and boundary values.
2. Worked through a deterministic shortest-path example. Manual route enumeration and backward induction both gave A → B → F → H, with cost 1063.
3. Worked through a four-city traveling-salesperson example with two tied optimal tours of cost 13.
4. Extended a route example to a random edge cost and examined when replacing costs with their means is valid.
5. Considered an adaptive route choice after observing a delay, including a value-of-information calculation.
6. Implemented a finite-horizon airline seat-inventory DP and explored value functions and bid-price thresholds in a dashboard.
7. Kept a separate flight-price regression notebook as another airfare-related experiment; it is documented separately because price prediction is not the same task as capacity-control optimization.

See [the detailed learning journey](learning_journey.md), [flight-price prediction notes](flight_price_prediction_notes.md), and [study references](references.md).

## Academic guidance and acknowledgment

Prof. Wang Zizhuo recommended foundational reading and hands-on exercises, including shortest paths and airline capacity control, and gave brief feedback on progress and the report. The implementation and experiments are presented as independent learning work. This does not imply co-authorship, formal research supervision, or endorsement of every code detail or recorded numerical result.

## Scope

The airline model uses illustrative synthetic fare distributions. It is not calibrated to airline booking data and is not a production pricing system. The separate flight-price notebook's original workbook was not supplied, so its model metrics remain recorded historical outputs rather than independently reproduced results.
