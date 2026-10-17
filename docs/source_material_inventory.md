# Source-material inventory

This inventory records the six files supplied for documenting the summer study. The repository tracks a concise account of what I did and the references I used; it does **not** reproduce every uploaded attachment byte-for-byte.

| Supplied file | What it documents | How it is represented here |
|---|---|---|
| `Practical_Deterministic_Dynamic_Programming.ipynb` | A worked practical on a deterministic shortest path and a four-city traveling-salesperson example, with code and written answers. | Outcomes summarized in [the learning journey](learning_journey.md). |
| `Exploring Shortest Route problems(2).pdf` | A 19-page written practical covering deterministic shortest paths, a stochastic edge, route PMFs, a stochastic Bellman equation, adaptive decisions, value of information, and scalability. | Main calculations and a question to recheck summarized in [the learning journey](learning_journey.md). |
| `Untitled-1(1).py` | An interactive Matplotlib prototype for a finite-horizon airline seat-inventory DP. | The current package and dashboard are a modular reimplementation; see [the learning journey](learning_journey.md) and `src/airline_dp/`. |
| `FLIGHT  PRICE PREDICTION IPYNB(1).ipynb` | A separate supervised regression experiment on flight-price records. | Reported metrics and reproducibility caveats summarized in [flight-price prediction notes](flight_price_prediction_notes.md). |
| `SSP_Weak_Conditions.pdf` | Bertsekas and Yu's theoretical paper on stochastic shortest-path problems under weak assumptions. | Bibliographic reference and link in [references](references.md); the full PDF is not redistributed in this repository. |
| `Lecture 22 proff Wang(4).pdf` | Zizhuo Wang's 2017 lecture slides on dynamic programming. | Attribution and topic summary in [references](references.md); the full slide deck is not redistributed in this repository. |

## Important distinctions

- The shortest-path practical, flight-price notebook, and airline seat-inventory model are related learning activities, but they are not one single model.
- Flight-price regression estimates ticket price from features; seat-inventory control optimizes an accept/reject decision under limited capacity and uncertain future demand.
- The reported machine-learning metrics are the original notebook's stored results. The dataset and saved model were not supplied, so they are not reproduced by the current automated tests.
- The current `src/airline_dp/` implementation is a structured refactor, with tests and a dashboard. It is not claimed to be a line-by-line transcription of the original prototype.
- The academic paper and lecture slides are linked/cited rather than committed as full PDFs, since redistribution permission was not established.
