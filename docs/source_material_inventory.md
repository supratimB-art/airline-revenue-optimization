# Source-material inventory

The original files supplied for the summer project are preserved in the accompanying archive `airline-revenue-optimization-source-materials.zip`. They are listed here so the exploratory work can be distinguished from the modular implementation in `src/airline_dp/`.

| Original file | Role |
|---|---|
| `Practical_Deterministic_Dynamic_Programming.ipynb` | Notebook practical on deterministic/stochastic shortest paths, backward induction, and an adaptive MDP extension. |
| `Exploring Shortest Route problems(2).pdf` | Written shortest-route practical/report. |
| `Untitled-1(1).py` | Original Matplotlib interactive seat-inventory DP prototype. |
| `FLIGHT  PRICE PREDICTION IPYNB(1).ipynb` | Separate flight-price prediction notebook; not the same problem as optimal seat-inventory control. |
| `SSP_Weak_Conditions.pdf` | Academic reference: Bertsekas and Yu, *Stochastic Shortest Path Problems Under Weak Conditions*. |
| `Lecture 22 proff Wang(4).pdf` | Dynamic-programming lecture slides by Zizhuo Wang (2017). |

## How these materials relate to the current repository

- The prototype's main model is represented in the modular package `src/airline_dp/`; its implementation has been restructured and tested rather than copied line-for-line.
- The shortest-path notebook/report document the learning progression into dynamic programming and stochastic models.
- The flight-price prediction notebook is a separate exploratory project and should not be presented as part of the seat-inventory Bellman model.
- The SSP paper and lecture slides are third-party academic materials. Before committing their full PDFs to a public repository, confirm redistribution permission. The SSP report is available from the author's MIT page: https://www.mit.edu/~dimitrib/SSP_Weak_Conditions.pdf

The archive preserves the six original files and filenames. This inventory is descriptive; it does not imply the individual files are already committed at these paths in this repository.
