# Study references and provenance

These are the source materials I used or recorded during the summer learning project. References are acknowledged here without asserting endorsement of the implementation.

## Dynamic programming lecture

**Zizhuo Wang.** *Lecture 22: Dynamic Programming*. Department of Industrial and Systems Engineering, University of Minnesota, 23 November 2017. The supplied lecture slides introduce recursion, shortest-path value functions, states, actions, Bellman equations, boundary values, and a coin-picking game. The original PDF is a course/lecture resource and is not copied into this repository.

## Stochastic shortest-path paper

**Dimitri P. Bertsekas and Huizhen Yu.** *Stochastic Shortest Path Problems Under Weak Conditions*. MIT Laboratory for Information and Decision Systems Report LIDS-2909, first dated August 2013; revised March 2015 and January 2016.

- Author-hosted PDF: https://www.mit.edu/~dimitrib/SSP_Weak_Conditions.pdf

The paper studies finite-state stochastic shortest-path problems under weaker conditions than the classical SSP assumptions. It discusses proper and improper policies and explains why Bellman-equation and value-/policy-iteration behavior can become delicate when nonpositive-cost cycles are possible. It is theoretical background for the stochastic-shortest-path learning exercise, not an empirical source for the airline model. The paper's full PDF is linked rather than redistributed here.

## Book recommended for the study path

Dimitri P. Bertsekas, *Dynamic Programming and Optimal Control*. The book was recommended as a learning resource. This README does not specify an edition or chapter beyond what is recorded in the project notes.

## Flight-price data source as recorded by the notebook

The flight-price notebook says it collected fare records from Yatra using Selenium-based web scraping. The workbook `Flight_Prices.xlsx` is not included. The recorded model scores should therefore be treated as notebook outputs rather than independently reproduced results. Check relevant site terms before any new scraping or redistribution.
