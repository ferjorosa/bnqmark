# Query Sweeps

High-level query generation across systematic parameter sweeps. This module is used to build balanced query datasets from generated Bayesian networks.

## Modules

- **`sweep`** — Sampling loops, threshold filtering, and query combination coverage.

## Main Functions

- `generate_queries_with_sampling(...)` — Generate queries across target/evidence combinations while enforcing a posterior-prior threshold.
- `sample_query_with_threshold(...)` — Sample one query that satisfies the informativeness threshold.
- `compute_query_probabilities(...)` — Compute exact posterior and prior probabilities for a query.

## Notes

- Covers combinations of target count, evidence count, and distance bucket.
- Applies informativeness filtering during sampling rather than post-filtering.
- Supports strict and lenient handling when valid queries cannot be sampled.
