# Query Generation and Analysis

Generates and analyzes probabilistic queries from Bayesian Networks for evaluating LLM probabilistic reasoning capabilities.

## Modules

- **`generation`** – Generates diverse queries from Bayesian Networks.
- **`analysis`** – Analyzes query structural properties and difficulty metrics.
- **`sweep`** – Generates queries with parameter sweeps and threshold filtering.

## Main Functions

- `generate_queries(...)` – Generate diverse queries with configurable query node counts, evidence counts, and distance buckets.
- `compute_query_structural_properties(...)` – Compute structural properties like Markov blanket sizes and target-evidence distances.
- `generate_queries_with_sampling(...)` – Generate queries with threshold filtering on posterior-prior differences.
