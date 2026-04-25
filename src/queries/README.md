# Queries

Generate, format, and analyze conditional probability queries for Bayesian network inference tasks.

## Modules

- **`generation`** — Low-level query sampling and constraint enforcement.
- **`sweep`** — Dataset-level query generation across target/evidence configurations.
- **`analysis`** — Structural query metrics such as distances and Markov blanket sizes.
- **`complexity`** — Inference complexity estimates and elimination analysis.
- **`formatting`** — Query string formatting for prompts and outputs.

## Main Functions

- `generate_single_query(...)` — Generate one query satisfying node, evidence, and distance constraints.
- `generate_queries_with_sampling(...)` — Generate benchmark query sweeps with informativeness filtering.
- `compute_query_structural_properties(...)` — Compute structural metadata for a query.
- `format_query_str(...)` — Format a query as text for LLM prompts.
