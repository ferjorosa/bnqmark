# Query Generation

Low-level query generation for Bayesian networks. This module samples target nodes, evidence nodes, and states while enforcing distance, uniqueness, and probability constraints.

## Modules

- **`generator`** — Query sampling and constraint checks.
- **`types`** — Type definitions such as `QuerySpec` and generation context objects.

## Main Functions

- `generate_single_query(...)` — Generate one query satisfying the requested constraints.
- `generate_queries(...)` — Generate multiple queries with configurable target/evidence counts and distance buckets.

## Notes

- Evidence nodes must satisfy distance constraints from all target nodes.
- Probability thresholds are computed with exact inference when enabled.
- Use `src.queries.sweep` for full benchmark query generation across parameter combinations.
