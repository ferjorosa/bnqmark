# Experiment Runner

Orchestrates LLM evaluation runs on Bayesian network inference queries.

## Modules

- **`core`** — Main experiment logic and LLM interaction.
- **`sequential`** — Sequential execution of experiments.
- **`parallel`** — Parallel execution with multiprocessing.
- **`batching`** — Query batching utilities for efficient processing.

## Main Functions

- `run_single_query(query, model_name, ...)` — Run a single experiment and return results.
- `run_discrete_queries_sequential(queries, ...)` — Run experiments sequentially.
- `run_discrete_queries_parallel(queries, ...)` — Run experiments in parallel.
- `create_query_batches(queries, batch_size)` — Split queries into batches for processing.
