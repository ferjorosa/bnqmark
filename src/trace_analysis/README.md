# Trace Analysis

Analyzes LLM reasoning traces to extract structured insights about model behavior.

## Modules

- **`core`** — Main analysis logic and LLM interaction for trace parsing.
- **`sequential`** — Sequential execution of trace analyses.
- **`parallel`** — Parallel execution with multiprocessing.
- **`batching`** — Experiment batching utilities.
- **`pydantic_models`** — Structured output schemas for different analysis types.

## Main Functions

- `run_single_analysis(experiment, analysis_type, ...)` — Analyze a single experiment trace.
- `run_trace_analysis_sequential(experiments, ...)` — Run analyses sequentially.
- `run_trace_analysis_parallel(experiments, ...)` — Run analyses in parallel.

## Analysis Types

- `raw_reasoning_arithmetic` — Check arithmetic correctness in raw reasoning.
- `raw_reasoning_inference_algorithm` — Identify inference algorithms used.
- `code_generation_strategy` — Analyze code generation approach.
- `code_generation_behaviour` — Analyze code execution behavior.
