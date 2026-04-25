# Utilities

Shared utility functions used across the project.

## Modules

- **`llm`** — LLM API interaction and response parsing.
  - `run_llm` — Execute LLM calls via OpenRouter.
  - `parser_basic` — Basic response parsing.
  - `parser_details` — Detailed response parsing with metadata.
- **`code_execution_utils`** — Safe code execution for the code generation protocol.
- **`distance_utils`** — Graph distance calculations for query analysis.
- **`error_utils`** — Error classification (e.g., token limit detection).
- **`pydantic_parser`** — Structured output parsing with Pydantic.
- **`tiktoken_utils`** — Token counting for prompts.
- **`yaml_utils`** — YAML configuration loading.

## Main Functions

- `run_llm_call(prompt, model, ...)` — Execute an LLM API call.
- `execute_and_extract_probability(code_str, ...)` — Safely execute generated code.
- `count_input_tokens(prompt)` — Count tokens in a prompt.
- `compute_shortest_distance(graph, node1, node2)` — Compute graph distances.
