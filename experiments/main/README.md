# Main Experiments

Run LLM inference experiments on Bayesian network queries. This is the core evaluation pipeline that tests Large Language Models on exact probabilistic inference tasks. The experiment runner orchestrates parallel API calls to OpenRouter, evaluates LLMs under two protocols (raw reasoning and code generation), extracts numerical probabilities from responses, and stores structured results in the database.

## Scripts

### `run_experiments.py`

Main experiment runner for evaluating LLMs on Bayesian network queries.

**Key functionality:**
- Loads query datasets and prompt templates from `config/experiments.yaml`
- Runs parallel LLM calls across multiple models with rate limiting
- Handles token limit errors with automatic retry logic
- Extracts probabilities using regex parsing for raw reasoning responses
- Delegates code execution to sandbox for code generation responses
- Stores results with full prompts, responses, metadata, and extracted probabilities

**Main function:**
- `main()` — Load configuration, run experiments across configured models

**Configuration:** Experiments are configured via `config/experiments.yaml` with model lists, prompt templates, and batch settings.

### `run_code_execution_and_extract_probability.py`

Standalone script for executing previously generated code and extracting probabilities.

**Use case:** Re-run code generation outputs when the main experiment only captured the generated code but not its execution result (e.g., due to timeout or sandbox issues).

**Main function:**
- `main()` — Execute code from database and update with extracted probabilities
