# Experiments

Scripts for running the full BNqMark experimental pipeline: data generation, LLM evaluation, and result analysis.

## Subdirectories

### `main/` — LLM Inference Experiments

Run Large Language Model evaluation experiments on Bayesian network inference queries. Orchestrates parallel API calls to OpenRouter, handles responses, extracts probabilities from both raw reasoning and code generation outputs, and stores results in the database.

### `generate_data/` — Dataset Generation

Generate the complete BNqMark dataset: Bayesian networks with controlled treewidth (4-20 variables, treewidth 2-12), conditional probability queries with informativeness filtering, and alternative naming variants for robustness testing.

### `export_data/` — Data Export

Export database tables to Parquet format for the HuggingFace dataset release.

### `result_analysis/` — Visualization and Analysis

Generate publication-quality plots (accuracy, answerability, and MAE heatmaps) and summary tables from experiment results.

### `trace_analysis/` — Reasoning Trace Analysis

⚠️ **Not part of the current paper.** ⚠️

Analyze LLM reasoning traces to extract behavioral insights (arithmetic correctness, inference algorithms, code strategies). Requires models that expose reasoning tokens, which closed-source APIs (GPT-5.4, Gemini, Claude, Grok) do not provide.

## Typical Workflow

1. `generate_data/generate_bn_dataset.py` — Generate Bayesian networks
2. `generate_data/generate_query_dataset.py` — Generate inference queries
3. `main/run_experiments.py` — Run LLM evaluation experiments
4. `result_analysis/plot_*.py` — Generate result visualizations
5. `export_data/export_experiments_to_parquet.py` — Export for HuggingFace
