---
license: mit
pretty_name: BNqMark-20
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- bayesian-networks
- probabilistic-inference
- large-language-models
- reasoning
- benchmark
- code-generation
size_categories:
- 1K<n<10K
configs:
- config_name: bns
  data_files:
  - split: train
    path: data/bns/train.parquet
- config_name: queries
  data_files:
  - split: train
    path: data/queries/train.parquet
- config_name: experiments
  data_files:
  - split: train
    path: data/experiments/train.parquet
---

# BNqMark-20

BNqMark-20 is a benchmark dataset for evaluating Large Language Models (LLMs) on exact probabilistic inference in discrete Bayesian Networks. It isolates probabilistic computation from linguistic interpretation by giving models complete conditional probability table (CPT) specifications and asking them to answer conditional probability queries.

The dataset includes 78 Bayesian networks with 4-20 binary variables, 434 conditional probability queries, and 7,812 LLM evaluation results from 9 frontier models. Experiments cover two protocols:
- **raw reasoning**, where the model computes the probability directly from the CPTs
- **code generation**, where the model writes Python code to solve the same inference problem

## Dataset Structure

The dataset is organized into three separate configurations (configs), each containing a different type of data:

### 1. `bns` - Bayesian Network Configurations (78 rows)

Each row represents a Bayesian network configuration with its structural and distributional properties.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Database row identifier |
| `bn_uuid` | string | Unique identifier for the Bayesian network |
| `n` | int64 | Network size parameter (number of nodes, 4-20) |
| `target_tw` | int64 | Target treewidth for generation |
| `achieved_tw` | int64 | Actual achieved treewidth (may vary from target) |
| `arity` | string | Variable arity specification (e.g., "fixed:2" for binary) |
| `alpha` | float64 | Dirichlet concentration parameter (1.0 = uniform, 0.5 = skewed) |
| `determinism` | float64 | Degree of determinism in CPTs |
| `variant_index` | int64 | Variant index for this parameter combination (0 or 1) |
| `num_edges` | int64 | Number of edges in the network |
| `num_nodes` | int64 | Number of nodes (same as `n`) |
| `base_sample_counter` | int64 | Base counter for sampling CPTs |
| `seed` | int64 | Random seed used for reproducible generation |
| `edge_density` | float64 | Edge density (num_edges / possible_edges) |
| `avg_markov_blanket_size` | float64 | Average Markov blanket size across all nodes |
| `naming_strategy` | string | Variable naming strategy used (`"simple"` in this release) |
| `bn_pickle` | binary | Serialized pgmpy DiscreteBayesianNetwork object |
| `dag_pickle` | binary | Serialized NetworkX DiGraph structure |
| `created_at` | timestamp | Timestamp when the record was created |

### 2. `queries` - Inference Queries (434 rows)

Each row represents a conditional probability query `P(Q=q | E=e)` with ground truth values.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Database row identifier |
| `bn_uuid` | string | Reference to the parent Bayesian network |
| `query_uuid` | string | Unique identifier for the query |
| `naming_strategy` | string | Variable naming strategy used (`"simple"` in this release) |
| `target` | string | Target variable(s) and value(s) as JSON (e.g., `{"V3": "s1"}`) |
| `evidence` | string | Evidence variable(s) and value(s) as JSON (e.g., `{"V1": "s0"}`) |
| `num_evidence` | int64 | Number of evidence variables (1 or 2) |
| `num_target` | int64 | Number of target variables (1 or 2) |
| `probability` | float64 | Ground truth conditional probability |
| `prior_probability` | float64 | Prior probability `P(Q=q)` without conditioning |
| `induced_width` | int64 | Induced width during variable elimination |
| `num_eliminated` | int64 | Number of variables eliminated to answer the query |
| `avg_markov_blanket_size_target` | float64 | Average Markov blanket size of target nodes |
| `avg_markov_blanket_size_evidence` | float64 | Average Markov blanket size of evidence nodes |
| `min_distance_target_evidence` | int64 | Minimum undirected distance between any target and evidence node (1, 2, or 3) |
| `min_distance_target_target` | float64 | Minimum distance between target nodes (null if single target) |
| `min_distance_evidence_evidence` | float64 | Minimum distance between evidence nodes (null if single evidence) |
| `evidence_distances` | string | List of distances as JSON string |
| `all_target_are_roots` | bool | Whether all target variables are root nodes |
| `all_target_are_leaves` | bool | Whether all target variables are leaf nodes |
| `all_evidence_are_roots` | bool | Whether all evidence variables are root nodes |
| `all_evidence_are_leaves` | bool | Whether all evidence variables are leaf nodes |
| `num_target_nodes` | int64 | Number of distinct target nodes |
| `num_evidence_nodes` | int64 | Number of distinct evidence nodes |
| `target_nodes` | string | List of target node names as JSON |
| `evidence_nodes` | string | List of evidence node names as JSON |
| `abs_diff` | float64 | Absolute difference `|P(Q|E) - P(Q)|` (informativeness measure) |
| `rel_diff` | float64 | Relative difference (informativeness measure) |
| `avg_distance_target_evidence` | float64 | Average distance between target and evidence nodes |
| `created_at` | timestamp | Timestamp when the record was created |

### 3. `experiments` - LLM Evaluation Results (7,812 rows)

Each row represents a single LLM evaluation run on a query.

| Column | Type | Description |
|--------|------|-------------|
| `query_uuid` | string | Reference to the query being evaluated |
| `naming_strategy` | string | Naming strategy used for the query (`"simple"` in this release) |
| `run` | int64 | Run number (1-indexed) |
| `experiment_type` | string | Protocol type: "raw_reasoning" or "code_generation" |
| `full_prompt` | string | Complete prompt sent to the LLM including CPTs and query |
| `response` | string | LLM's raw response text |
| `response_reasoning_summary` | string | Reasoning summary provided by model (if available) |
| `response_metadata` | string | JSON metadata about the response |
| `model_name` | string | Full model identifier (e.g., "openai/gpt-5.4") |
| `reasoning_model` | bool | Whether this is a reasoning-capable model |
| `openai_reasoning_effort` | string | OpenAI reasoning effort level ("low", "medium", "high") |
| `openai_reasoning_summary` | string | OpenAI reasoning summary level ("auto", "concise", "detailed") |
| `input_tokens` | int64 | Number of input tokens consumed |
| `output_tokens` | int64 | Number of output tokens generated |
| `usage_metadata` | string | JSON metadata about token usage |
| `temperature` | float64 | Temperature parameter used for generation |
| `llm_probability` | float64 | Extracted probability from LLM response (if parseable) |
| `code_followed_formatting_instructions` | bool | (Code gen only) Whether code used proper `<code>` tags |
| `code_pgmpy_library_fix` | bool | (Code gen only) Whether pgmpy API fix was applied |
| `code_exception_type` | string | (Code gen only) Exception type if code execution failed |
| `code_exception_message` | string | (Code gen only) Exception message if code execution failed |
| `started_at` | timestamp | When the LLM query started |
| `finished_at` | timestamp | When the LLM query finished |
| `created_at` | timestamp | When the record was created in database |

## Usage

```python
from datasets import load_dataset

# Load each configuration separately
bns = load_dataset("ferjorosa/bnqmark-20", "bns")["train"]
queries = load_dataset("ferjorosa/bnqmark-20", "queries")["train"]
experiments = load_dataset("ferjorosa/bnqmark-20", "experiments")["train"]

# Example: Get a specific Bayesian network
bn = bns[0]
print(f"Network {bn['bn_uuid']}: {bn['num_nodes']} nodes, tw={bn['achieved_tw']}")

# Example: Find queries for a specific network
import pandas as pd
queries_df = pd.DataFrame(queries)
network_queries = queries_df[queries_df["bn_uuid"] == bn["bn_uuid"]]

# Example: Get experiment results for a specific model
experiments_df = pd.DataFrame(experiments)
gpt_results = experiments_df[experiments_df["model_name"] == "openai/gpt-5.4"]
```

## Dataset Statistics

### Bayesian Networks
- **Total configurations**: 78 Bayesian networks using the `simple` naming strategy
- **Network sizes (n)**: 4, 6, 8, 10, 12, 14, 16, 18, 20 nodes
- **Treewidth range**: 2-12 (achieved)
- **Alpha values**: 1.0 (uniform), 0.5 (skewed)

### Queries
- **Total query records**: 434 benchmark queries using the `simple` naming strategy
- **Query configurations**:
  - Target cardinality: 1 or 2 variables
  - Evidence cardinality: 1 or 2 variables
  - Distance strata: min 1, 2, or 3 edges between target and evidence
- **Informativeness threshold**: `|P(Q|E) - P(Q)| >= 0.1`

### Experiments
- **Models evaluated**: 9 frontier LLMs via [OpenRouter](https://openrouter.ai/)
  - [openai/gpt-5.4](https://openrouter.ai/openai/gpt-5.4)
  - [x-ai/grok-4.20](https://openrouter.ai/x-ai/grok-4.20)
  - [anthropic/claude-sonnet-4.6](https://openrouter.ai/anthropic/claude-sonnet-4.6)
  - [google/gemini-3.1-pro-preview](https://openrouter.ai/google/gemini-3.1-pro-preview)
  - [qwen/qwen3-max-thinking](https://openrouter.ai/qwen/qwen3-max-thinking)
  - [moonshotai/kimi-k2.5](https://openrouter.ai/moonshotai/kimi-k2.5)
  - [deepseek/deepseek-v3.2-speciale](https://openrouter.ai/deepseek/deepseek-v3.2-speciale)
  - [z-ai/glm-5](https://openrouter.ai/z-ai/glm-5)
  - [minimax/minimax-m2.7](https://openrouter.ai/minimax/minimax-m2.7)
- **API Provider**: All models accessed through [OpenRouter](https://openrouter.ai/) with temperature=1.0 and maximum reasoning effort
- **Protocols**: raw_reasoning, code_generation

## Citation

```bibtex
@dataset{bnqmark_20,
  title={BNqMark-20 Dataset},
  author={Fernando Rodriguez and Bojan Mihaljevic},
  year={2026},
  url={https://huggingface.co/datasets/ferjorosa/bnqmark-20},
  note={Benchmark dataset for exact probabilistic inference in discrete Bayesian networks}
}
```

## License

This dataset is released under the MIT License.
