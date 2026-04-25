# Naming Variants

Generates alternative variable naming strategies for Bayesian networks to test LLM robustness.

## Module

- **`naming_variants`** — Creates name mappings and renamed network variants.

## Main Functions

- `create_name_mapping_from_strategy(nodes, strategy)` — Generate a name mapping for a given strategy.
- `create_bn_naming_variant(bn, naming_strategy, ...)` — Create a renamed Bayesian network variant.
- `create_dag_naming_variant(dag, naming_strategy)` — Create a renamed DAG variant.

## Naming Strategies

- `simple` — V0, V1, V2, ... (default used in the benchmark)
- `descriptive` — Human-readable descriptive names
