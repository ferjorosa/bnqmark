# Bayesian Network Generation and Analysis

Generates discrete Bayesian Networks with controllable structural and distributional properties for evaluating LLM probabilistic reasoning.

## Modules

- **`sweep`** – Main submodule. Generates multiple BNs with systematic parameter sweeps.
- **`generation`** – Lower-level single-network generation.
- **`evaluation`** – Validation helpers for naming variants.
- **`analysis`** – Visualization and structural metrics.

## Main Function

- `generate_bayesian_networks_and_metadata(ns, treewidths, arity_specs, ...)` – Primary entry point for BN dataset generation.
