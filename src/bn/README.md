# Bayesian Networks

Generate, evaluate, and analyze discrete Bayesian Networks with controllable structural and distributional properties.

## Modules

- **`generation`** — Single-network generation from DAGs and CPD settings.
- **`sweep`** — Dataset-level generation across network sizes, treewidths, arities, and CPT parameters.
- **`evaluation`** — Validation helpers for generated networks and naming variants.
- **`analysis`** — Structural metrics and visualization utilities.

## Main Functions

- `generate_single_bayesian_network(...)` — Generate one Bayesian network.
- `generate_bayesian_networks_and_metadata(...)` — Generate benchmark network sweeps with metadata.
