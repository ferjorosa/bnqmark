# Query Generation

This module provides the core functionality for generating probabilistic queries from Bayesian networks with strict constraint enforcement.

## Overview

The query generation system creates queries by selecting target nodes, evidence nodes, and their states, while enforcing distance and probability constraints to ensure meaningful and diverse queries for LLM evaluation.

## Core Function

### `generate_single_query()`

Generates a single query meeting all specified constraints:

```python
from src.queries.generation import generate_single_query

query = generate_single_query(
    model=bayesian_network,
    query_node_count=2,           # Number of target variables
    evidence_count=1,             # Number of evidence variables
    distance_bucket=(3, 5),       # Evidence distance constraints
    min_abs_diff=0.1,            # Minimum |posterior - prior|
    max_tries=500,               # Maximum sampling attempts
    seed=42                      # For reproducibility
)
```

## Distance Constraints

**Key Behavior**: Evidence nodes must satisfy distance constraints from **ALL** query nodes simultaneously.

### Example
- Query nodes: `[A, B]`
- Distance constraint: `(3, 5)`
- Evidence candidate: `X`

**Requirements**:
- Distance from `A` to `X` must be ∈ [3, 5] **AND**
- Distance from `B` to `X` must be ∈ [3, 5]

If either distance violates the constraint, candidate `X` is rejected.

### Why This Matters
This ensures that the final minimum distance will always be ≥ `dmin`, preventing constraint violations that could occur if evidence nodes were selected based on proximity to only one query node.

## Constraint Enforcement

The generator enforces constraints in order of computational cost:

1. **Distance constraints** (always enforced) - Pre-filters evidence candidates
2. **Probability threshold** (optional) - Computes exact probabilities via variable elimination
3. **Uniqueness** (optional) - Prevents duplicate queries

If any constraint fails, the query is rejected and generation retries up to `max_tries`.

## Architecture

- **`generator.py`** - Main generation logic with constraint enforcement
- **`types.py`** - Type definitions (`QuerySpec`, `QueryGenerationContext`, etc.)

## Usage Patterns

**Single Query**: Use `generate_single_query()` directly for one-off generation.

**Multiple Queries**: Use the sweep module (`src.queries.sweep`) which orchestrates multiple calls to `generate_single_query()` across parameter combinations.

## Key Features

- **Strict constraint enforcement** - No fallback mechanisms, constraints are always honored
- **Efficient pre-filtering** - Distance constraints filter evidence candidates before expensive probability computation
- **Exact probability computation** - Uses pgmpy's VariableElimination for ground truth probabilities
- **Reproducible** - Deterministic results with seed parameter
- **Modular design** - Clean separation between generation, constraint checking, and metadata computation
