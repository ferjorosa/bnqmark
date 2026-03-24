# Query Sweep Module

This module provides high-level functions for generating queries from Bayesian Networks with systematic parameter sweeps and threshold-based sampling.

## Key Features

- **Exhaustive combination coverage**: Ensures balanced representation of all query configurations
- **Threshold filtering during generation**: More efficient than post-filtering
- **Configurable failure handling**: Strict mode (raise exception) or lenient mode (skip and continue)
- **Reproducible**: Uses random seeds for deterministic generation

## Main Function

### `generate_queries_with_sampling()`

Generate queries with automatic sampling until a probability difference threshold is met.

```python
from src.queries.sweep import generate_queries_with_sampling

queries = generate_queries_with_sampling(
    bn=my_bayesian_network,
    queries_per_combination=5,       # 5 instances of each combo
    query_node_counts=(1, 2),        # 2 options
    evidence_counts=(1, 2, 3),       # 3 options
    # → Total: 2×3 = 6 combinations × 5 = 30 queries

    distance_buckets=[(1, 100)],     # Single broad bucket
    min_abs_diff=0.1,                # Threshold: |posterior - prior| >= 0.1
    max_tries_per_query=200,         # Max sampling attempts
    strict_mode=False,               # Continue on failure
    seed=42,
)

print(f"Generated {len(queries)} queries (all meeting threshold)")
```

## How It Works

1. **Create combinations**: All combinations of `(query_node_count, evidence_count)` are generated
   - Example: `(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)`

2. **Sample for each combination**: For each combination, generate N instances by sampling
   - Samples up to `max_tries_per_query` times per instance
   - Keeps only queries where `|posterior - prior| >= min_abs_diff`
   - No post-filtering needed!

3. **Handle failures gracefully**:
   - `strict_mode=True`: Raises exception if any query fails
   - `strict_mode=False`: Logs warning and continues (returns fewer queries)

## Helper Functions

### `sample_query_with_threshold()`

Low-level function to sample a single query meeting the threshold.

```python
from src.queries.sweep import sample_query_with_threshold
import numpy as np

rng = np.random.default_rng(42)
query = sample_query_with_threshold(
    bn=my_bn,
    query_node_count=2,
    evidence_count=1,
    distance_buckets=[(1, 100)],
    min_abs_diff=0.1,
    max_tries=200,
    rng=rng,
)
```

### `compute_query_probabilities()`

Compute exact posterior and prior probabilities for a query.

```python
from src.queries.sweep import compute_query_probabilities

posterior, prior = compute_query_probabilities(bn, query)
abs_diff = abs(posterior - prior)
```

## Comparison with Basic Generation

**Basic approach** (from `generation` module):
```python
from src.queries import generate_queries

# Generates 30 random queries (no threshold)
queries = generate_queries(
    model=bn,
    num_queries=30,
    query_node_counts=[1, 2],
    evidence_counts=[1, 2, 3],
    seed=42,
)

# Need to filter afterwards (wasteful if many don't meet threshold)
filtered = [q for q in queries if meets_threshold(q)]
```

**Sweep approach** (this module):
```python
from src.queries.sweep import generate_queries_with_sampling

# Generates queries meeting threshold (no post-filtering needed)
queries = generate_queries_with_sampling(
    bn=bn,
    queries_per_combination=5,
    query_node_counts=(1, 2),
    evidence_counts=(1, 2, 3),
    min_abs_diff=0.1,
    max_tries_per_query=200,
    seed=42,
)

# All queries already meet threshold!
```

## Use Cases

### Dataset Generation

Use this for generating balanced query datasets:

```python
# See: experiments/discrete/generate_query_dataset.py
for bn in bayesian_networks:
    queries = generate_queries_with_sampling(
        bn=bn,
        queries_per_combination=5,
        query_node_counts=(1, 2),
        evidence_counts=(1, 2, 3),
        min_abs_diff=0.1,
        seed=base_seed + bn_idx,
    )
```

### Experiment Sweeps

Use for systematic parameter sweeps:

```python
for min_diff in [0.05, 0.10, 0.15, 0.20]:
    queries = generate_queries_with_sampling(
        bn=bn,
        min_abs_diff=min_diff,
        queries_per_combination=10,
        seed=42,
    )
    print(f"Threshold {min_diff}: {len(queries)} queries")
```

## Design Philosophy

This module mirrors the design of `src/bn/sweep/sweep.py`:
- High-level functions for systematic parameter sweeps
- Efficient generation with constraints
- Reusable across different experiments
- Clean separation from low-level generation logic
