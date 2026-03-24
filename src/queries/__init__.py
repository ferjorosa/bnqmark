"""
Queries Module.

This module provides functionality for generating and analyzing probabilistic queries
from Bayesian Networks for evaluating LLM probabilistic reasoning capabilities.

The key insight is that different query characteristics (number of variables,
evidence distance, etc.) create different reasoning challenges for LLMs, making
systematic query generation crucial for comprehensive evaluation.

Main Functions (start here):
    generate_single_query() - Generate ONE query with constraints
    generate_queries_with_sampling() - Generate MANY queries with parameter sweeps
    compute_query_structural_properties() - Analyze query structural properties

Example (Single Query):
    >>> from src.queries import generate_single_query
    >>> # Generate one query with strict constraints
    >>> query = generate_single_query(
    ...     model=my_bayesian_network,
    ...     query_node_count=1,
    ...     evidence_count=2,
    ...     distance_bucket=(3, 100),
    ...     min_abs_diff=0.1,
    ...     seed=42,
    ... )

Example (Multiple Queries with Sweeps):
    >>> from src.queries import generate_queries_with_sampling, QuerySpec
    >>> # Generate queries with parameter sweeps
    >>> queries = generate_queries_with_sampling(
    ...     bn=my_bayesian_network,
    ...     queries_per_combination=5,
    ...     query_node_counts=(1, 2),
    ...     evidence_counts=(1, 2, 3),
    ...     distance_buckets=[(3, 100)],
    ...     min_abs_diff=0.1,
    ...     seed=42,
    ... )

Advanced Usage:
    For dataset-level query generation, see the scripts in:
    experiments/discrete/
"""

# Generation API
# Analysis API
from .analysis import (
    compute_distance_metrics,
    compute_markov_blanket_metrics,
    compute_query_structural_properties,
    compute_topological_properties,
)
from .generation import (
    QueryGenerationMetadata,
    QuerySpec,
    generate_single_query,
)

# Sweep API
from .sweep import generate_queries

__all__ = [
    # Generation API
    "generate_single_query",
    "QuerySpec",
    "QueryGenerationMetadata",
    # Analysis API
    "compute_query_structural_properties",
    "compute_markov_blanket_metrics",
    "compute_distance_metrics",
    "compute_topological_properties",
    # Sweep API
    "generate_queries",
]
