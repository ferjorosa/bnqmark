"""
Query Complexity Analysis for Bayesian Networks.

This package provides functions for analyzing the computational complexity of
probabilistic queries in Bayesian Networks, including variable elimination
complexity, barren node identification, and conditional independence analysis.

The main function compute_query_complexity() performs a comprehensive analysis
of query difficulty by:
1. Removing conditionally independent variables
2. Identifying and removing barren nodes
3. Computing variable elimination complexity on the reduced network

This provides accurate complexity estimates for exact inference algorithms.

Main API:
    - compute_query_complexity: Compute comprehensive complexity metrics
    - ComplexityMetrics: Dataclass containing all complexity information

Example:
    >>> from src.queries.complexity import compute_query_complexity
    >>> complexity = compute_query_complexity(
    ...     bn=bayesian_network,
    ...     target_nodes=["Disease"],
    ...     evidence_nodes=["Symptom1", "Symptom2"],
    ...     verbose=True,
    ... )
    >>> print(f"Induced width: {complexity.induced_width}")
    >>> print(f"Total cost: {complexity.total_cost:,}")
"""

from .complexity import compute_query_complexity
from .types import ComplexityMetrics

__all__ = [
    "compute_query_complexity",
    "ComplexityMetrics",
]
