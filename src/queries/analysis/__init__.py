"""
Query Analysis Module.

This module provides functions for analyzing queries in Bayesian Networks, including
structural properties, computational complexity, and optimization analysis.

Main Functions:
    Structural Properties:
    - compute_query_structural_properties() - Comprehensive structural analysis
    - compute_markov_blanket_metrics() - Markov blanket size metrics
    - compute_distance_metrics() - Distance metrics between node sets
    - compute_topological_properties() - Topological properties of nodes

    Computational Complexity:
    - compute_query_complexity() - Variable elimination complexity analysis
    - identify_barren_nodes() - Find nodes that can be removed from queries
    - find_conditionally_independent_vars() - Find variables independent of queries

Example:
    >>> from src.queries.analysis import (
    ...     compute_query_structural_properties,
    ...     compute_query_complexity,
    ... )
    >>> # Structural analysis
    >>> properties = compute_query_structural_properties(
    ...     bn=bayesian_network,
    ...     target_nodes=["Disease"],
    ...     evidence_nodes=["Symptom1", "Symptom2"],
    ... )
    >>> print(f"Target MB size: {properties['avg_markov_blanket_size_target']}")
    >>> # Complexity analysis
    >>> complexity = compute_query_complexity(
    ...     bn=bayesian_network,
    ...     target_nodes=["Disease"],
    ...     evidence_nodes=["Symptom1", "Symptom2"],
    ... )
    >>> print(f"Induced width: {complexity.induced_width}")
    >>> print(f"Total cost: {complexity.total_cost:,}")
"""

# Structural properties
# Computational complexity
from ..complexity import (
    ComplexityMetrics,
    compute_query_complexity,
)
from ..complexity.node_analysis import (
    find_conditionally_independent_vars,
    identify_barren_nodes,
)
from .analysis import (
    compute_distance_metrics,
    compute_markov_blanket_metrics,
    compute_query_structural_properties,
    compute_topological_properties,
)

__all__ = [
    # Structural properties
    "compute_query_structural_properties",
    "compute_markov_blanket_metrics",
    "compute_distance_metrics",
    "compute_topological_properties",
    # Computational complexity
    "compute_query_complexity",
    "ComplexityMetrics",
    "identify_barren_nodes",
    "find_conditionally_independent_vars",
]
