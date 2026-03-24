"""
DAG Module.

This module provides functionality for generating and analyzing
directed acyclic graphs (DAGs) with controllable treewidth for
probabilistic reasoning experiments.

The key insight is that treewidth correlates with exact inference hardness in
Bayesian networks, making it a crucial parameter for systematic evaluation of
computational complexity and model performance.

Main Functions (start here):
    generate_single_dag() - Generate a single DAG with target treewidth
    analyze_graph_properties() - Analyze graph structural properties
    verify_dag_properties() - Verify DAG meets expected properties

Example:
    >>> from src.dag import generate_single_dag, analyze_graph_properties
    >>> # Generate a single DAG
    >>> dag, treewidth, meta = generate_single_dag(
    ...     n_nodes=10, target_treewidth=3, node_naming="simple", seed=42
    ... )
    >>> print(f"Generated DAG with {dag.number_of_nodes()} nodes")
    >>> print(f"Achieved treewidth: {treewidth}")

    >>> # Analyze the DAG
    >>> props = analyze_graph_properties(dag)
    >>> print(
    ...     f"Density: {props['density']:.2f}, Topological levels:
    ...     {props['topological_levels']}"
    ... )

Advanced Usage:
    For fine-grained control, import from submodules:
    >>> from src.dag.generation.core import (
    ...     generate_graph_with_target_treewidth,
    ...     undirected_to_dag,
    ... )
"""

# Generation API
# Analysis API
from .analysis import (
    analyze_graph_properties,
    verify_dag_properties,
)
from .generation import (
    DAGGenerationMetadata,
    NamingStrategy,
    generate_single_dag,
)

__all__ = [
    # Generation API
    "generate_single_dag",
    "DAGGenerationMetadata",
    "NamingStrategy",
    # Analysis API
    "analyze_graph_properties",
    "verify_dag_properties",
]
