"""
DAG Generation Module.

This module provides functionality for generating directed acyclic graphs (DAGs)
with controllable treewidth for probabilistic reasoning experiments.

Main Function:
    generate_single_dag() - Generate a single DAG with target treewidth

Example:
    >>> from src.dag import generate_single_dag
    >>> # Generate a single DAG
    >>> dag, treewidth, meta = generate_single_dag(
    ...     n_nodes=10, target_treewidth=3, node_naming="simple", seed=42
    ... )
    >>> print(f"Generated DAG with {dag.number_of_nodes()} nodes")
    >>> print(f"Achieved treewidth: {treewidth}")

Advanced Usage:
    For fine-grained control, import from submodules:
    >>> from src.dag.generation.core import (
    ...     generate_graph_with_target_treewidth,
    ...     undirected_to_dag,
    ... )
"""

# Main entry point - use this!
# Re-export key types for type hints and advanced usage
from .core.types import (
    DAGGenerationMetadata,
    NamingStrategy,
)
from .generator import generate_single_dag

__all__ = [
    # Main API - use this function!
    "generate_single_dag",
    # Types (for type hints)
    "DAGGenerationMetadata",
    "NamingStrategy",
]
