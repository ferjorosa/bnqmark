"""
Main entry point for DAG generation.

This module provides the high-level API for generating directed acyclic graphs
with controlled treewidth for probabilistic reasoning experiments.

The key function is generate_single_dag() which creates a DAG with approximate
target treewidth using specified node naming and conversion methods.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.approximation import treewidth

from src.dag.generation.core.generation import (
    generate_graph_with_target_treewidth,
    undirected_to_dag,
)
from src.dag.generation.core.naming import generate_node_names, relabel_graph_nodes
from src.dag.generation.core.types import DAGGenerationMetadata, NamingStrategy


def generate_single_dag(
    n_nodes: int,
    target_treewidth: int,
    dag_method: str = "random",
    max_iterations: int = 1000,
    node_naming: NamingStrategy = NamingStrategy.SIMPLE,
    seed: int | None = None,
) -> tuple[nx.DiGraph, int, dict[str, Any]]:
    """
    Generate a single DAG with approximately the target treewidth.

    This is the main entry point for DAG generation. It combines undirected
    graph generation (with target treewidth) and DAG conversion to produce
    directed acyclic graphs suitable for Bayesian network experiments.

    **Recommended Method Combination:**
    - dag_method='random': Preserves treewidth, diverse DAG structure
    - dag_method='topological': Preserves treewidth, ordered structure

    **WARNING:** Avoid dag_method='bfs' or 'dfs' as they reduce treewidth to 1!

    Args:
        n_nodes: Number of nodes in the final DAG
        target_treewidth: Desired treewidth (approximate)
                         Controls inference complexity - higher = harder
        dag_method: DAG conversion method ('random', 'topological', 'bfs', 'dfs')
                   Default: 'random' (recommended for treewidth preservation)
        max_iterations: Maximum iterations for treewidth search (default: 1000)
        node_naming: Node naming strategy (NamingStrategy enum)
                    Default: NamingStrategy.SIMPLE (V0, V1, V2, ...)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dag, achieved_treewidth, metadata_dict) where:
        - dag: NetworkX DiGraph (the generated DAG)
        - achieved_treewidth: Final treewidth of the DAG's underlying undirected graph
        - metadata_dict: Dictionary with generation details and statistics

    Raises:
        ValueError: If target_treewidth >= n_nodes or invalid method names

    Example:
        >>> # Generate DAG with approximate treewidth 3 (recommended)
        >>> dag, tw, meta = generate_single_dag(
        ...     n_nodes=10, target_treewidth=3, dag_method="random", seed=42
        ... )
        >>> print(f"Target: 3, Achieved: {tw}")
        Target: 3, Achieved: 3

        >>> # Generate with confusing node names to test robustness
        >>> dag, tw, meta = generate_single_dag(
        ...     n_nodes=8,
        ...     target_treewidth=2,
        ...     node_naming=NamingStrategy.CONFUSING,
        ...     seed=123,
        ... )
        >>> print(list(dag.nodes())[:3])
        ['X_7a4f2b', 'Q_9c1e8d', 'Z_3b6a9f']

        >>> # Access metadata
        >>> print(f"Exact match: {meta['exact_treewidth']}")
        >>> print(f"Number of edges: {meta['dag_edges']}")
    """
    if target_treewidth >= n_nodes:
        raise ValueError(
            f"target_treewidth ({target_treewidth}) must be less than "
            f"n_nodes ({n_nodes})",
        )

    # Generate base undirected graph using iterative method
    base_graph, achieved_treewidth, diff = generate_graph_with_target_treewidth(
        n_nodes,
        target_treewidth,
        max_iterations,
        seed=seed,
    )

    # Convert to DAG
    dag = undirected_to_dag(base_graph, dag_method, seed=seed)

    # Apply node naming strategy
    node_names = None
    if (
        node_naming != NamingStrategy.DEFAULT
    ):  # DEFAULT keeps numeric labels 0, 1, 2, ...
        node_names = generate_node_names(n_nodes, node_naming, seed=seed)
        dag = relabel_graph_nodes(dag, node_names)

    # Verify final treewidth of the DAG's underlying undirected graph
    final_undirected = dag.to_undirected()
    final_treewidth_float, _ = treewidth.treewidth_min_degree(final_undirected)
    final_treewidth = int(final_treewidth_float)

    # Create metadata (store string value for serialization)
    metadata = DAGGenerationMetadata(
        n_nodes=n_nodes,
        target_treewidth=target_treewidth,
        achieved_treewidth=achieved_treewidth,
        treewidth_difference=diff,
        exact_treewidth=(diff == 0),
        dag_method=dag_method,
        node_naming=node_naming.value,  # Store string value
        max_iterations=max_iterations,
        base_graph_edges=base_graph.number_of_edges(),
        dag_edges=dag.number_of_edges(),
        final_treewidth=final_treewidth,
        node_names=node_names,
        seed=seed,
    )

    return dag, final_treewidth, metadata.to_dict()  # ty: ignore
