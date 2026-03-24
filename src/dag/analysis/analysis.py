"""
Graph analysis utilities for DAG generation.

This module provides functions for analyzing structural and complexity
properties of generated graphs and DAGs.
"""

from typing import Any

import networkx as nx
from networkx.algorithms.approximation import treewidth


def analyze_graph_properties(graph: nx.Graph) -> dict[str, Any]:
    """
    Analyze various structural and complexity properties of a graph.

    This utility function computes key properties useful for understanding
    the characteristics of generated graphs, particularly for experimental
    analysis of computational complexity and structural features.

    Args:
        graph: NetworkX Graph or DiGraph to analyze

    Returns:
        Dictionary containing graph properties:
        - 'n_nodes': Number of nodes
        - 'n_edges': Number of edges
        - 'is_connected': Whether the underlying undirected graph is connected
        - 'density': Edge density (ratio of actual to possible edges)
        - 'treewidth': Approximate treewidth (None if computation fails)
        - 'tree_decomposition_size': Number of nodes in tree decomposition
        - 'is_dag': Whether the graph is a DAG (DiGraph only)
        - 'max_path_length': Longest path length (DAG only)
        - 'topological_levels': Number of topological levels (DAG only)

    Example:
        >>> from src.dag import generate_single_dag
        >>> dag, _, _ = generate_single_dag(10, 3, seed=42)
        >>> props = analyze_graph_properties(dag)
        >>> print(f"Nodes: {props['n_nodes']}, Treewidth: {props['treewidth']}")
        Nodes: 10, Treewidth: 3
        >>> print(f"Is DAG: {props['is_dag']}, Levels: {props['topological_levels']}")
        Is DAG: True, Levels: 4
    """
    undirected_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    properties = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "is_connected": nx.is_connected(undirected_graph),
        "density": nx.density(graph),
    }

    # Compute treewidth
    try:
        width, decomposition = treewidth.treewidth_min_degree(undirected_graph)
        properties["treewidth"] = width
        properties["tree_decomposition_size"] = len(decomposition.nodes())
    except Exception as e:
        properties["treewidth"] = None
        properties["treewidth_error"] = str(e)

    # DAG-specific properties
    if isinstance(graph, nx.DiGraph):
        properties["is_dag"] = nx.is_directed_acyclic_graph(graph)
        if properties["is_dag"]:
            properties["max_path_length"] = nx.dag_longest_path_length(graph)
            properties["topological_levels"] = len(
                list(nx.topological_generations(graph))
            )

    return properties


def verify_dag_properties(
    dag: nx.DiGraph,
    expected_n_nodes: int,
    expected_treewidth: int,
    tolerance: int = 0,
) -> dict[str, Any]:
    """
    Verify that a DAG meets expected properties.

    Useful for testing and validation of DAG generation.

    Args:
        dag: The DAG to verify
        expected_n_nodes: Expected number of nodes
        expected_treewidth: Expected treewidth
        tolerance: Allowable difference from expected_treewidth

    Returns:
        Dictionary with verification results:
        - 'is_valid': Whether all checks passed
        - 'is_dag': Whether graph is a DAG
        - 'correct_n_nodes': Whether node count matches
        - 'treewidth_ok': Whether treewidth is within tolerance
        - 'actual_treewidth': The actual treewidth
        - 'issues': List of any issues found

    Example:
        >>> from src.dag import generate_single_dag
        >>> dag, tw, _ = generate_single_dag(10, 3, seed=42)
        >>> result = verify_dag_properties(dag, 10, 3, tolerance=1)
        >>> print(result["is_valid"])
        True
    """
    issues = []

    # Check if it's a DAG
    is_dag = nx.is_directed_acyclic_graph(dag)
    if not is_dag:
        issues.append("Graph is not a DAG")

    # Check node count
    actual_n_nodes = dag.number_of_nodes()
    correct_n_nodes = actual_n_nodes == expected_n_nodes
    if not correct_n_nodes:
        issues.append(f"Expected {expected_n_nodes} nodes, got {actual_n_nodes}")

    # Check treewidth
    try:
        undirected = dag.to_undirected()
        actual_treewidth, _ = treewidth.treewidth_min_degree(undirected)
        treewidth_diff = abs(actual_treewidth - expected_treewidth)
        treewidth_ok = treewidth_diff <= tolerance

        if not treewidth_ok:
            issues.append(
                f"Treewidth {actual_treewidth} differs from expected "
                f"{expected_treewidth} by more than tolerance {tolerance}",
            )
    except Exception as e:
        actual_treewidth = None
        treewidth_ok = False
        issues.append(f"Failed to compute treewidth: {str(e)}")

    return {
        "is_valid": len(issues) == 0,
        "is_dag": is_dag,
        "correct_n_nodes": correct_n_nodes,
        "treewidth_ok": treewidth_ok,
        "actual_treewidth": actual_treewidth,
        "issues": issues,
    }
