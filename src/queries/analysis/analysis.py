"""
Query Structural Analysis Utilities.

This module provides functions for analyzing the structural properties of queries
in Bayesian Networks, including Markov blanket analysis, distance metrics, and
topological properties.

These utilities are useful for understanding query difficulty from a structural
perspective and for post-hoc analysis of generated queries.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.utils.distance_utils import (
    compute_all_pairwise_distances,
    compute_min_distance_between_sets,
    compute_min_distance_within_set,
)


def compute_markov_blanket_metrics(
    bn: Any,
    target_nodes: list[str],
    evidence_nodes: list[str],
) -> dict[str, float]:
    """
    Compute Markov blanket size metrics for target and evidence nodes.

    The Markov blanket of a node consists of its parents, children, and
    children's other parents. This function computes average Markov blanket
    sizes for both target and evidence node sets.

    Args:
        bn: pgmpy DiscreteBayesianNetwork object
        target_nodes: List of query/target variable names
        evidence_nodes: List of evidence variable names

    Returns:
        Dictionary containing:
        - avg_markov_blanket_size_target: Average MB size for target nodes
        - avg_markov_blanket_size_evidence: Average MB size for evidence nodes
    """

    def _get_markov_blanket_size(node: str) -> int:
        """Compute Markov blanket size for a single node."""
        parents = set(bn.get_parents(node))
        children = set(bn.get_children(node))

        # Children's other parents (co-parents)
        co_parents = set()
        for child in children:
            child_parents = set(bn.get_parents(child))
            co_parents.update(child_parents - {node})

        # Markov blanket = parents + children + co-parents
        markov_blanket = parents | children | co_parents
        return len(markov_blanket)

    # Compute average Markov blanket size for target nodes
    if target_nodes:
        mb_sizes_target = [_get_markov_blanket_size(node) for node in target_nodes]
        avg_mb_target = float(np.mean(mb_sizes_target))
    else:
        avg_mb_target = 0.0

    # Compute average Markov blanket size for evidence nodes
    if evidence_nodes:
        mb_sizes_evidence = [_get_markov_blanket_size(node) for node in evidence_nodes]
        avg_mb_evidence = float(np.mean(mb_sizes_evidence))
    else:
        avg_mb_evidence = 0.0

    return {
        "avg_markov_blanket_size_target": avg_mb_target,
        "avg_markov_blanket_size_evidence": avg_mb_evidence,
    }


def compute_distance_metrics(
    bn: Any,
    target_nodes: list[str],
    evidence_nodes: list[str],
) -> dict[str, int | str | list[int] | None]:
    """
    Compute shortest path distance metrics between different node sets.

    Distances are computed in the undirected version of the Bayesian network
    graph, representing the minimum number of edges between nodes regardless
    of causal direction.

    Args:
        bn: pgmpy DiscreteBayesianNetwork object
        target_nodes: List of query/target variable names
        evidence_nodes: List of evidence variable names

    Returns:
        Dictionary containing:
        - min_distance_target_evidence: Min distance between any target and
          evidence node
        - min_distance_target_target: Min distance between any pair of target nodes
        - min_distance_evidence_evidence: Min distance between any pair of
          evidence nodes
        - evidence_distances: Sorted list of all pairwise distances between target
          and evidence nodes, e.g., [1, 1, 2, 3]

        Returns None for distances when insufficient nodes exist for computation.
    """
    # Create undirected graph for distance calculations
    graph = nx.Graph()
    graph.add_nodes_from(bn.nodes())
    graph.add_edges_from(
        bn.edges()
    )  # NetworkX Graph automatically makes edges undirected

    return {
        "min_distance_target_evidence": compute_min_distance_between_sets(
            graph,
            target_nodes,
            evidence_nodes,
        ),
        "min_distance_target_target": compute_min_distance_within_set(
            graph, target_nodes
        ),
        "min_distance_evidence_evidence": compute_min_distance_within_set(
            graph,
            evidence_nodes,
        ),
        "evidence_distances": compute_all_pairwise_distances(
            graph,
            target_nodes,
            evidence_nodes,
        ),
    }


def compute_topological_properties(
    bn: Any,
    target_nodes: list[str],
    evidence_nodes: list[str],
) -> dict[str, bool | None]:
    """
    Compute topological properties of target and evidence nodes.

    Analyzes whether nodes are roots (no parents) or leaves (no children)
    in the Bayesian network DAG structure.

    Args:
        bn: pgmpy DiscreteBayesianNetwork object
        target_nodes: List of query/target variable names
        evidence_nodes: List of evidence variable names

    Returns:
        Dictionary containing:
        - all_target_are_roots: True if all target nodes have no parents
        - all_target_are_leaves: True if all target nodes have no children
        - all_evidence_are_roots: True if all evidence nodes have no parents
        - all_evidence_are_leaves: True if all evidence nodes have no children

        Returns None for properties when no nodes exist in the respective set.

    Example:
        >>> props = compute_topological_properties(bn, ["A"], ["B", "C"])
        >>> if props["all_evidence_are_leaves"]:
        ...     print("All evidence nodes are leaf nodes")
    """

    def _all_are_roots(nodes: list[str]) -> bool | None:
        """Check if all nodes in the list are root nodes (no parents)."""
        if not nodes:
            return None
        return all(len(bn.get_parents(node)) == 0 for node in nodes)

    def _all_are_leaves(nodes: list[str]) -> bool | None:
        """Check if all nodes in the list are leaf nodes (no children)."""
        if not nodes:
            return None
        return all(len(bn.get_children(node)) == 0 for node in nodes)

    return {
        "all_target_are_roots": _all_are_roots(target_nodes),
        "all_target_are_leaves": _all_are_leaves(target_nodes),
        "all_evidence_are_roots": _all_are_roots(evidence_nodes),
        "all_evidence_are_leaves": _all_are_leaves(evidence_nodes),
    }


def compute_query_structural_properties(
    bn: Any,
    target_nodes: list[str],
    evidence_nodes: list[str],
) -> dict[str, Any]:
    """
    Compute comprehensive structural properties of query and evidence nodes.

    This is the main function that combines all structural analysis metrics
    into a single comprehensive result. It computes Markov blanket sizes,
    distance metrics, and topological properties.

    This function is a modernized and improved version of the old
    get_query_metadata() function from bn_query_sweep.py.

    Args:
        bn: pgmpy DiscreteBayesianNetwork object
        target_nodes: List of query/target variable names
        evidence_nodes: List of evidence variable names

    Returns:
        Dictionary containing all structural metrics:

        Markov Blanket Metrics:
        - avg_markov_blanket_size_target: Average MB size for target nodes
        - avg_markov_blanket_size_evidence: Average MB size for evidence nodes

        Distance Metrics:
        - min_distance_target_evidence: Min distance between target and evidence
        - min_distance_target_target: Min distance between target nodes
        - min_distance_evidence_evidence: Min distance between evidence nodes

        Topological Properties:
        - all_target_are_roots: Whether all targets are root nodes
        - all_target_are_leaves: Whether all targets are leaf nodes
        - all_evidence_are_roots: Whether all evidence are root nodes
        - all_evidence_are_leaves: Whether all evidence are leaf nodes

    Note:
        This function provides a comprehensive structural analysis that can help
        understand query difficulty and complexity from a graph-theoretic perspective.
        It's particularly useful for:

        - Analyzing the structural complexity of generated queries
        - Understanding why certain queries might be harder for LLMs
        - Post-hoc analysis of query generation results
        - Filtering or stratifying queries by structural properties
    """
    # Validate inputs
    if not isinstance(target_nodes, list):
        raise TypeError("target_nodes must be a list")
    if not isinstance(evidence_nodes, list):
        raise TypeError("evidence_nodes must be a list")

    # Check that all nodes exist in the network
    all_nodes = set(bn.nodes())
    invalid_targets = set(target_nodes) - all_nodes
    invalid_evidence = set(evidence_nodes) - all_nodes

    if invalid_targets:
        raise ValueError(f"Target nodes not found in network: {invalid_targets}")
    if invalid_evidence:
        raise ValueError(f"Evidence nodes not found in network: {invalid_evidence}")

    # Check for overlap between target and evidence nodes
    overlap = set(target_nodes) & set(evidence_nodes)
    if overlap:
        raise ValueError(f"Nodes cannot be both target and evidence: {overlap}")

    # Compute all structural metrics
    result: dict[str, Any] = {}

    # Markov blanket metrics
    mb_metrics = compute_markov_blanket_metrics(bn, target_nodes, evidence_nodes)
    result.update(mb_metrics)

    # Distance metrics
    distance_metrics = compute_distance_metrics(bn, target_nodes, evidence_nodes)
    result.update(distance_metrics)

    # Topological properties
    topo_properties = compute_topological_properties(bn, target_nodes, evidence_nodes)
    result.update(topo_properties)

    # Add basic counts for convenience
    result.update(
        {
            "num_target_nodes": len(target_nodes),
            "num_evidence_nodes": len(evidence_nodes),
            "target_nodes": target_nodes.copy(),
            "evidence_nodes": evidence_nodes.copy(),
        },
    )

    return result


__all__ = [
    "compute_query_structural_properties",
    "compute_markov_blanket_metrics",
    "compute_distance_metrics",
    "compute_topological_properties",
]
