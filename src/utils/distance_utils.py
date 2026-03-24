"""
Distance computation utilities for Bayesian network graphs.

This module provides shared distance computation helper functions used across
the query generation and analysis modules.
"""

from __future__ import annotations

import networkx as nx


def compute_shortest_distance(
    graph: nx.Graph | nx.DiGraph,
    node1: str,
    node2: str,
    no_path_value: int | None = None,
) -> int | None:
    """
    Compute shortest distance between two nodes.

    Args:
        graph: NetworkX graph (directed or undirected)
        node1: First node
        node2: Second node
        no_path_value: Value to return when no path exists
                      (None = return None, int = return that value)

    Returns:
        Distance or no_path_value if no path exists
    """
    # Convert to undirected if needed
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    try:
        return nx.shortest_path_length(graph, node1, node2)
    except nx.NetworkXNoPath:
        return no_path_value


def compute_min_distance_between_sets(
    graph: nx.Graph | nx.DiGraph,
    nodes1: list[str],
    nodes2: list[str],
) -> int | None:
    """
    Compute minimum distance between any node in nodes1 and nodes2.

    Args:
        graph: NetworkX graph (directed or undirected)
        nodes1: First set of nodes
        nodes2: Second set of nodes

    Returns:
        Minimum distance, or None if no valid distances exist
    """
    if not nodes1 or not nodes2:
        return None

    min_dist = float("inf")
    for node1 in nodes1:
        for node2 in nodes2:
            dist = compute_shortest_distance(graph, node1, node2, no_path_value=None)
            if dist is not None:
                min_dist = min(min_dist, dist)

    return int(min_dist) if min_dist != float("inf") else None


def compute_min_distance_within_set(
    graph: nx.Graph | nx.DiGraph,
    nodes: list[str],
) -> int | None:
    """
    Compute minimum distance between any pair of nodes within the same set.

    Args:
        graph: NetworkX graph (directed or undirected)
        nodes: List of nodes

    Returns:
        Minimum distance within set, or None if insufficient nodes or no paths
    """
    if len(nodes) < 2:
        return None

    min_dist = float("inf")
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i + 1 :]:
            dist = compute_shortest_distance(graph, node1, node2, no_path_value=None)
            if dist is not None:
                min_dist = min(min_dist, dist)

    return int(min_dist) if min_dist != float("inf") else None


def compute_all_pairwise_distances(
    graph: nx.Graph | nx.DiGraph,
    nodes1: list[str],
    nodes2: list[str],
) -> list[int] | None:
    """
    Compute all pairwise distances between target and evidence nodes.

    Args:
        graph: NetworkX graph (directed or undirected)
        nodes1: First set of nodes
        nodes2: Second set of nodes

    Returns:
        Sorted list of distances, e.g., [1, 1, 2, 3], or None if no valid distances
    """
    if not nodes1 or not nodes2:
        return None

    distances = []
    for node1 in nodes1:
        for node2 in nodes2:
            dist = compute_shortest_distance(graph, node1, node2, no_path_value=None)
            if dist is not None:
                distances.append(dist)

    return sorted(distances) if distances else None
