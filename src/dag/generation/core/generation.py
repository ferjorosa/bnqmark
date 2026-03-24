"""
Core DAG generation algorithms.

This module contains the low-level implementation for generating undirected
graphs with target treewidth and converting them to DAGs.

Most users should use the high-level API from generator.py instead of
importing directly from this module.
"""

import random

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import treewidth


def generate_graph_with_target_treewidth(
    n_nodes: int,
    target_treewidth: int,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> tuple[nx.Graph, int, int]:
    """
    Generate an undirected graph trying to achieve a specific treewidth.

    This method uses a heuristic iterative approach: starts with a random tree
    (treewidth=1) and iteratively adds edges while monitoring treewidth
    approximations until the target is reached or max_iterations is exceeded.

    **Advantages:**
    - More diverse graph structures than deterministic methods
    - Can produce more "natural" looking networks
    - Good for testing robustness across different topologies

    **Disadvantages:**
    - Only approximate treewidth (uses approximation algorithms)
    - May not reach exact target treewidth
    - Success depends on target difficulty

    Args:
        n_nodes: Number of nodes in the graph
        target_treewidth: Desired treewidth (will try to approximate this)
        max_iterations: Maximum number of generation attempts (default: 1000)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_graph, achieved_treewidth, difference_from_target)
        - best_graph: NetworkX Graph with closest achieved treewidth
        - achieved_treewidth: Actual treewidth of the returned graph
        - difference_from_target: |achieved_treewidth - target_treewidth|

    Example:
        >>> graph, tw, diff = generate_graph_with_target_treewidth(10, 3, seed=42)
        >>> print(f"Target: 3, Achieved: {tw}, Diff: {diff}")
        Target: 3, Achieved: 3, Diff: 0
    """
    # Set seed once at the beginning for reproducible results
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    best_graph: nx.Graph | None = None
    best_treewidth = 0
    best_diff = 999999  # Large integer to ensure first iteration always updates

    for _iteration in range(max_iterations):
        # Start with a random tree (treewidth = 1)
        graph = nx.random_labeled_tree(n_nodes)

        # Iteratively add edges to increase treewidth
        max_edge_additions = n_nodes * (n_nodes - 1) // 2 - (
            n_nodes - 1
        )  # Max possible edges - tree edges

        for _ in range(max_edge_additions):
            # Compute current treewidth
            current_width, _ = treewidth.treewidth_min_degree(graph)

            if current_width >= target_treewidth:
                break

            # Try to add a random edge
            nodes = list(graph.nodes())
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                u, v = np.random.choice(nodes, 2, replace=False)
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    break
                attempts += 1

            if attempts >= 50:  # No more edges can be added
                break

        # Check final treewidth
        final_width, _ = treewidth.treewidth_min_degree(graph)
        final_width_int = int(final_width)
        diff = abs(final_width_int - target_treewidth)

        if diff < best_diff:
            best_diff = diff
            best_treewidth = final_width_int
            best_graph = graph.copy()

        if diff == 0:  # Exact match found
            break

    assert best_graph is not None
    return best_graph, best_treewidth, best_diff


def _bfs_to_dag(graph: nx.Graph, root: int | None = None) -> list[tuple]:
    """Convert graph to DAG using BFS spanning tree."""
    if root is None:
        root = np.random.choice(list(graph.nodes()))
    return list(nx.bfs_edges(graph, root))


def _dfs_to_dag(graph: nx.Graph, root: int | None = None) -> list[tuple]:
    """Convert graph to DAG using DFS spanning tree."""
    if root is None:
        root = np.random.choice(list(graph.nodes()))
    return list(nx.dfs_edges(graph, root))


def _random_to_dag(graph: nx.Graph) -> list[tuple]:
    """Convert graph to DAG using random edge orientation."""
    edges = list(graph.edges())
    np.random.shuffle(edges)

    temp_dag = nx.DiGraph()
    temp_dag.add_nodes_from(graph.nodes())
    dag_edges = []

    for u, v in edges:
        # Try both orientations and pick one that doesn't create a cycle
        for edge_candidate in [(u, v), (v, u)]:
            temp_dag.add_edge(*edge_candidate)
            if nx.is_directed_acyclic_graph(temp_dag):
                dag_edges.append(edge_candidate)
                break
            else:
                temp_dag.remove_edge(*edge_candidate)

    return dag_edges


def _topological_to_dag(graph: nx.Graph) -> list[tuple]:
    """Convert graph to DAG using topological ordering."""
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)
    node_order = {node: i for i, node in enumerate(nodes)}

    dag_edges = []
    for u, v in graph.edges():
        if node_order[u] < node_order[v]:
            dag_edges.append((u, v))
        else:
            dag_edges.append((v, u))

    return dag_edges


def undirected_to_dag(
    graph: nx.Graph,
    method: str = "random",
    root: int | None = None,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Convert an undirected graph to a DAG using various methods.

    **IMPORTANT:** Choice of method significantly affects treewidth preservation:
    - 'bfs'/'dfs': Create spanning trees → treewidth = 1 (Naive Bayes structure)
    - 'random'/'topological': Preserve all edges → maintain original treewidth

    **Method Details:**
    - 'bfs': Breadth-first spanning tree (good for hierarchical structures)
    - 'dfs': Depth-first spanning tree (good for deep hierarchical structures)
    - 'random': Random edge orientation while avoiding cycles (preserves complexity)
    - 'topological': Random node ordering with consistent edge directions
        (preserves complexity)

    **Recommendation for Treewidth Experiments:**
    Use 'random' or 'topological' to preserve the treewidth of the original graph.
    Use 'bfs' or 'dfs' only if you specifically want simple tree structures.

    Args:
        graph: Undirected NetworkX graph to convert
        method: Conversion method ('bfs', 'dfs', 'random', 'topological')
                Default: 'random' (recommended for treewidth preservation)
        root: Root node for tree-based methods ('bfs'/'dfs'). Chosen randomly if None.
        seed: Random seed for reproducibility

    Returns:
        NetworkX DiGraph that is a DAG

    Raises:
        ValueError: If method is not one of the supported options

    Example:
        >>> # Preserve treewidth (recommended for experiments)
        >>> dag = undirected_to_dag(graph, method="random", seed=42)

        >>> # Create simple tree structure
        >>> tree_dag = undirected_to_dag(graph, method="bfs", root=0)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if not nx.is_connected(graph):
        # Handle each connected component separately
        dag_edges = []
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            sub_dag = undirected_to_dag(subgraph, method, root, seed)
            dag_edges.extend(sub_dag.edges())
        return nx.DiGraph(dag_edges)

    if method == "bfs":
        dag_edges = _bfs_to_dag(graph, root)
    elif method == "dfs":
        dag_edges = _dfs_to_dag(graph, root)
    elif method == "random":
        dag_edges = _random_to_dag(graph)
    elif method == "topological":
        dag_edges = _topological_to_dag(graph)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'bfs', 'dfs', 'random', or 'topological'",
        )

    return nx.DiGraph(dag_edges)
