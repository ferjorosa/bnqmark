"""
Visualization utilities for Bayesian Networks and graphs.

This module provides functions for drawing Bayesian Networks and NetworkX graphs
with automatic hierarchical layouts and treewidth computation.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


def _draw_digraph_hierarchical(
    graph: nx.DiGraph,
    title: str = "Network Structure (Hierarchical Layout)",
    node_size: int = 3000,
    node_color: str = "lightblue",
    font_size: int = 12,
    figsize: tuple[int, int] = (10, 6),
    show_treewidth: bool = True,
) -> dict[str, Any]:
    """
    Core method to draw a NetworkX DiGraph with hierarchical layout.

    Parents are placed above children.

    Args:
        graph: NetworkX DiGraph to draw
        title: Plot title
        node_size: Size of nodes
        node_color: Color of nodes
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display treewidth in title
            (default: True)

    Returns:
        Dictionary with layout info and treewidth (if computed)
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("Graph must be a NetworkX DiGraph")

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG (Directed Acyclic Graph)")

    # Topological sort of nodes
    topo_order = list(nx.topological_sort(graph))

    # Assign y-levels based on topological layers
    layers: dict[Any, int] = {}
    for node in topo_order:
        parents = list(graph.predecessors(node))
        if parents:
            layers[node] = max(layers[p] for p in parents) + 1
        else:
            layers[node] = 0  # root nodes at top layer

    # Assign x-positions to spread nodes horizontally
    layer_nodes: dict[int, list[Any]] = {}
    for node, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    pos = {}
    for layer, nodes in layer_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (
                i - n / 2,
                -layer,
            )  # center horizontally, invert y-axis for top-down

    # Compute treewidth if requested
    display_title = title
    result = {"positions": pos, "layers": layers, "layer_nodes": layer_nodes}

    if show_treewidth:
        try:
            from networkx.algorithms.approximation import treewidth

            # Convert to undirected for treewidth computation
            undirected_graph = graph.to_undirected()
            width, decomposition = treewidth.treewidth_min_degree(undirected_graph)
            result["treewidth"] = {"width": width, "decomposition": decomposition}
            display_title = f"{title} (Treewidth ≈ {width})"
        except ImportError:
            logger.warning(
                "Could not compute treewidth. "
                "NetworkX approximation module not available.",
            )

    # Draw the network
    plt.figure(figsize=figsize)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        font_weight="bold",
        arrows=True,
    )
    plt.title(display_title)
    plt.show()

    return result


def draw_bayesian_network(
    model,
    node_size: int = 3000,
    node_color: str = "lightblue",
    font_size: int = 12,
    figsize: tuple[int, int] = (10, 6),
    show_treewidth: bool = True,
):
    """
    Draws a Bayesian network (pgmpy) with an automatic hierarchical layout.

    Parents are placed above children.

    Args:
        model: pgmpy BayesianNetwork or LinearGaussianBayesianNetwork
        node_size: Size of nodes
        node_color: Color of nodes
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display approximate treewidth
            (default: True)

    Returns:
        Dictionary with layout info and treewidth (if computed)

    Example:
        >>> from src.bn.analysis import draw_bayesian_network
        >>> result = draw_bayesian_network(bn, show_treewidth=True)
        >>> print(
        ...     f"Network treewidth: "
        ...     f"{result.get('treewidth', {}).get('width', 'unknown')}"
        ... )
    """
    # Convert BayesianNetwork edges to NetworkX DiGraph
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    return _draw_digraph_hierarchical(
        graph,
        title="Bayesian Network Structure (Hierarchical Layout)",
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        figsize=figsize,
        show_treewidth=show_treewidth,
    )


def draw_networkx_graph(
    graph: nx.DiGraph,
    title: str | None = None,
    node_size: int = 3000,
    node_color: str = "lightgreen",
    font_size: int = 12,
    figsize: tuple[int, int] = (10, 6),
    show_treewidth: bool = True,
):
    """
    Draws a NetworkX DiGraph with an automatic hierarchical layout.

    Parents are placed above children.

    Args:
        graph: NetworkX DiGraph to draw
        title: Plot title (auto-generated if None)
        node_size: Size of nodes
        node_color: Color of nodes
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display approximate treewidth
            (default: True)

    Returns:
        Dictionary with layout info and treewidth (if computed)

    Example:
        >>> import networkx as nx
        >>> from src.bn.analysis import draw_networkx_graph
        >>> G = nx.DiGraph([("A", "B"), ("B", "C")])
        >>> result = draw_networkx_graph(G, title="My Graph")
    """
    if title is None:
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        title = f"NetworkX Graph ({n_nodes} nodes, {n_edges} edges)"

    return _draw_digraph_hierarchical(
        graph,
        title=title,
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        figsize=figsize,
        show_treewidth=show_treewidth,
    )
