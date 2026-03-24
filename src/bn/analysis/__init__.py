"""
Bayesian Network Analysis Module.

This module provides analysis functions for Bayesian Networks, including
network-level metrics and visualization utilities.

Main Functions:
    Network Metrics:
    - compute_average_markov_blanket_size() - Average MB size across all nodes
    - num_edges() - Number of edges in the network

    Visualization:
    - draw_bayesian_network() - Draw BN with hierarchical layout
    - draw_networkx_graph() - Draw NetworkX graph with hierarchical layout

Example:
    >>> from src.bn.analysis import (
    ...     compute_average_markov_blanket_size,
    ...     draw_bayesian_network,
    ... )
    >>> # Analyze network properties
    >>> avg_mb_size = compute_average_markov_blanket_size(bn)
    >>> print(f"Average Markov blanket size: {avg_mb_size:.2f}")
    >>> # Visualize the network
    >>> draw_bayesian_network(bn, show_treewidth=True)
"""

# Network metrics
from .network_metrics import (
    compute_average_markov_blanket_size,
    num_edges,
)

# Visualization
from .visualization import (
    draw_bayesian_network,
    draw_networkx_graph,
)

__all__ = [
    # Network metrics
    "compute_average_markov_blanket_size",
    "num_edges",
    # Visualization
    "draw_bayesian_network",
    "draw_networkx_graph",
]
