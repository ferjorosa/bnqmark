"""
Network-level metrics for Bayesian Networks.

This module provides functions for computing structural properties and metrics
that characterize the entire Bayesian Network, independent of specific queries.
"""


def num_edges(bn):
    """
    Count the number of edges in a Bayesian Network.

    Args:
        bn: pgmpy BayesianNetwork or DiscreteBayesianNetwork

    Returns:
        int: Number of directed edges in the network

    Example:
        >>> edges = num_edges(bayesian_network)
        >>> print(f"Network has {edges} edges")
    """
    # For pgmpy BayesianModel, the edges can be accessed by .edges
    return len(list(bn.edges()))


def compute_average_markov_blanket_size(bn):
    """
    Compute the average Markov blanket size across all nodes in the network.

    The Markov blanket of a node consists of:
    - Its parents
    - Its children
    - Other parents of its children (co-parents)

    This function computes the Markov blanket size for each node and returns
    the average across the entire network. This is a network-level metric
    that characterizes the overall connectivity and local complexity.

    Args:
        bn: pgmpy BayesianNetwork or DiscreteBayesianNetwork

    Returns:
        float: Average Markov blanket size across all nodes in the network.
               Returns 0.0 if the network has no nodes.

    Example:
        >>> avg_mb = compute_average_markov_blanket_size(bayesian_network)
        >>> print(f"Average Markov blanket size: {avg_mb:.2f}")

    Note:
        This is different from query-specific Markov blanket analysis (which
        computes MB sizes only for specific target/evidence nodes). This function
        provides a global characterization of the network's local complexity.
    """
    # Compute Markov blanket for each node: parents, children,
    # and other parents of children
    node_blankets = []
    for node in bn.nodes():
        parents = set(bn.predecessors(node))
        children = set(bn.successors(node))
        other_parents = set()
        for child in children:
            other_parents.update(bn.predecessors(child))
        # Markov blanket = parents ∪ children ∪ (other parents of children)
        # minus the node itself
        blanket = parents | children | other_parents
        blanket.discard(node)
        node_blankets.append(len(blanket))

    if node_blankets:
        return sum(node_blankets) / len(node_blankets)
    else:
        return 0.0
