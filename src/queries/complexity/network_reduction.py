"""
Network reduction utilities for query complexity analysis.

This module provides functions for creating reduced Bayesian Networks by:
- Building networks with only specified variables
- Marginalizing CPDs when parents are removed
- Adding fallback uniform CPDs when necessary
"""

import logging

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

logger = logging.getLogger(__name__)


def create_marginal_cpd(node, cpd, kept_parents, cardinality):
    """
    Create a marginal CPD by summing out removed parents.

    Args:
        node: Variable name for the CPD
        cpd: Original TabularCPD from source network
        kept_parents: List of parent variables to keep
        cardinality: Dictionary mapping variables to their cardinalities

    Returns:
        TabularCPD: Marginalized CPD with only kept parents
    """
    node_card = cardinality[node]
    cpd_values = cpd.values.copy()
    cpd_variable_order = list(cpd.variables)

    # Find indices of removed parents (skip index 0 which is the node)
    kept_parents_set = set(kept_parents)
    axes_to_sum = []
    for i, var in enumerate(cpd_variable_order[1:], start=1):
        if var not in kept_parents_set:
            axes_to_sum.append(i)

    # Sum over removed parent dimensions
    if axes_to_sum:
        marginal_values = np.sum(cpd_values, axis=tuple(axes_to_sum))
    else:
        marginal_values = cpd_values

    # Handle case with no kept parents
    if len(kept_parents) == 0:
        marginal_values = marginal_values.flatten()
        if marginal_values.sum() > 0:
            marginal_values = marginal_values / marginal_values.sum()
        else:
            marginal_values = np.ones(node_card) / node_card
        return TabularCPD(
            variable=node,
            variable_card=node_card,
            values=marginal_values.reshape(-1, 1),
        )

    # Reorder dimensions to match kept_parents order
    desired_order = [0]  # Start with node
    for kept_parent in kept_parents:
        orig_pos = cpd_variable_order.index(kept_parent)
        num_removed_before = sum(
            1
            for i, v in enumerate(cpd_variable_order[1:orig_pos], 1)
            if v not in kept_parents_set
        )
        current_pos = orig_pos - num_removed_before
        desired_order.append(current_pos)

    marginal_values = np.transpose(marginal_values, axes=desired_order)
    kept_parents_cards = [cardinality[p] for p in kept_parents]
    marginal_values = marginal_values / marginal_values.sum(axis=0, keepdims=True)

    return TabularCPD(
        variable=node,
        variable_card=node_card,
        values=marginal_values,
        evidence=kept_parents,
        evidence_card=kept_parents_cards,
    )


def add_cpd_to_network(network, node, source_bn, vars_to_keep, verbose=False):
    """
    Add CPD for a node to the network, handling removed parents.

    Args:
        network: Target BayesianNetwork to add CPD to
        node: Variable name to add CPD for
        source_bn: Source BayesianNetwork containing original CPD
        vars_to_keep: Set of variables being kept in the reduced network
        verbose: Whether to print debug information

    Returns:
        bool: True if CPD was successfully added, False otherwise
    """
    try:
        cpd = source_bn.get_cpds(node)
        parents = list(cpd.variables)
        parents.remove(node)

        # Check if all parents are kept
        if all(p in vars_to_keep for p in parents):
            network.add_cpds(cpd)
            return True

        # Some parents were removed - create marginal CPD
        kept_parents = [p for p in parents if p in vars_to_keep]
        cardinality = source_bn.get_cardinality()
        marginal_cpd = create_marginal_cpd(node, cpd, kept_parents, cardinality)
        network.add_cpds(marginal_cpd)
        return True

    except Exception as e:
        if verbose:
            logger.warning(f"Could not copy CPD for {node}: {e}")
        return False


def add_uniform_cpd(network, node, cardinality, verbose=False):
    """
    Add a uniform (fallback) CPD to a network.

    Args:
        network: Target BayesianNetwork
        node: Variable name to add CPD for
        cardinality: Dictionary mapping variables to their cardinalities
        verbose: Whether to print debug information
    """
    try:
        node_card = cardinality[node]
        uniform_values = np.ones((node_card, 1)) / node_card
        fallback_cpd = TabularCPD(
            variable=node,
            variable_card=node_card,
            values=uniform_values,
        )
        network.add_cpds(fallback_cpd)
        if verbose:
            logger.debug(f"Added uniform fallback CPD for {node}")
    except Exception as e:
        if verbose:
            logger.error(f"Error creating fallback CPD for {node}: {e}")
        raise


def build_reduced_network(source_bn, vars_to_keep, verbose=False):
    """
    Build a reduced network containing only specified variables.

    Args:
        source_bn: Source BayesianNetwork
        vars_to_keep: Set of variables to keep in reduced network
        verbose: Whether to print debug information

    Returns:
        DiscreteBayesianNetwork: Reduced network with only kept variables
    """
    reduced_bn = DiscreteBayesianNetwork()
    reduced_bn.add_nodes_from(vars_to_keep)

    # Add edges connecting kept variables
    for edge in source_bn.edges():
        u, v = edge
        if u in vars_to_keep and v in vars_to_keep:
            reduced_bn.add_edge(u, v)

    # Copy CPDs for kept variables
    cardinality = source_bn.get_cardinality()
    for node in vars_to_keep:
        cpd_added = add_cpd_to_network(
            reduced_bn,
            node,
            source_bn,
            vars_to_keep,
            verbose,
        )
        if not cpd_added:
            add_uniform_cpd(reduced_bn, node, cardinality, verbose)

    return reduced_bn
