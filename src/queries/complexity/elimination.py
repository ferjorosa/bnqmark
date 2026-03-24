"""
Variable elimination complexity analysis.

This module provides functions for analyzing the computational complexity of
variable elimination inference, including:
- Computing optimal elimination orders
- Computing induced width
- Simulating variable elimination to estimate computational cost
"""

import logging

logger = logging.getLogger(__name__)


def compute_elimination_order(reduced_bn, eliminate_vars):
    """
    Compute optimal elimination order for variables.

    Args:
        reduced_bn: BayesianNetwork to compute order for
        eliminate_vars: Set of variables to eliminate

    Returns:
        list: Optimal elimination order
    """
    from pgmpy.inference.EliminationOrder import WeightedMinFill

    if eliminate_vars:
        orderer = WeightedMinFill(reduced_bn)
        return orderer.get_elimination_order(nodes=list(eliminate_vars))
    return []


def compute_induced_width(reduced_bn, elim_order, keep_vars, verbose=False):
    """
    Compute induced width for elimination order.

    Args:
        reduced_bn: BayesianNetwork
        elim_order: Elimination order for variables to eliminate
        keep_vars: Set of variables to keep (targets)
        verbose: Whether to print debug information

    Returns:
        int: Induced width of the elimination order
    """
    from pgmpy.inference import VariableElimination

    if not elim_order:
        return 0

    complete_elim_order = elim_order + list(keep_vars)
    ve = VariableElimination(reduced_bn)
    try:
        return ve.induced_width(complete_elim_order)
    except ValueError:
        if verbose:
            logger.warning("Induced graph has no cliques, setting induced width to 0")
        return 0


def simulate_variable_elimination(
    reduced_bn,
    elim_order,
    keep_vars,
    evidence_vars_set,
    verbose=False,
):
    """
    Simulate variable elimination to compute cost metrics.

    Args:
        reduced_bn: BayesianNetwork to perform elimination on
        elim_order: Order in which to eliminate variables
        keep_vars: Set of variables to keep (targets)
        evidence_vars_set: Set of evidence variables
        verbose: Whether to print debug information

    Returns:
        dict: Dictionary containing cost, max_factor_size, and factor_sizes
    """
    cost = 0
    max_factor_size = 0
    factor_sizes = []

    # Get effective cardinalities (evidence variables have cardinality 1)
    card = reduced_bn.get_cardinality()
    effective_card = card.copy()
    for evar in evidence_vars_set:
        if evar in effective_card:
            effective_card[evar] = 1

    moral = reduced_bn.to_markov_model()

    for step, x in enumerate(elim_order):
        nbrs = list(moral.neighbors(x))
        size = 1
        for v in nbrs + [x]:
            size *= effective_card[v]

        cost += size
        max_factor_size = max(max_factor_size, size)
        factor_sizes.append(size)

        if verbose:
            logger.debug(
                f"Step {step + 1}: Eliminating {x}, neighbors: {nbrs}, "
                f"factor size: {size}",
            )

        # Connect neighbors (fill-in) and remove x
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                moral.add_edge(nbrs[i], nbrs[j])
        moral.remove_node(x)

    # Calculate final factor size
    if keep_vars:
        final_factor_size = 1
        for v in keep_vars:
            final_factor_size *= effective_card[v]
        cost += final_factor_size
        max_factor_size = max(max_factor_size, final_factor_size)
        if verbose:
            logger.debug(
                f"Final factor (target variables): {sorted(keep_vars)}, "
                f"size: {final_factor_size}",
            )

    return {
        "cost": cost,
        "max_factor_size": max_factor_size,
        "factor_sizes": factor_sizes,
    }
