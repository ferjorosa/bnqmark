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


def _scope_size(scope, effective_card):
    """Return the dense table size for a factor scope."""
    size = 1
    for var in scope:
        size *= effective_card[var]
    return size


def _get_factor_scopes(reduced_bn):
    """Return current factor scopes initialized from the network CPDs."""
    cpds = reduced_bn.get_cpds()
    if not isinstance(cpds, list):
        cpds = [cpds]
    return [set(cpd.variables) for cpd in cpds]


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
        return orderer.get_elimination_order(
            nodes=list(eliminate_vars),
            show_progress=False,
        )
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
        dict: Dictionary containing factor-size and scalar-operation metrics
    """
    cost = 0
    max_factor_size = 0
    factor_sizes = []
    scalar_additions = 0
    scalar_multiplications = 0
    scalar_additions_by_step = []
    scalar_multiplications_by_step = []

    # Get effective cardinalities (evidence variables have cardinality 1)
    card = reduced_bn.get_cardinality()
    effective_card = card.copy()
    for evar in evidence_vars_set:
        if evar in effective_card:
            effective_card[evar] = 1

    moral = reduced_bn.to_markov_model()
    factor_scopes = _get_factor_scopes(reduced_bn)

    for step, x in enumerate(elim_order):
        nbrs = list(moral.neighbors(x))
        size = _scope_size(nbrs + [x], effective_card)

        cost += size
        max_factor_size = max(max_factor_size, size)
        factor_sizes.append(size)

        relevant_scopes = [scope for scope in factor_scopes if x in scope]
        other_scopes = [scope for scope in factor_scopes if x not in scope]
        if relevant_scopes:
            joint_scope = set().union(*relevant_scopes)
            joint_size = _scope_size(joint_scope, effective_card)
            output_scope = joint_scope - {x}
            output_size = _scope_size(output_scope, effective_card)
            x_card = effective_card[x]

            step_multiplications = max(len(relevant_scopes) - 1, 0) * joint_size
            step_additions = max(x_card - 1, 0) * output_size

            scalar_multiplications += step_multiplications
            scalar_additions += step_additions
            scalar_multiplications_by_step.append(step_multiplications)
            scalar_additions_by_step.append(step_additions)

            factor_scopes = other_scopes + [output_scope]
        else:
            scalar_multiplications_by_step.append(0)
            scalar_additions_by_step.append(0)

        if verbose:
            logger.debug(
                f"Step {step + 1}: Eliminating {x}, neighbors: {nbrs}, "
                f"factor size: {size}, scalar multiplications: "
                f"{scalar_multiplications_by_step[-1]}, scalar additions: "
                f"{scalar_additions_by_step[-1]}",
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

    if factor_scopes:
        final_scope = set().union(*factor_scopes)
        final_join_size = _scope_size(final_scope, effective_card)
        final_join_multiplications = max(len(factor_scopes) - 1, 0) * final_join_size
    else:
        final_join_multiplications = 0

    target_factor_size = _scope_size(keep_vars, effective_card) if keep_vars else 1
    normalization_additions = max(target_factor_size - 1, 0)

    scalar_multiplications += final_join_multiplications
    scalar_additions += normalization_additions

    if verbose:
        logger.debug(
            f"Final join scalar multiplications: {final_join_multiplications}",
        )
        logger.debug(f"Normalization scalar additions: {normalization_additions}")

    return {
        "cost": cost,
        "max_factor_size": max_factor_size,
        "factor_sizes": factor_sizes,
        "scalar_additions": scalar_additions,
        "scalar_multiplications": scalar_multiplications,
        "scalar_additions_by_step": scalar_additions_by_step,
        "scalar_multiplications_by_step": scalar_multiplications_by_step,
        "final_join_multiplications": final_join_multiplications,
        "normalization_additions": normalization_additions,
    }
