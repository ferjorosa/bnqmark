"""
Main query complexity computation.

This module provides the primary function for computing query complexity metrics
by orchestrating network reduction, node analysis, and elimination complexity.
"""

import logging

import numpy as np

from .elimination import (
    compute_elimination_order,
    compute_induced_width,
    simulate_variable_elimination,
)
from .network_reduction import build_reduced_network
from .node_analysis import (
    find_conditionally_independent_vars,
    identify_barren_nodes,
)
from .types import ComplexityMetrics

logger = logging.getLogger(__name__)


def compute_query_complexity(bn, target_nodes, evidence_nodes, verbose=False):
    """
    Compute query complexity by removing independent/barren nodes.

    This function provides a comprehensive analysis of query computational complexity:
    1. Creates a copy of the BN to avoid modifying the original
    2. Identifies and removes variables independent of targets given evidence
    3. Identifies and removes barren nodes
    4. Computes variable elimination complexity on the reduced network

    The complexity metrics include:
    - Induced width: Maximum clique size in the elimination graph
    - Total cost: Sum of all intermediate factor sizes during elimination
    - Max factor size: Largest intermediate factor encountered
    - Elimination order: Optimal order for variable elimination

    Parameters:
        bn: Bayesian network (pgmpy DiscreteBayesianNetwork)
        target_nodes: List of target/query variable names
        evidence_nodes: List of evidence variable names (or empty list)
        verbose: If True, print detailed progress information

    Returns:
        ComplexityMetrics: Comprehensive complexity metrics dataclass containing:
            - original_num_vars: Number of variables in original network
            - reduced_num_vars: Number of variables after reduction
            - num_independent_vars: Number of independent variables removed
            - num_barren_vars: Number of barren variables removed
            - induced_width: Induced width of elimination order
            - total_cost: Total computational cost (sum of factor sizes)
            - max_factor_size: Maximum intermediate factor size
            - scalar_additions: Total scalar additions for dense tabular VE
            - scalar_multiplications: Total scalar multiplications for dense tabular VE
            - elimination_order: Optimal elimination order
            - log_total_cost: Log2 of total cost
            - log_max_factor_size: Log2 of max factor size
            - And many more fields (see ComplexityMetrics dataclass)

    Example:
        >>> from src.queries.complexity import compute_query_complexity
        >>> # Analyze query complexity
        >>> complexity = compute_query_complexity(
        ...     bn=bayesian_network,
        ...     target_nodes=["Disease"],
        ...     evidence_nodes=["Symptom1", "Symptom2"],
        ...     verbose=True,
        ... )
        >>> print(f"Induced width: {complexity.induced_width}")
        >>> print(f"Total cost: {complexity.total_cost:,}")
        >>> print(f"Variables removed: {complexity.num_vars_removed}")

    Note:
        This function performs exact complexity analysis by simulating variable
        elimination. For large networks, this can be computationally expensive.
        The complexity metrics provide insights into why certain queries are
        harder than others and can guide algorithm selection.
    """
    if verbose:
        logger.debug(
            f"Original network: {len(bn.nodes())} nodes, {bn.number_of_edges()} edges",
        )

    # Step 1: Find conditionally independent variables
    independent_vars = find_conditionally_independent_vars(
        bn,
        target_nodes,
        evidence_nodes,
    )
    if verbose:
        logger.debug(
            f"Found {len(independent_vars)} independent variables: "
            f"{sorted(independent_vars)}",
        )

    # Step 2: Create intermediate network with independent variables removed
    vars_after_independent_removal = set(bn.nodes()) - independent_vars
    intermediate_bn = build_reduced_network(bn, vars_after_independent_removal, verbose)

    if verbose:
        logger.debug(
            f"After removing independent variables: "
            f"{len(intermediate_bn.nodes())} nodes, "
            f"{intermediate_bn.number_of_edges()} edges",
        )

    # Step 3: Find barren nodes in the intermediate network
    barren_vars = set()
    identify_barren_nodes(intermediate_bn, target_nodes, evidence_nodes, barren_vars)
    if verbose:
        logger.debug(
            f"Found {len(barren_vars)} barren variables: {sorted(barren_vars)}",
        )

    # Step 4: Create final reduced network (removing both independent and barren nodes)
    vars_to_remove = independent_vars | barren_vars
    vars_to_keep = set(bn.nodes()) - vars_to_remove

    if verbose:
        logger.debug(
            f"Removing {len(vars_to_remove)} variables "
            f"(independent: {len(independent_vars)}, "
            f"barren: {len(barren_vars)}), keeping {len(vars_to_keep)} variables",
        )
        logger.debug(f"Variables to keep: {sorted(vars_to_keep)}")

    reduced_bn = build_reduced_network(intermediate_bn, vars_to_keep, verbose)

    if verbose:
        logger.debug(
            f"Reduced network: {len(reduced_bn.nodes())} nodes, "
            f"{reduced_bn.number_of_edges()} edges",
        )

    # Step 5: Compute complexity on reduced network
    reduced_bn.check_model()

    if verbose:
        card = reduced_bn.get_cardinality()
        logger.debug(f"Variable cardinalities: {dict(card)}")

    # Identify variables to keep vs eliminate
    all_vars = set(reduced_bn.nodes())
    target_vars_set = set(target_nodes)
    evidence_vars_set = set(evidence_nodes)
    keep_vars = target_vars_set & all_vars
    eliminate_vars = all_vars - keep_vars

    if verbose:
        logger.debug(f"Variables to keep (targets): {sorted(keep_vars)}")
        logger.debug(f"Variables to eliminate: {sorted(eliminate_vars)}")
        logger.debug(f"Evidence variables: {sorted(evidence_vars_set & all_vars)}")

    # Compute elimination order
    elim_order = compute_elimination_order(reduced_bn, eliminate_vars)
    complete_elim_order = elim_order + list(keep_vars)

    if verbose:
        logger.debug(f"Elimination order: {elim_order}")
        if elim_order:
            logger.debug(f"Complete elimination order: {complete_elim_order}")

    # Compute induced width
    induced_width = compute_induced_width(reduced_bn, elim_order, keep_vars, verbose)

    if verbose:
        logger.debug(f"Induced width: {induced_width}")

    # Simulate variable elimination
    elim_results = simulate_variable_elimination(
        reduced_bn,
        elim_order,
        keep_vars,
        evidence_vars_set,
        verbose,
    )

    # Build complexity metrics dataclass
    complexity_metrics = ComplexityMetrics(
        original_num_vars=len(bn.nodes()),
        reduced_num_vars=len(all_vars),
        num_independent_vars=len(independent_vars),
        num_barren_vars=len(barren_vars),
        num_vars_removed=len(vars_to_remove),
        num_edges=reduced_bn.number_of_edges(),
        num_target_vars=len(target_nodes),
        num_evidence_vars=len(evidence_nodes),
        num_eliminated_vars=len(elim_order),
        elimination_order=elim_order,
        complete_elimination_order=complete_elim_order,
        induced_width=induced_width,
        total_cost=elim_results["cost"],
        max_factor_size=elim_results["max_factor_size"],
        avg_factor_size=(elim_results["cost"] / len(elim_order) if elim_order else 0),
        factor_sizes=elim_results["factor_sizes"],
        scalar_additions=elim_results["scalar_additions"],
        scalar_multiplications=elim_results["scalar_multiplications"],
        scalar_additions_by_step=elim_results["scalar_additions_by_step"],
        scalar_multiplications_by_step=elim_results["scalar_multiplications_by_step"],
        final_join_multiplications=elim_results["final_join_multiplications"],
        normalization_additions=elim_results["normalization_additions"],
        log_total_cost=(
            np.log2(elim_results["cost"]) if elim_results["cost"] > 0 else 0
        ),
        log_max_factor_size=(
            np.log2(elim_results["max_factor_size"])
            if elim_results["max_factor_size"] > 0
            else 0
        ),
        keep_vars=sorted(keep_vars),
        eliminate_vars=sorted(eliminate_vars),
    )

    if verbose:
        logger.debug("\nQuery Complexity Summary:")
        logger.debug(f"  Original variables: {len(bn.nodes())}")
        logger.debug(
            f"  Variables removed (independent + barren): {len(vars_to_remove)}",
        )
        logger.debug(f"  Reduced network variables: {len(all_vars)}")
        logger.debug(f"  Variables eliminated: {len(elim_order)}/{len(all_vars)}")
        logger.debug(f"  Target variables kept: {sorted(keep_vars)}")
        logger.debug(f"  Induced width: {induced_width}")
        logger.debug(f"  Total factor work: {elim_results['cost']:,}")
        logger.debug(
            f"  Max intermediate factor size: {elim_results['max_factor_size']:,}",
        )
        logger.debug(f"  Scalar additions: {elim_results['scalar_additions']:,}")
        logger.debug(
            f"  Scalar multiplications: {elim_results['scalar_multiplications']:,}",
        )
        avg_size = elim_results["cost"] / len(elim_order) if elim_order else 0
        logger.debug(f"  Average factor size: {avg_size:.1f}")
        log_cost = np.log2(elim_results["cost"]) if elim_results["cost"] > 0 else 0
        log_max = (
            np.log2(elim_results["max_factor_size"])
            if elim_results["max_factor_size"] > 0
            else 0
        )
        logger.debug(f"  Log2(total cost): {log_cost:.2f}")
        logger.debug(f"  Log2(max factor size): {log_max:.2f}")

    return complexity_metrics
