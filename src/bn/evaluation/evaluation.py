"""
Evaluation and verification functions for Bayesian Network naming variants.

This module provides functions to verify that naming variants preserve
structure and CPTs correctly, which is essential for ablation testing.
"""

from __future__ import annotations

from pgmpy.models import DiscreteBayesianNetwork

from src.utils.bn_utils import get_cpds_list


def compare_bn_structures(
    bn1: DiscreteBayesianNetwork,
    bn2: DiscreteBayesianNetwork,
    name_mapping: dict[str, str],
) -> bool:
    """
    Compare if two BNs have the same structure (edges, ignoring node names).

    Args:
        bn1: First Bayesian Network (original)
        bn2: Second Bayesian Network (variant)
        name_mapping: Dictionary mapping bn1 node names -> bn2 node names

    Returns:
        True if structures are identical (same number of edges,
        same connectivity pattern)
    """
    if bn1.number_of_nodes() != bn2.number_of_nodes():
        return False
    if bn1.number_of_edges() != bn2.number_of_edges():
        return False

    # Use name_mapping to compare edges directly
    mapped_edges = {
        (name_mapping.get(u, u), name_mapping.get(v, v)) for u, v in bn1.edges()
    }
    variant_edges = set(bn2.edges())
    return mapped_edges == variant_edges


def compare_cpt_values(
    bn1: DiscreteBayesianNetwork,
    bn2: DiscreteBayesianNetwork,
    name_mapping: dict[str, str],
) -> bool:
    """
    Compare if two BNs have the same CPT values (ignoring variable names).

    Args:
        bn1: First Bayesian Network (original)
        bn2: Second Bayesian Network (variant)
        name_mapping: Dictionary mapping bn1 node names -> bn2 node names

    Returns:
        True if all CPT values are identical (within numerical precision)
    """
    cpds1 = {cpd.variable: cpd for cpd in get_cpds_list(bn1)}
    cpds2 = {cpd.variable: cpd for cpd in get_cpds_list(bn2)}

    # Use name_mapping to compare corresponding CPDs
    for old_var, old_cpd in cpds1.items():
        new_var = name_mapping.get(old_var, old_var)  # ty: ignore
        if new_var not in cpds2:
            return False

        new_cpd = cpds2[new_var]

        # Compare values (should be identical)
        if not (old_cpd.values == new_cpd.values).all():
            return False

        # Compare cardinalities
        if old_cpd.variable_card != new_cpd.variable_card:
            return False

        # Compare evidence variables using mapping (order matters for CPD structure)
        # Use variables[1:] instead of get_evidence() to match the order
        # used when creating CPDs
        old_evidence_vars = old_cpd.variables[1:] if len(old_cpd.variables) > 1 else []
        new_evidence_vars = new_cpd.variables[1:] if len(new_cpd.variables) > 1 else []
        if len(old_evidence_vars) != len(new_evidence_vars):
            return False

        # Map evidence variables and compare in order
        # (order matters for CPD values structure)
        mapped_new_evidence = [name_mapping.get(ev, ev) for ev in old_evidence_vars]
        if mapped_new_evidence != new_evidence_vars:
            return False

    return True


def verify_naming_variant(
    original_bn: DiscreteBayesianNetwork,
    variant_bn: DiscreteBayesianNetwork,
    name_mapping: dict[str, str],
) -> tuple[bool, bool, str]:
    """
    Verify that a naming variant preserves structure and CPTs.

    Args:
        original_bn: Original Bayesian Network
        variant_bn: Naming variant Bayesian Network
        name_mapping: Dictionary mapping original node names -> variant node names

    Returns:
        Tuple of (structure_match, cpt_match, error_message)
        - structure_match: True if structure is preserved
        - cpt_match: True if CPT values are preserved
        - error_message: Empty string if both match, otherwise descriptive error message
    """
    structure_match = compare_bn_structures(original_bn, variant_bn, name_mapping)
    cpt_match = compare_cpt_values(original_bn, variant_bn, name_mapping)

    error_message = ""
    if not structure_match:
        error_message += "Structure does not match. "
    if not cpt_match:
        error_message += "CPT values do not match. "

    return structure_match, cpt_match, error_message
