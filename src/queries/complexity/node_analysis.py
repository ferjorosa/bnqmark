"""
Node analysis functions for query complexity.

This module provides functions for identifying nodes that can be removed from
a Bayesian Network without affecting query results, including:
- Conditionally independent variables (d-separation)
- Barren nodes (nodes with no relevant descendants)
"""

import logging

logger = logging.getLogger(__name__)


def is_barren(node, bn, query_vars, evidence_nodes, barren_vars):
    """
    Check if a node is barren.

    A barren node is one that can be removed from the network without affecting
    the query result. This includes:
    - Leaf nodes that are not query targets or evidence
    - Nodes whose only descendants are also barren

    Args:
        node: Node identifier to check
        bn: pgmpy BayesianNetwork
        query_vars: Set of query/target variable names
        evidence_nodes: Set of evidence variable names
        barren_vars: Set of already identified barren variables

    Returns:
        bool: True if the node is barren and can be removed
    """
    if node in query_vars or node in evidence_nodes:
        return False  # Targets and evidence are never barren

    children = list(bn.get_children(node))
    if len(children) == 0:
        return True  # Leaf node is barren

    # Check if all descendants are barren
    descendants = set()
    stack = [node]
    visited = {node}
    while stack:
        current = stack.pop()
        for child in bn.get_children(current):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                stack.append(child)

    # Remove query and evidence from descendants check
    descendants = descendants

    if len(descendants) == 0:
        return True

    # All descendants must be barren
    all_descendants_barren = True
    for desc in descendants:
        if desc not in barren_vars:
            all_descendants_barren = False
            break

    return all_descendants_barren


def identify_barren_nodes(bn, query_vars, evidence_nodes, barren_vars):
    """
    Iteratively identify barren nodes in the Bayesian Network.

    This function uses a fixed-point algorithm to identify all barren nodes.
    It repeatedly scans the network until no new barren nodes are found.

    Args:
        bn: pgmpy BayesianNetwork
        query_vars: Set of query/target variable names
        evidence_nodes: Set of evidence variable names
        barren_vars: Set to update with identified barren variables (modified in-place)

    Returns:
        Set: The updated barren_vars set containing all identified barren nodes
    """
    changed = True
    while changed:
        changed = False
        for node in bn.nodes():
            if node in query_vars or node in evidence_nodes or node in barren_vars:
                continue

            if is_barren(node, bn, query_vars, evidence_nodes, barren_vars):
                barren_vars.add(node)
                changed = True
    return barren_vars


def find_conditionally_independent_vars(bn, query_vars, evidence_nodes):
    """
    Find variables independent of query vars given evidence.

    Uses d-separation to identify variables that are conditionally independent
    of the query variables given the evidence. These variables can be safely
    removed without affecting the query result.

    Args:
        bn: pgmpy BayesianNetwork
        query_vars: Set of query/target variable names
        evidence_nodes: Set of evidence variable names

    Returns:
        Set: Variables that are conditionally independent of all query variables
    """
    independent_vars = set()
    for var in bn.nodes():
        if var in query_vars or var in evidence_nodes:
            continue  # Skip query and evidence variables

        # Check if this variable is independent of all query variables given evidence
        is_independent = True
        for query_var in query_vars:
            # Use d-separation: variable is independent of query_var given evidence
            if bn.is_dconnected(var, query_var, observed=evidence_nodes):
                is_independent = False
                break

        if is_independent:
            independent_vars.add(var)
    return independent_vars
