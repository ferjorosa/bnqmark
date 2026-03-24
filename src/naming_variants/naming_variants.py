"""
Shared naming variant generation for Bayesian Networks, DAGs, and Queries.

This module provides functionality to create naming variants of Bayesian Networks,
DAGs, and queries, enabling clean ablation testing where structure and CPTs remain
identical but node names vary. This is useful for evaluating how node naming affects
LLM probabilistic reasoning performance.

Naming Strategies:
    The module supports different naming strategies via NamingStrategy enum:
    - SIMPLE: V0, V1, V2, ... (clear and systematic)
    - CONFUSING: X_445aFa, S_af3a34, ... (random alphanumeric, harder to parse)
    - SEMANTIC: Rain, Sprinkler, WetGrass, ... (meaningful domain names)
    - MIXED: Combination of different strategies
    - DEFAULT: 0, 1, 2, ... (numeric labels)

This is shared code used by both bn_generation and dag_generation modules.
"""

from __future__ import annotations

import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.dag import NamingStrategy
from src.dag.generation.core.naming import generate_node_names


def create_name_mapping_from_strategy(
    old_node_names: list[str],
    naming_strategy: NamingStrategy,
    seed: int | None = None,
) -> dict[str, str]:
    """
    Create a name mapping from old names to new names using a naming strategy.

    This ensures consistent ordering - the first old node name maps to the first
    new name generated, the second to the second, etc.

    Args:
        old_node_names: List of existing node names (maintains order)
        naming_strategy: Naming strategy (NamingStrategy enum)
        seed: Random seed for reproducible name generation

    Returns:
        Dictionary mapping old_name -> new_name
    """
    n_nodes = len(old_node_names)
    new_names = generate_node_names(n_nodes, strategy=naming_strategy, seed=seed)

    # Create mapping: old_name -> new_name (preserving order)
    mapping = dict(zip(old_node_names, new_names, strict=False))
    return mapping


def create_bn_naming_variant(
    bn: DiscreteBayesianNetwork,
    naming_strategy: NamingStrategy,
    seed: int | None = None,
) -> DiscreteBayesianNetwork:
    """
    Create a new BN with relabeled node names.

    This function creates a new Bayesian Network with the same structure and CPTs
    as the original, but with different node names. This enables ablation testing
    where structure/probabilities are identical but names vary.

    Args:
        bn: Original DiscreteBayesianNetwork
        naming_strategy: Naming strategy to apply (NamingStrategy enum)
        seed: Random seed for reproducible name generation

    Returns:
        New DiscreteBayesianNetwork with relabeled nodes
    """
    # Get old node names in consistent order
    old_nodes = list(bn.nodes())

    # Create name mapping from strategy
    name_mapping = create_name_mapping_from_strategy(
        old_nodes,
        naming_strategy,
        seed=seed,
    )

    # 1. Relabel the DAG structure (edges)
    old_edges = list(bn.edges())
    new_edges = [(name_mapping.get(u, u), name_mapping.get(v, v)) for u, v in old_edges]
    new_bn = DiscreteBayesianNetwork(new_edges)

    # 2. Create new CPDs with relabeled names
    new_cpds: list[TabularCPD] = []
    for old_cpd in bn.get_cpds():
        # Map variable name
        old_variable = old_cpd.variable
        new_variable = name_mapping.get(old_variable, old_variable)

        # Map evidence variables (if any)
        # Use variables[1:] instead of get_evidence() to preserve the order
        # that matches the values array structure
        old_variables = old_cpd.variables  # [variable, ev1, ev2, ...]
        old_evidence_vars = old_variables[1:] if len(old_variables) > 1 else []
        new_evidence = None
        new_evidence_card = None
        if old_evidence_vars:
            new_evidence = [
                name_mapping.get(ev_var, ev_var) for ev_var in old_evidence_vars
            ]
            # Evidence card matches the order of variables[1:]
            old_cardinality = old_cpd.cardinality  # [var_card, ev1_card, ev2_card, ...]
            new_evidence_card = old_cardinality[1:].tolist()

        # Map state_names dict keys (variable names as keys)
        new_state_names: dict[str, list[str]] = {}
        for old_var_name, state_list in old_cpd.state_names.items():
            new_var_name = name_mapping.get(old_var_name, old_var_name)
            # Keep the same state labels (e.g., 's0', 's1', etc.)
            new_state_names[new_var_name] = state_list

        # Get values in the format expected by TabularCPD
        # Use get_values() which returns proper 2D format
        cpd_values = old_cpd.get_values()

        # Create new CPD with same values but new names
        new_cpd = TabularCPD(
            variable=new_variable,
            variable_card=old_cpd.variable_card,
            values=cpd_values,
            evidence=new_evidence,
            evidence_card=new_evidence_card,
            state_names=new_state_names,
        )
        new_cpds.append(new_cpd)

    # 3. Add CPDs to new model
    new_bn.add_cpds(*new_cpds)
    new_bn.check_model()

    return new_bn


def create_dag_naming_variant(
    dag: nx.DiGraph,
    naming_strategy: NamingStrategy,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Create a DAG with relabeled node names.

    Args:
        dag: Original NetworkX DiGraph
        naming_strategy: Naming strategy to apply (NamingStrategy enum)
        seed: Random seed for reproducible name generation

    Returns:
        New DiGraph with relabeled nodes
    """
    # Get old node names in consistent order
    old_nodes = list(dag.nodes())

    # Create name mapping from strategy
    name_mapping = create_name_mapping_from_strategy(
        old_nodes,
        naming_strategy,
        seed=seed,
    )

    return nx.relabel_nodes(dag, name_mapping, copy=True)
