"""
Main entry point for Bayesian Network generation.

This module provides the primary API for generating Bayesian Networks:

**generate_single_bn()** - Generate one complete BN from scratch

See function docstring for detailed examples.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from src.bn.generation.core.generation import generate_discrete_bn_from_dag
from src.bn.generation.core.types import ArityStrategy
from src.dag import NamingStrategy, generate_single_dag


def generate_single_bn(
    n_nodes: int,
    target_treewidth: int,
    arity_strategy: dict[str, Any] | ArityStrategy,
    dirichlet_alpha: float = 1.0,
    determinism_fraction: float = 0.0,
    node_naming: NamingStrategy = NamingStrategy.SIMPLE,
    seed: int | None = None,
) -> tuple[DiscreteBayesianNetwork, nx.DiGraph, dict[str, Any]]:
    """
    Generate a single Bayesian Network from scratch.

    This is the simplest API - generates one complete BN with a random DAG structure
    and sampled CPTs. Perfect for quick prototyping or single network generation.

    Args:
        n_nodes: Number of variables in the network
        target_treewidth: Target structural complexity
            (1=simple tree, higher=more complex)
            Controls inference difficulty - higher treewidth = harder inference
        arity_strategy: How to assign cardinalities (number of states) to variables
                       Can be:
                       - Dict: {"type": "fixed", "fixed": 3} for all vars with 3 states
                        - Dict: {"type": "range", "min": 2, "max": 4} for
                          random 2-4 states
                       - ArityStrategy object
        dirichlet_alpha: CPT distribution skewness (< 1 skewed, 1 uniform, > 1 flat)
                        Default 1.0 creates uniform distributions
        determinism_fraction: Fraction of deterministic relationships (0.0-1.0)
                             0.0 = all probabilistic (recommended)
                             1.0 = all deterministic
        node_naming: Naming strategy for variables (NamingStrategy enum)
                    Options: NamingStrategy.SIMPLE, NamingStrategy.CONFUSING,
                            NamingStrategy.SEMANTIC, NamingStrategy.MIXED
        seed: Random seed for reproducibility (None for random)

    Returns:
        Tuple of (bayesian_network, dag, metadata):
        - bayesian_network: pgmpy DiscreteBayesianNetwork with CPTs
        - dag: NetworkX DiGraph structure (same node names as BN)
        - metadata: Dictionary with generation parameters and network properties

    Example:
        >>> # Generate a simple network
        >>> bn, dag, meta = generate_single_bn(
        ...     n_nodes=10,
        ...     target_treewidth=3,
        ...     arity_strategy={"type": "range", "min": 2, "max": 4},
        ...     seed=42,
        ... )
        >>> print(f"Generated BN with {bn.number_of_nodes()} nodes")
        >>> print(f"Achieved treewidth: {meta['achieved_treewidth']}")

        >>> # Generate with semantic names
        >>> bn, dag, meta = generate_single_bn(
        ...     n_nodes=5,
        ...     target_treewidth=2,
        ...     arity_strategy={"type": "fixed", "fixed": 2},
        ...     node_naming=NamingStrategy.SEMANTIC,
        ...     dirichlet_alpha=0.5,  # Skewed distributions
        ...     seed=123,
        ... )
    """
    # Generate DAG structure
    dag, achieved_tw, dag_meta = generate_single_dag(
        n_nodes,
        target_treewidth,
        node_naming=node_naming,
        seed=seed,
    )

    # Generate BN from DAG
    bn, bn_meta = generate_discrete_bn_from_dag(
        dag,
        arity_strategy,
        dirichlet_alpha,
        determinism_fraction,
        seed,
    )

    # Combine metadata
    combined_meta = {
        **bn_meta,
        "n_nodes": n_nodes,
        "target_treewidth": target_treewidth,
        "achieved_treewidth": achieved_tw,
        "node_naming": node_naming.value,
        **dag_meta,
    }

    return bn, dag, combined_meta
