"""
Core Bayesian Network generation from DAGs with controllable CPT properties.

This module provides the core functionality for building pgmpy DiscreteBayesianNetwork
instances from NetworkX DAGs, sampling Conditional Probability Tables (CPTs) according
to configurable parameters:

- Variable arity strategy (fixed or ranged)
- CPT skewness via Dirichlet(alpha)
- Determinism fraction: proportion of CPT columns set to 0/1

Example (API):
    >>> import networkx as nx
    >>> from src.dag_generation import generate_single_dag
    >>> dag, _, _ = generate_single_dag(5, 2, seed=42)
    >>> bn, meta = generate_discrete_bn_from_dag(
    ...     dag,
    ...     arity_strategy={"type": "range", "min": 2, "max": 3},
    ...     dirichlet_alpha=0.5,
    ...     determinism_fraction=0.0,
    ...     seed=123,
    ... )
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product as itertools_product
from typing import Any

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.bn.generation.core.types import ArityStrategy, BNGenerationMetadata

# ------------------------------
# Core generation functions
# ------------------------------


def _enumerate_parent_assignments(
    parents: Sequence[Any],
    parent_cards: dict[Any, int],
) -> list[tuple[int, ...]]:
    """
    Enumerate parent assignments as tuples of state indices in cartesian order.

    For parents [P1, P2] with cards [c1, c2], produces:
        (0,0), (0,1), ..., (0,c2-1), (1,0), ..., (c1-1, c2-1)

    Args:
        parents: Sequence of parent node identifiers
        parent_cards: Dictionary mapping parent -> cardinality

    Returns:
        List of tuples representing all possible parent state combinations
    """
    if not parents:
        return []
    ranges = [list(range(parent_cards[p])) for p in parents]
    # Product in natural order: parents[0] is slowest changing
    return list(itertools_product(*ranges))


def _sample_cpt_for_node(
    parents: Sequence[Any],
    var_card: int,
    parent_cards: dict[Any, int],
    dirichlet_alpha: float,
    determinism_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a CPT matrix for a node given its parents.

    Args:
        parents: Sequence of parent node identifiers
        var_card: Cardinality (number of states) for this node
        parent_cards: Dictionary mapping parent -> cardinality
        dirichlet_alpha: Dirichlet concentration parameter
            (<=1 skewed, 1 uniform, >1 flat)
        determinism_fraction: Fraction of CPT columns to set deterministically (0/1)
        rng: Random number generator

    Returns:
        Array of shape (var_card, product(parent_cards)) for conditional nodes,
        or shape (var_card,) for root nodes.
    """
    if not parents:
        # Root prior
        if determinism_fraction > 0.0 and rng.random() < determinism_fraction:
            one_hot = np.zeros(var_card, dtype=float)
            one_hot[int(rng.integers(0, var_card))] = 1.0
            return one_hot
        probs = rng.dirichlet([dirichlet_alpha] * var_card)
        return probs

    parent_assignments = _enumerate_parent_assignments(parents, parent_cards)
    num_cols = len(parent_assignments)
    values = np.zeros((var_card, num_cols), dtype=float)

    deterministic_cols = set()
    if determinism_fraction > 0.0:
        num_deterministic = int(round(determinism_fraction * num_cols))
        if num_deterministic > 0:
            deterministic_cols = set(
                rng.choice(num_cols, size=num_deterministic, replace=False),
            )

    for col in range(num_cols):
        if col in deterministic_cols:
            idx = int(rng.integers(0, var_card))
            values[idx, col] = 1.0
        else:
            values[:, col] = rng.dirichlet([dirichlet_alpha] * var_card)

    return values


def generate_discrete_bn_from_dag(
    dag: nx.DiGraph,
    arity_strategy: dict[str, Any] | ArityStrategy,
    dirichlet_alpha: float = 1.0,
    determinism_fraction: float = 0.0,
    seed: int | None = None,
) -> tuple[DiscreteBayesianNetwork, dict[str, Any]]:
    """
    Generate a DiscreteBayesianNetwork with sampled CPTs for the given DAG.

    Args:
        dag: A NetworkX DiGraph (assumed acyclic)
        arity_strategy: dict or ArityStrategy specifying node cardinalities
        dirichlet_alpha: Dirichlet concentration for CPT columns
            (<=1 skewed, 1 uniform, >1 flat)
        determinism_fraction: fraction of CPT columns set to deterministic 0/1
            (0.0 recommended by default)
        seed: RNG seed

    Returns:
        (model, metadata) where model is a pgmpy DiscreteBayesianNetwork
        and metadata includes chosen arities and generation parameters.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Input graph must be a DAG")

    rng = np.random.default_rng(seed)
    if isinstance(arity_strategy, dict):
        strat = ArityStrategy(**arity_strategy)  # ty: ignore
    else:
        strat = arity_strategy

    nodes = list(dag.nodes())
    node_cards: dict[Any, int] = strat.draw_cardinalities(nodes, rng)

    model = DiscreteBayesianNetwork(list(dag.edges()))

    # Create state names per node
    state_names: dict[Any, list[str]] = {
        n: [f"s{i}" for i in range(node_cards[n])] for n in nodes
    }

    # Build CPDs in topological order
    cpds: list[TabularCPD] = []
    for node in nx.topological_sort(dag):
        parents = list(dag.predecessors(node))
        var_card = node_cards[node]
        parent_cards = {p: node_cards[p] for p in parents}

        values = _sample_cpt_for_node(
            parents=parents,
            var_card=var_card,
            parent_cards=parent_cards,
            dirichlet_alpha=dirichlet_alpha,
            determinism_fraction=determinism_fraction,
            rng=rng,
        )

        if parents:
            evidence = parents
            evidence_card = [node_cards[p] for p in parents]
            cpd = TabularCPD(
                variable=node,
                variable_card=var_card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
                state_names={
                    **{node: state_names[node]},
                    **{p: state_names[p] for p in parents},
                },
            )
        else:
            cpd = TabularCPD(
                variable=node,
                variable_card=var_card,
                values=values.reshape(var_card, 1),
                state_names={node: state_names[node]},
            )

        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()

    # Create metadata as dataclass for type safety
    metadata = BNGenerationMetadata(
        node_cardinalities=node_cards,
        dirichlet_alpha=dirichlet_alpha,
        determinism_fraction=determinism_fraction,
        seed=seed,
    )
    # Return as dict for backward compatibility
    return model, metadata.to_dict()


def generate_variants_for_dag(
    dag: nx.DiGraph,
    variants: list[dict[str, Any]],
    base_seed: int = 0,
) -> list[tuple[DiscreteBayesianNetwork, dict[str, Any]]]:
    """
    Generate multiple BN variants for a single DAG.

    'variants' is a list of dicts with keys accepted by generate_discrete_bn_from_dag.
    Each variant gets a deterministically offset seed (base_seed + idx).

    Args:
        dag: NetworkX DiGraph to generate variants for
        variants: List of configuration dicts for generate_discrete_bn_from_dag
        base_seed: Base seed for reproducibility (each variant gets offset seed)

    Returns:
        List of (bn, metadata) tuples
    """
    results: list[tuple[DiscreteBayesianNetwork, dict[str, Any]]] = []
    for idx, cfg in enumerate(variants):
        cfg = dict(cfg)  # shallow copy
        if "seed" not in cfg:
            cfg["seed"] = int(base_seed + idx * 9973)  # prime step
        bn, meta = generate_discrete_bn_from_dag(dag, **cfg)
        meta["variant_index"] = idx
        results.append((bn, meta))
    return results
