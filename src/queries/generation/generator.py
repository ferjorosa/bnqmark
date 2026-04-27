"""
Core Query Generation Module.

This module provides the fundamental building block for query generation:
generating a SINGLE query with strict constraint enforcement.

For generating MULTIPLE queries, use the sweep module instead:
    from src.queries.sweep import generate_queries_with_sampling
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from src.utils.bn_utils import get_cpds_list
from src.utils.distance_utils import compute_shortest_distance

from .types import QueryGenerationContext, QueryGenerationMetadata, QuerySpec


def _to_nx_dag(model: DiscreteBayesianNetwork) -> nx.DiGraph:
    """Convert pgmpy BayesianNetwork to NetworkX DiGraph for analysis."""
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())
    return graph


def _all_state_labels(model: DiscreteBayesianNetwork) -> dict[str, list[str]]:
    """Extract all possible state labels for each variable in the network."""
    labels: dict[str, list[str]] = {}
    for cpd in get_cpds_list(model):
        var = cpd.variable
        labels[var] = list(cpd.state_names[var])  # ty: ignore
    return labels


def _choose_states(
    rng: np.random.Generator,
    state_labels: dict[str, list[str]],
    nodes: Sequence[str],
) -> dict[str, str]:
    """Randomly choose states for the given nodes."""
    return {n: rng.choice(state_labels[n]).item() for n in nodes}


def _make_query_key(
    targets: dict[str, str],
    evidence: dict[str, str],
) -> tuple[Any, ...]:
    """Create a hashable key for query deduplication."""
    return (
        tuple(sorted(targets.items())),
        tuple(sorted(evidence.items())) if evidence else (),
    )


def _compute_query_probabilities(
    model: DiscreteBayesianNetwork,
    query_vars: list[str],
    query_states: list[str],
    evidence: dict[str, str] | None,
) -> tuple[float | None, float | None]:
    """
    Compute exact posterior and prior probabilities for a query.

    Args:
        model: Bayesian network
        query_vars: List of query variable names
        query_states: List of query states (same order as query_vars)
        evidence: Evidence dict or None

    Returns:
        Tuple of (posterior_probability, prior_probability)
        Returns (None, None) if computation fails
    """
    try:
        from pgmpy.inference import VariableElimination

        infer = VariableElimination(model)

        # Compute posterior (with evidence)
        try:
            result = infer.query(
                variables=query_vars,  # ty: ignore
                evidence=evidence,  # ty: ignore
                show_progress=False,
            )
            assignment = dict(zip(query_vars, query_states, strict=False))
            posterior_prob = float(result.get_value(**assignment))
        except Exception:
            posterior_prob = None

        # Compute prior (no evidence)
        try:
            prior_result = infer.query(
                variables=query_vars,  # ty: ignore
                evidence=None,
                show_progress=False,
            )
            prior_assignment = dict(zip(query_vars, query_states, strict=False))
            prior_prob = float(prior_result.get_value(**prior_assignment))
        except Exception:
            prior_prob = None

        return posterior_prob, prior_prob

    except Exception:
        return None, None


def _select_query_nodes(
    ctx: QueryGenerationContext,
    count: int,
) -> list[str]:
    """Randomly select query nodes."""
    return list(ctx.rng.choice(ctx.nodes, size=count, replace=False))


def _filter_evidence_pool(
    ctx: QueryGenerationContext,
    query_nodes: list[str],
) -> set[str]:
    """Pre-filter nodes that satisfy distance constraints from query nodes."""
    pool = set()

    for candidate in ctx.nodes:
        if candidate in query_nodes:
            continue

        # Check if this candidate satisfies distance constraints from ALL query nodes
        valid_candidate = True

        for q_node in query_nodes:
            d = compute_shortest_distance(ctx.G, q_node, candidate, no_path_value=None)
            # If no path exists or distance violates constraint,
            # this candidate is invalid
            if d is None or not (ctx.dmin <= d <= ctx.dmax):
                valid_candidate = False
                break

        # Only add candidate if ALL distances are within constraints
        if valid_candidate:
            pool.add(candidate)

    return pool


def _select_evidence_nodes(
    ctx: QueryGenerationContext,
    query_nodes: list[str],
    count: int,
) -> list[str] | None:
    """Select evidence nodes from pre-filtered pool."""
    if count == 0:
        return []

    pool = _filter_evidence_pool(ctx, query_nodes)
    if len(pool) < count:
        return None

    return list(ctx.rng.choice(list(pool), size=count, replace=False))


def _create_query_representation(
    ctx: QueryGenerationContext,
    query_nodes: list[str],
    evidence_nodes: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign states and create query representation."""
    q_states = _choose_states(ctx.rng, ctx.state_labels, query_nodes)
    e_states = _choose_states(ctx.rng, ctx.state_labels, evidence_nodes)

    targets = {n: q_states[n] for n in query_nodes}
    evidence = {n: e_states[n] for n in evidence_nodes}

    return targets, evidence


def _check_uniqueness(
    targets: dict[str, str],
    evidence: dict[str, str],
    seen_queries: set[tuple[Any, ...]] | None,
) -> bool:
    """Check if query is unique. Returns True if unique or no dedup requested."""
    if seen_queries is None:
        return True

    query_key = _make_query_key(targets, evidence)
    return query_key not in seen_queries


def _check_probability_threshold(
    ctx: QueryGenerationContext,
    targets: dict[str, str],
    evidence: dict[str, str],
    min_abs_diff: float | None,
) -> tuple[bool, float | None, float | None]:
    """
    Check probability threshold constraint.

    Returns:
        Tuple of (passes_threshold, posterior_prob, prior_prob)
    """
    if min_abs_diff is None:
        return True, None, None

    query_vars = list(targets.keys())
    query_states = list(targets.values())

    posterior_prob, prior_prob = _compute_query_probabilities(
        ctx.model,
        query_vars,
        query_states,
        evidence if evidence else None,
    )

    if posterior_prob is None or prior_prob is None:
        return False, None, None

    abs_diff = abs(posterior_prob - prior_prob)
    passes = abs_diff >= min_abs_diff

    return passes, posterior_prob, prior_prob


def _compute_min_distance(
    ctx: QueryGenerationContext,
    query_nodes: list[str],
    evidence_nodes: list[str],
) -> int:
    """
    Compute minimum distance between any query-evidence pair.

    Note: This assumes all nodes are connected (filtered by _filter_evidence_pool).
    """
    if not evidence_nodes:
        return 0

    distances = [
        d
        for d in (
            compute_shortest_distance(ctx.G, t, e, no_path_value=None)
            for t in query_nodes
            for e in evidence_nodes
        )
        if d is not None
    ]

    return min(distances) if distances else 0


def _build_query_spec(
    targets: dict[str, str],
    evidence: dict[str, str],
    distance_bucket: tuple[int, int],
    min_dist: int,
    posterior_prob: float | None,
    prior_prob: float | None,
) -> QuerySpec:
    """Build final QuerySpec with metadata."""
    meta = QueryGenerationMetadata(
        num_query_nodes=len(targets),
        num_evidence_nodes=len(evidence),
        distance_bucket=distance_bucket,
        min_target_evidence_distance=min_dist,
    )

    query_spec = QuerySpec(targets=targets, evidence=evidence, meta=meta)

    if posterior_prob is not None and prior_prob is not None:
        query_spec.posterior_probability = posterior_prob
        query_spec.prior_probability = prior_prob

    return query_spec


def generate_single_query(
    model: DiscreteBayesianNetwork,
    query_node_count: int,
    evidence_count: int,
    distance_bucket: tuple[int, int],
    *,
    max_tries: int = 50,
    min_abs_diff: float | None = None,
    seen_queries: set[tuple[Any, ...]] | None = None,
    seed: int | None = None,
) -> QuerySpec | None:
    """
    Generate a SINGLE query meeting all specified constraints.

    This is the core function for generating one query with strict constraint
    checking. It tries up to max_tries times to find a query that satisfies:
    - Distance constraints (always enforced)
    - Probability threshold (if min_abs_diff provided)
    - Uniqueness (if seen_queries provided)

    Returns None if constraints cannot be satisfied within max_tries.
    No fallback mechanism - constraints are always honored or generation fails.

    Args:
        model: pgmpy DiscreteBayesianNetwork to generate query from
        query_node_count: Number of query variables (target nodes)
        evidence_count: Number of evidence variables
        distance_bucket: (min_distance, max_distance) tuple for evidence placement
                        Controls how far evidence is from query variables
        max_tries: Maximum number of sampling attempts (default: 500)
        min_abs_diff: Optional minimum |posterior - prior| threshold
                     If provided, queries must meet this threshold
        seen_queries: Optional set of query keys for deduplication
                     If provided, generated query must be unique
        seed: Random seed for reproducibility (default: None)

    Returns:
        QuerySpec if successful, None if constraints cannot be satisfied
        If successful and min_abs_diff was provided, the QuerySpec will have
        posterior_probability and prior_probability set.
    """
    # Setup context
    ctx = QueryGenerationContext(
        model=model,
        G=_to_nx_dag(model),
        nodes=list(model.nodes()),
        state_labels=_all_state_labels(model),
        rng=np.random.default_rng(seed),
        distance_bucket=distance_bucket,
    )

    # Validate inputs
    if query_node_count > len(ctx.nodes):
        return None
    if evidence_count > len(ctx.nodes) - query_node_count:
        return None

    # Try to generate valid query
    for _ in range(max_tries):
        # 1. Select nodes
        query_nodes = _select_query_nodes(ctx, query_node_count)

        evidence_nodes = _select_evidence_nodes(ctx, query_nodes, evidence_count)
        if evidence_nodes is None:
            continue  # Distance constraints not satisfiable

        # 2. Create query representation
        targets, evidence = _create_query_representation(
            ctx,
            query_nodes,
            evidence_nodes,
        )

        # 3. Check constraints
        if not _check_uniqueness(targets, evidence, seen_queries):
            continue  # Duplicate query

        passes_threshold, posterior_prob, prior_prob = _check_probability_threshold(
            ctx,
            targets,
            evidence,
            min_abs_diff,
        )
        if not passes_threshold:
            continue  # Doesn't meet probability threshold

        # 4. Success! Build and return QuerySpec
        min_dist = _compute_min_distance(ctx, query_nodes, evidence_nodes)
        return _build_query_spec(
            targets,
            evidence,
            distance_bucket,
            min_dist,
            posterior_prob,
            prior_prob,
        )

    # Failed after max_tries
    return None


__all__ = ["generate_single_query"]
