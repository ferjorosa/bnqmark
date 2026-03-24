"""
Query generation with parameter sweeps and threshold-based sampling.

This module provides high-level functions for generating multiple queries from
Bayesian Networks with systematic parameter sweeps. All constraint checking
(distance, threshold, deduplication) is delegated to generate_single_query(),
making this module a simple orchestration layer.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from src.queries.generation import QuerySpec, generate_single_query
from src.queries.generation.generator import _make_query_key


def generate_queries(
    bn: DiscreteBayesianNetwork,
    queries_per_combination: int,
    query_node_counts: tuple[int, ...],
    evidence_counts: tuple[int, ...],
    distance_buckets: list[tuple[int, int]],
    min_abs_diff: float,
    max_tries_per_query: int,
    strict_mode: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> list[QuerySpec]:
    """
    Generate queries using sampling approach with comprehensive constraint checking.

    This function orchestrates query generation by:
    1. Creating all combinations of (query_node_count, evidence_count)
    2. For each combination, generating queries_per_combination instances
    3. Delegating all constraint checking to generate_single_query():
       - Distance constraints (always enforced)
       - Probability threshold (via min_abs_diff)
       - Deduplication (via seen_queries)

    All generated queries are guaranteed to meet constraints.

    Args:
        bn: Bayesian network to generate queries from
        queries_per_combination: Number of query instances per combination
        query_node_counts: Tuple of possible query node counts
        evidence_counts: Tuple of possible evidence node counts
        distance_buckets: List of (min_distance, max_distance) tuples for
            evidence placement. Generates queries for each distance bucket
        min_abs_diff: Minimum absolute difference threshold between posterior
            and prior
        max_tries_per_query: Maximum sampling attempts per query
        strict_mode: If True, raise exception on failure; if False, skip failed
            queries (default: False)
        verbose: If True, log progress and failure details during generation
            (default: False)
        seed: Random seed for reproducibility (default: None)

    Returns:
        List of QuerySpec objects, all meeting the specified constraints
        Each QuerySpec has posterior_probability and prior_probability set.

    Raises:
        RuntimeError: If strict_mode=True and a query cannot be generated
    """
    # Setup logging
    logger = logging.getLogger(__name__)

    # Generate all combinations
    combinations = list(product(query_node_counts, evidence_counts))

    # Create RNG
    rng = np.random.default_rng(seed)

    results: list[QuerySpec] = []
    seen_queries: set[tuple[Any, ...]] = set()

    # Progress tracking - now includes distance buckets in the calculation
    total_expected = len(combinations) * len(distance_buckets) * queries_per_combination
    total_generated = 0

    if verbose:
        logger.info(
            f"Starting query generation: {len(combinations)} combinations × "
            f"{len(distance_buckets)} distance buckets × "
            f"{queries_per_combination} queries = {total_expected} total",
        )
        logger.info(f"Distance buckets: {distance_buckets}")
        logger.info(
            f"Parameters: min_abs_diff={min_abs_diff}, max_tries={max_tries_per_query}",
        )

    # Generate queries for each combination and distance bucket
    for combo_idx, (q_count, e_count) in enumerate(combinations):
        for bucket_idx, dist_bucket in enumerate(distance_buckets):
            combo_generated = 0

            if verbose:
                logger.info(
                    f"Combination {combo_idx + 1}/{len(combinations)}, "
                    f"Distance {bucket_idx + 1}/{len(distance_buckets)}: "
                    f"q={q_count}, e={e_count}, distance={dist_bucket}",
                )

            for instance_idx in range(queries_per_combination):
                # Generate single query with all constraints
                query = generate_single_query(
                    model=bn,
                    query_node_count=q_count,
                    evidence_count=e_count,
                    distance_bucket=dist_bucket,
                    max_tries=max_tries_per_query,
                    min_abs_diff=min_abs_diff,
                    seen_queries=seen_queries,
                    seed=int(rng.integers(0, 2**31)),
                )

                # Handle failure
                if query is None:
                    if verbose:
                        logger.warning(
                            f"Failed instance {instance_idx + 1}/"
                            f"{queries_per_combination}: "
                            f"No valid query after {max_tries_per_query} "
                            f"attempts",
                        )

                    if strict_mode:
                        msg = (
                            f"Failed to generate unique query meeting all constraints "
                            f"(distance={dist_bucket}, min_abs_diff={min_abs_diff}) "
                            f"for combination (q={q_count}, e={e_count}), "
                            f"instance {instance_idx} "
                            f"after {max_tries_per_query} attempts"
                        )
                        raise RuntimeError(msg)
                    else:
                        # Skip this query and continue
                        continue

                # Success - add to seen set and results
                query_key = _make_query_key(query.targets, query.evidence)
                seen_queries.add(query_key)
                results.append(query)
                combo_generated += 1
                total_generated += 1

            # Log combination results
            if verbose:
                logger.info(
                    f"Generated {combo_generated}/{queries_per_combination} queries",
                )

    # Final summary
    if verbose:
        success_rate = (
            (total_generated / total_expected) * 100 if total_expected > 0 else 0
        )
        logger.info(
            f"Final: {total_generated}/{total_expected} queries "
            f"({success_rate:.1f}% success rate)",
        )

    return results


__all__ = ["generate_queries"]
