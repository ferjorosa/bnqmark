"""
Bayesian Network generation with parameter sweeps.

This module provides high-level functions for generating multiple Bayesian Networks
with systematic parameter sweeps. The base function generates BN-DAG pairs with default
naming, which can then optionally have naming variants applied as a post-processing step
(potentially together with queries for clean ablation testing).
"""

from __future__ import annotations

from itertools import product
from typing import Any

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from src.bn.generation.core.generation import generate_variants_for_dag
from src.bn.generation.core.types import BaseBNMetadata
from src.dag import NamingStrategy, generate_single_dag


def _arity_to_str(spec: dict[str, Any]) -> str:
    """
    Convert arity specification to string representation.

    Args:
        spec: Arity specification dict with 'type' key and either 'fixed' or 'min'/'max'

    Returns:
        String representation like "fixed:2" or "range:2-4"
    """
    if spec["type"] == "fixed":
        return f"fixed:{spec['fixed']}"
    return f"range:{spec['min']}-{spec['max']}"


def _get_treewidths_for_n(
    n: int,
    treewidths: list[int] | dict[int, list[int]],
) -> list[int]:
    """
    Get treewidths for a specific number of variables.

    Args:
        n: Number of variables
        treewidths: Either a list of treewidths (same for all n) or a dict
            mapping n to treewidths

    Returns:
        List of treewidths for this n

    Raises:
        ValueError: If n is not found in treewidths dict when treewidths is a dict
    """
    if isinstance(treewidths, dict):
        n_treewidths = treewidths.get(n, [])
        if not n_treewidths:
            raise ValueError(
                f"No treewidths specified for n={n} in treewidths dictionary",
            )
        return n_treewidths
    return treewidths


def _generate_dag_for_params(
    n: int,
    tw: int,
    default_naming: NamingStrategy,
    base_seed: int,
    sample_counter: int,
) -> tuple[nx.DiGraph, int]:
    """
    Generate a DAG for given parameters.

    Args:
        n: Number of nodes
        tw: Target treewidth
        default_naming: Naming strategy for nodes
        base_seed: Base seed for reproducibility
        sample_counter: Sample counter for seed offset

    Returns:
        Tuple of (dag, achieved_treewidth)
    """
    dag, achieved_tw, _ = generate_single_dag(
        n,
        tw,
        node_naming=default_naming,
        seed=base_seed + sample_counter,
    )
    return dag, achieved_tw


def _generate_bn_variants_for_config(
    dag: nx.DiGraph,
    arity: dict[str, Any],
    alpha: float,
    det: float,
    variants_per_combo: int,
    base_seed: int,
    sample_counter: int,
) -> list[tuple[DiscreteBayesianNetwork, dict[str, Any]]]:
    """
    Generate BN variants for a given DAG and CPT configuration.

    Args:
        dag: DAG to generate BNs from
        arity: Arity specification
        alpha: Dirichlet alpha parameter
        det: Determinism fraction
        variants_per_combo: Number of variants to generate
        base_seed: Base seed for reproducibility
        sample_counter: Sample counter for seed offset

    Returns:
        List of (bn, metadata) tuples
    """
    cfgs = [
        {
            "arity_strategy": arity,
            "dirichlet_alpha": alpha,
            "determinism_fraction": det,
        }
        for _ in range(variants_per_combo)
    ]
    return generate_variants_for_dag(dag, cfgs, base_seed=base_seed + sample_counter)


def _create_base_metadata(
    n: int,
    tw: int,
    achieved_tw: int,
    arity: dict[str, Any],
    meta: dict[str, Any],
    variant_idx: int,
    bn: DiscreteBayesianNetwork,
    sample_counter: int,
    default_naming: str,
) -> BaseBNMetadata:
    """
    Create base metadata for a BN-DAG pair.

    Args:
        n: Number of variables
        tw: Target treewidth
        achieved_tw: Achieved treewidth
        arity: Arity specification
        meta: Metadata from BN generation (dict)
        variant_idx: Variant index
        bn: The Bayesian Network
        sample_counter: Base sample counter
        default_naming: Default naming strategy

    Returns:
        BaseBNMetadata instance
    """
    return BaseBNMetadata(
        n=n,
        target_tw=tw,
        achieved_tw=achieved_tw,
        naming=default_naming,  # Will be overridden later (store as string)
        arity=_arity_to_str(arity),
        alpha=meta["dirichlet_alpha"],
        determinism=meta["determinism_fraction"],
        seed=meta["seed"],
        variant_index=variant_idx,
        num_edges=bn.number_of_edges(),
        num_nodes=bn.number_of_nodes(),
        base_sample_counter=sample_counter,  # Track original BN
    )


def generate_bayesian_networks_and_metadata(
    ns: list[int],
    treewidths: list[int] | dict[int, list[int]],
    arity_specs: list[dict[str, Any]],
    dirichlet_alphas: list[float],
    determinism_fracs: list[float],
    variants_per_combo: int = 4,
    base_seed: int = 42,
    default_naming: NamingStrategy = NamingStrategy.SIMPLE,
) -> list[tuple[DiscreteBayesianNetwork, nx.DiGraph, dict[str, Any]]]:
    """
    Generate Bayesian networks with parameter sweeps (base version with default naming).

    This function sweeps over DAG/BN generation parameters and materializes multiple
    discrete BN variants per DAG. All networks are generated with the default naming
    strategy. Naming variants can be applied later as a post-processing step, optionally
    together with queries for clean ablation testing.

    This generates one unique BN structure per (n, tw, arity, alpha, det) combination.
    The DAG and BN are stored together to ensure alignment.

    Parameters:
        ns: List of numbers of variables
        treewidths: Either a list of target treewidths (same for all n) or a dict
          mapping each n to a list of treewidths (different treewidths per n)
        arity_specs: List of arity specifications (fixed or range)
        dirichlet_alphas: List of Dirichlet alpha values for CPT skewness
        determinism_fracs: List of determinism fractions (mostly 0%)
        variants_per_combo: Number of variants per parameter combination (default: 4)
        base_seed: Base seed for reproducibility (default: 42)
        default_naming: Default naming strategy used during BN generation
          (default: NamingStrategy.SIMPLE)

    Returns:
        List of (bn, dag, metadata) tuples with default naming.
        The 'dag' and 'bn' are aligned with the same node names.
    """
    base_bn_dag_pairs: list[
        tuple[DiscreteBayesianNetwork, nx.DiGraph, dict[str, Any]]
    ] = []
    sample_counter = 0

    for n in ns:
        n_treewidths = _get_treewidths_for_n(n, treewidths)

        for tw in n_treewidths:
            # Generate DAG with default naming (same structure regardless of naming)
            dag, achieved_tw = _generate_dag_for_params(
                n,
                tw,
                default_naming,
                base_seed,
                sample_counter,
            )

            # Generate BN variants for all parameter combinations
            for arity, alpha, det in product(
                arity_specs,
                dirichlet_alphas,
                determinism_fracs,
            ):
                variants = _generate_bn_variants_for_config(
                    dag,
                    arity,
                    alpha,
                    det,
                    variants_per_combo,
                    base_seed,
                    sample_counter,
                )

                # Store base BN-DAG pairs with their metadata
                for idx, (bn, meta) in enumerate(variants):
                    base_meta = _create_base_metadata(
                        n,
                        tw,
                        achieved_tw,
                        arity,
                        meta,
                        idx,
                        bn,
                        sample_counter,
                        default_naming,
                    )
                    # Convert to dict for compatibility with naming_variants functions
                    base_bn_dag_pairs.append((bn, dag.copy(), base_meta.to_dict()))

                sample_counter += 1

    return base_bn_dag_pairs
