#!/usr/bin/env python3
"""
Query Dataset Generation Script.

This script generates a dataset of probabilistic queries from Bayesian
networks using the refactored code from src/queries/ with a sampling-based
approach.

The script:
1. Loads BN dataset from parquet file (bns.parquet)
   - Assumes all BNs use SIMPLE naming strategy (V0, V1, V2, ...)
2. Generates queries for each BN using sampling approach:
   - Creates all combinations of (query_node_count, evidence_count)
   - For each combination, generates N instances by sampling
   - Samples up to max_tries times until min_abs_diff threshold is met
   - Keeps only queries that pass the threshold (no post-filtering)
   - Computes exact probabilities (posterior and prior) during generation
3. Computes query structural properties and complexity metrics
4. Exports results to parquet file and generates analysis plots

Key Features:
- Exhaustive combination coverage: ensures balanced representation
- Threshold filtering during generation: more efficient than post-filtering
- Single broad distance bucket: analyze distances post-hoc via
  evidence_distances field
- Configurable: queries_per_combination, min_abs_diff, max_tries_per_query,
  strict_mode

Output files:
- queries.parquet: Query dataset where all queries meet
  |posterior - prior| > min_abs_diff. Each query has a unique query_uuid and
  is linked to its BN via bn_uuid and naming_strategy, includes
  evidence_distances

Note: Naming variants can be applied later as a separate step. This script
generates queries for the base BNs with SIMPLE naming only.
Run plot_query_dataset.py to generate visualization plots.
"""

import json
import pickle
import uuid
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.queries import (
    QuerySpec,
    compute_query_structural_properties,
    generate_queries,
)
from src.queries.complexity import compute_query_complexity


def _load_bn_dataset(
    parquet_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Load BN dataset from parquet file.

    Unpickles BN and DAG objects from the pickle columns and returns
    them in the same format as before for compatibility.

    Args:
        parquet_path: Path to the parquet file containing BN dataset

    Returns:
        Tuple of (all_bayesian_networks, df) where:
        - all_bayesian_networks: List of dicts with 'bn', 'dag', 'meta' keys
        - df: DataFrame with all metadata columns
    """
    print(f"Loading BN dataset from {parquet_path}...")

    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    # Unpickle BN and DAG objects
    all_bayesian_networks = []
    for _, row in df.iterrows():
        bn = pickle.loads(row["bn_pickle"])
        dag = pickle.loads(row["dag_pickle"])

        # Extract metadata from row (exclude pickle columns)
        meta = {k: v for k, v in row.items() if not k.endswith("_pickle")}

        all_bayesian_networks.append(
            {
                "bn": bn,
                "dag": dag,
                "meta": meta,
            },
        )

    print(f"✓ Loaded {len(all_bayesian_networks)} Bayesian networks")
    print(f"✓ Loaded metadata for {len(df)} networks")

    return all_bayesian_networks, df


def _process_single_query(
    bn: Any,
    query: QuerySpec,
    bn_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Process a single query and compute all metadata.

    Each query gets a unique query_uuid for identification and tracking.
    The result includes bn_uuid and naming_strategy as links to the BN dataset,
    keeping the query dataset focused on query-specific properties. All BN metadata
    can be accessed via the bn_uuid in the original BN dataset.

    Structural properties (including evidence_distances) are computed via
    compute_query_structural_properties().
    """
    # Extract target variables (targets is already a dictionary)
    target_nodes = list(query.targets.keys())
    evidence_nodes = list(query.evidence.keys()) if query.evidence else []

    # Use probabilities computed during sweep generation
    posterior_prob = query.posterior_probability
    prior_prob = query.prior_probability

    # Fallback to recomputation if not available
    # if posterior_prob is None or prior_prob is None:
    #     posterior_prob, prior_prob = compute_query_probabilities(bn, query)

    # Compute structural properties (includes evidence_distances)
    query_metadata = compute_query_structural_properties(
        bn,
        target_nodes,
        evidence_nodes,
    )

    # Compute complexity metrics (may fail on large/complex networks)
    complexity = compute_query_complexity(
        bn,
        target_nodes,
        evidence_nodes,
        verbose=False,
    )
    induced_width = complexity.induced_width
    num_eliminated = complexity.num_eliminated_vars

    # Build result row - only keep bn_uuid as link to BN dataset
    # Serialize dictionaries as JSON strings to avoid parquet schema issues
    row = {
        "bn_uuid": bn_row.get("bn_uuid"),
        "query_uuid": str(uuid.uuid4()),
        "naming_strategy": bn_row.get("naming_strategy"),
        "target": json.dumps(query.targets),
        "evidence": json.dumps(query.evidence),
        "num_evidence": query.meta.get("num_evidence_nodes"),
        "num_target": query.meta.get("num_query_nodes"),
        "probability": posterior_prob,
        "prior_probability": prior_prob,
        "induced_width": induced_width,
        "num_eliminated": num_eliminated,
        "distance_bucket": str(query.meta.distance_bucket),  # Store as string tuple
    }
    row.update(query_metadata)

    return row


def _generate_all_queries_with_sweep(
    all_bayesian_networks: list[dict[str, Any]],
    df: pd.DataFrame,
    queries_per_combination: int,
    query_node_counts: tuple,
    evidence_counts: tuple,
    distance_buckets: list[tuple],
    min_abs_diff: float,
    max_tries_per_query: int,
    strict_mode: bool,
    base_seed: int,
) -> pd.DataFrame:
    """
    Generate queries for all BNs using the sweep module.

    Wrapper around generate_queries_with_sampling that processes multiple BNs
    and converts the results to a DataFrame with all metadata.

    Args:
        all_bayesian_networks: List of BN dictionaries
        df: Metadata DataFrame for BNs
        queries_per_combination: Number of query instances per combination
        query_node_counts: Tuple of query node counts
        evidence_counts: Tuple of evidence node counts
        distance_buckets: List of distance bucket ranges
        min_abs_diff: Minimum absolute difference threshold
        max_tries_per_query: Maximum sampling attempts per query
        strict_mode: If True, raise exception on failure; if False, continue
        base_seed: Base random seed

    Returns:
        DataFrame with generated queries (all meeting min_abs_diff threshold)
    """
    combinations = list(product(query_node_counts, evidence_counts))
    max_possible = (
        len(all_bayesian_networks)
        * len(combinations)
        * len(distance_buckets)
        * queries_per_combination
    )

    print("Step 2: Generating queries with sampling approach...")
    print(f"  • Queries per combination: {queries_per_combination}")
    print(f"  • Query node counts: {query_node_counts}")
    print(f"  • Evidence counts: {evidence_counts}")
    print(f"  • Distance buckets: {distance_buckets}")
    print(f"  • Combinations: {len(combinations)} ({combinations})")
    print(
        f"  • Total queries per BN: "
        f"{len(combinations) * len(distance_buckets) * queries_per_combination}",
    )
    print(f"  • Maximum possible queries: {max_possible}")
    print(f"  • Min absolute difference: {min_abs_diff}")
    print(f"  • Max tries per query: {max_tries_per_query}")
    print(f"  • Strict mode: {strict_mode}")
    print(f"  • Base seed: {base_seed}")
    print()

    query_rows = []
    total_generated = 0
    total_failed = 0

    for bn_idx, bn_dict in enumerate(all_bayesian_networks):
        bn = bn_dict["bn"]
        bn_row = df.iloc[bn_idx].to_dict()
        bn_uuid = bn_row.get("bn_uuid", f"BN_{bn_idx}")

        print(
            f"Processing BN {bn_idx + 1}/{len(all_bayesian_networks)} "
            f"(uuid: {bn_uuid})...",
        )

        # Generate queries using sweep module
        try:
            queries = generate_queries(
                bn=bn,
                queries_per_combination=queries_per_combination,
                query_node_counts=query_node_counts,
                evidence_counts=evidence_counts,
                distance_buckets=distance_buckets,
                min_abs_diff=min_abs_diff,
                max_tries_per_query=max_tries_per_query,
                strict_mode=strict_mode,
                verbose=True,  # Enable detailed logging from sweep module
                seed=base_seed + bn_idx,
            )

            # Process each query
            for _, query in enumerate(queries):
                row = _process_single_query(bn, query, bn_row)
                query_rows.append(row)
                total_generated += 1

            expected = len(combinations) * queries_per_combination
            actual = len(queries)
            failed_this_bn = expected - actual
            total_failed += failed_this_bn

        except Exception as e:
            print(f"  Error: Failed to generate queries for BN {bn_idx}: {e}")
            expected = len(combinations) * queries_per_combination
            total_failed += expected

    print()
    print(f"✓ Generated {total_generated} queries total (all meeting threshold)")
    print(f"  • Maximum possible: {max_possible}")
    print(f"  • Success rate: {100 * total_generated / max_possible:.1f}%")
    if total_failed > 0:
        print(f"⚠ Failed to generate {total_failed} queries")
    print()

    return pd.DataFrame(query_rows)


def _export_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Export results to parquet file."""
    print("Step 4: Exporting results...")

    parquet_path = output_dir / "queries.parquet"
    df.to_parquet(parquet_path, index=False, compression="snappy")

    # Calculate file size for reporting
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)

    print(f"✓ Saved filtered query dataset to: {parquet_path}")
    print(f"  • File size: {file_size_mb:.2f} MB")
    print(f"  • Rows: {len(df)}")
    print(f"  • Columns: {len(df.columns)}")
    print()


def _compute_statistics_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probability differences without filtering.

    Args:
        df: DataFrame with query data

    Returns:
        DataFrame with added abs_diff and rel_diff columns
    """
    print("Step 3: Computing statistics...")

    # Compute differences
    prior = df["prior_probability"].astype(float)
    posterior = df["probability"].astype(float)
    abs_diff = np.abs(posterior - prior)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(prior != 0, (posterior - prior) / prior, np.nan)

    df = df.copy()
    df["abs_diff"] = abs_diff
    df["rel_diff"] = rel_diff

    # Compute average distances from evidence_distances list
    avg_distances = []
    if "evidence_distances" in df.columns:
        for distances in df["evidence_distances"]:
            if (
                distances is not None
                and isinstance(distances, list)
                and len(distances) > 0
            ):
                avg_distances.append(np.mean(distances))
            else:
                avg_distances.append(np.nan)
        df["avg_distance_target_evidence"] = avg_distances

    # Show statistics
    valid_mask = ~(prior.isna() | posterior.isna())
    print(f"✓ Computed differences for {valid_mask.sum()} valid queries")
    print(f"  • Mean absolute difference: {abs_diff[valid_mask].mean():.4f}")
    print(f"  • Min absolute difference: {abs_diff[valid_mask].min():.4f}")
    print(f"  • Max absolute difference: {abs_diff[valid_mask].max():.4f}")
    print()

    return df


def main():
    """Main function to generate query dataset."""
    print("=== Query Dataset Generation ===")
    print("Using refactored code from src/queries/")
    print("Using sampling-based approach with threshold filtering during generation")
    print()

    # Configuration
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    bn_parquet_path = data_dir / "bns.parquet"

    if not bn_parquet_path.exists():
        raise FileNotFoundError(
            f"BN dataset not found: {bn_parquet_path}\n"
            "Please run generate_bn_dataset.py first to create the dataset.",
        )

    # Query generation parameters (sampling-based approach)
    queries_per_combination = 1  # Number of instances per combination
    query_node_counts = (1, 2)
    evidence_counts = (1, 2)
    distance_buckets = [(1, 100), (2, 100), (3, 100)]
    # distance_buckets = [(1, 100)]
    min_abs_diff = 0.1  # |posterior - prior| >= min_abs_diff
    max_tries_per_query = 50  # Max sampling attempts per query
    strict_mode = False  # Continue on failure (don't raise exception)
    base_seed = 1000

    # Step 1: Load BN dataset
    print("Step 1: Loading BN dataset...")
    all_bayesian_networks, df = _load_bn_dataset(bn_parquet_path)
    print()

    # Step 2: Generate queries with sampling approach
    queries_df = _generate_all_queries_with_sweep(
        all_bayesian_networks,
        df,
        queries_per_combination,
        query_node_counts,
        evidence_counts,
        distance_buckets,
        min_abs_diff,
        max_tries_per_query,
        strict_mode,
        base_seed,
    )

    # Step 3: Compute statistics (no filtering needed)
    queries_df = _compute_statistics_only(queries_df)

    # Step 4: Export results
    _export_results(queries_df, data_dir)

    # Calculate max possible for final summary
    combinations = list(product(query_node_counts, evidence_counts))
    max_possible = (
        len(all_bayesian_networks)
        * len(combinations)
        * len(distance_buckets)
        * queries_per_combination
    )

    print("=== Generation Complete ===")
    print(f"✅ Successfully processed {len(all_bayesian_networks)} Bayesian networks")
    print(
        f"✅ Generated {len(queries_df)} queries (all meeting |diff| > {min_abs_diff})",
    )
    print(f"  • Maximum possible queries: {max_possible}")
    print(f"  • Success rate: {100 * len(queries_df) / max_possible:.1f}%")
    print(
        f"✅ Query combinations: "
        f"{len(list(product(query_node_counts, evidence_counts)))}",
    )
    print(f"✅ Queries per combination: {queries_per_combination}")
    print(f"✅ Exported to {data_dir / 'queries.parquet'}")
    print()
    print("Run plot_query_dataset.py to generate visualization plots.")
    print("Ready for LLM probabilistic reasoning evaluation! 🚀")


if __name__ == "__main__":
    main()
