#!/usr/bin/env python3
"""
Bayesian Network Dataset Generation Script.

This script generates a dataset of Bayesian networks using the refactored code
from src/bn/generation/.

The script:
1. Generates BNs with systematic parameter sweeps across network sizes,
   treewidths, arity specs, and Dirichlet alphas
2. All networks use SIMPLE naming strategy (V0, V1, V2, ...)
3. Assigns a unique UUID to each Bayesian network for easy linking
4. Computes additional metadata (number of nodes/edges, Markov blanket sizes, etc.)
5. Exports everything to a single parquet file

Output file:
- bns.parquet: Single file containing:
  * All metadata columns (n, treewidth, naming_strategy, etc.) as regular columns
  * bn_uuid: Unique identifier for each BN (for linking with queries)
  * bn_pickle: Serialized pgmpy DiscreteBayesianNetwork object
  * dag_pickle: Serialized networkx DiGraph object

Note: Naming variants can be applied later as a separate step. The UUID enables
easy linking between Bayesian networks and their queries in downstream analysis
and database storage.
"""

import pickle
import uuid
from pathlib import Path

import pandas as pd

from src.bn import generate_bayesian_networks_and_metadata
from src.bn.analysis import compute_average_markov_blanket_size, num_edges
from src.dag.generation.core.types import NamingStrategy


def _generate_bayesian_networks(
    ns: list[int],
    treewidths: dict[int, list[int]],
    arity_specs: list[dict],
    dirichlet_alphas: list[float],
    determinism_fracs: list[float],
    variants_per_combo: int,
    base_seed: int,
) -> list[tuple]:
    """
    Generate Bayesian networks with systematic parameter sweeps.

    All networks are generated with SIMPLE naming strategy (V0, V1, V2, ...).
    Naming variants can be applied later as a separate step.

    Args:
        ns: List of network sizes
        treewidths: Dictionary mapping network sizes to treewidth values
        arity_specs: List of arity specifications
        dirichlet_alphas: List of Dirichlet alpha values
        determinism_fracs: List of determinism fractions
        variants_per_combo: Number of variants per parameter combination
        base_seed: Base random seed

    Returns:
        List of (bn, dag, metadata) tuples with SIMPLE naming
    """
    print("Step 1: Defining generation parameters...")
    print(f"  • Network sizes (n): {ns}")
    print(f"  • Treewidths: {treewidths}")
    print(f"  • Arity specs: {arity_specs}")
    print(f"  • Dirichlet alphas: {dirichlet_alphas}")
    print(f"  • Determinism fractions: {determinism_fracs}")
    print("  • Naming strategy: SIMPLE (V0, V1, V2, ...)")
    print(f"  • Variants per combo: {variants_per_combo}")
    print(f"  • Base seed: {base_seed}")
    print()

    print("Step 2: Generating Bayesian networks...")
    bn_dag_pairs = generate_bayesian_networks_and_metadata(
        ns=ns,
        treewidths=treewidths,
        arity_specs=arity_specs,
        dirichlet_alphas=dirichlet_alphas,
        determinism_fracs=determinism_fracs,
        variants_per_combo=variants_per_combo,
        base_seed=base_seed,
        default_naming=NamingStrategy.SIMPLE,
    )

    print(f"✓ Generated {len(bn_dag_pairs)} Bayesian networks")
    print()

    return bn_dag_pairs


def _compute_additional_metadata(
    all_bn_dag_pairs: list[tuple],
) -> pd.DataFrame:
    """
    Compute additional metadata and create a complete DataFrame with serialized objects.

    This function:
    1. Generates unique UUIDs for each Bayesian network
    2. Computes structural metadata (nodes, edges, Markov blanket sizes)
    3. Expands all metadata fields as regular columns
    4. Serializes BN and DAG objects as pickle bytes
    5. Creates a single DataFrame with all information

    Args:
        all_bn_dag_pairs: List of (bn, dag, metadata) tuples with SIMPLE naming

    Returns:
        pd.DataFrame: Complete dataframe with metadata and serialized objects
    """
    print("Step 3: Creating comprehensive dataset DataFrame...")

    # Generate UUIDs, compute additional metadata, and expand metadata fields
    print("  • Generating UUIDs, computing metadata, and expanding metadata fields...")
    bn_uuids = []
    num_nodes_list = []
    num_edges_list = []
    avg_mb_size_list = []
    bn_pickles = []
    dag_pickles = []

    # Collect all metadata fields to expand as columns
    expanded_metadata = []

    for bn, dag, meta in all_bn_dag_pairs:
        # Generate UUID for this BN
        bn_uuid = str(uuid.uuid4())
        bn_uuids.append(bn_uuid)

        # Compute structural metrics
        n_nodes = len(bn.nodes())
        n_edges = num_edges(bn)
        avg_mb = compute_average_markov_blanket_size(bn)

        num_nodes_list.append(n_nodes)
        num_edges_list.append(n_edges)
        avg_mb_size_list.append(avg_mb)

        # Serialize objects to pickle bytes
        bn_pickles.append(pickle.dumps(bn))
        dag_pickles.append(pickle.dumps(dag))

        # Expand metadata dict as flat structure (add UUID and computed metrics)
        expanded_meta = meta.copy()
        expanded_meta["bn_uuid"] = bn_uuid
        expanded_meta["num_nodes"] = n_nodes
        expanded_meta["num_edges"] = n_edges
        expanded_meta["edge_density"] = n_edges / n_nodes if n_nodes > 0 else 0
        expanded_meta["avg_markov_blanket_size"] = avg_mb

        expanded_meta["naming_strategy"] = meta.get("naming")

        expanded_metadata.append(expanded_meta)

    # Create DataFrame from expanded metadata
    df = pd.DataFrame(expanded_metadata)

    # Reorder to put UUID first
    if "bn_uuid" in df.columns:
        cols = ["bn_uuid"] + [col for col in df.columns if col != "bn_uuid"]
        df = df[cols]

    # Add serialized object columns
    df["bn_pickle"] = bn_pickles
    df["dag_pickle"] = dag_pickles

    print(f"✓ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    print(f"  • Columns: {list(df.columns)}")
    print()

    # Display summary statistics
    print("Step 4: Dataset summary...")
    print(f"  • Total networks: {len(all_bn_dag_pairs)}")
    print(f"  • Total metadata rows: {len(df)}")
    print(f"  • Network size range: {min(num_nodes_list)}-{max(num_nodes_list)} nodes")
    print(f"  • Edge count range: {min(num_edges_list)}-{max(num_edges_list)} edges")
    min_mb = min(avg_mb_size_list)
    max_mb = max(avg_mb_size_list)
    print(f"  • Avg MB size range: {min_mb:.2f}-{max_mb:.2f}")
    print()

    # Show first few rows (excluding pickle columns for readability)
    print("First 5 rows of metadata (pickle columns excluded from display):")
    display_df = df.drop(columns=["bn_pickle", "dag_pickle"])
    print(display_df.head().to_string())
    print()

    return df


def _export_results(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export complete dataset to a single parquet file.

    The parquet file contains:
    - All metadata columns (n, treewidth, naming_strategy, etc.) as regular columns
    - bn_uuid: Unique identifier for linking
    - Serialized objects: bn_pickle, dag_pickle

    Args:
        df: DataFrame with all metadata and serialized objects
        output_dir: Directory to save output files
    """
    print("Step 5: Exporting results to parquet...")

    # Save to parquet (with compression for efficiency)
    parquet_path = output_dir / "bns.parquet"
    df.to_parquet(parquet_path, index=False, compression="snappy")

    # Calculate file size for reporting
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)

    print(f"✓ Saved complete dataset to: {parquet_path}")
    print(f"  • File size: {file_size_mb:.2f} MB")
    print(f"  • Rows: {len(df)}")
    print(f"  • Columns: {len(df.columns)}")


def main():
    """Main function to generate BN dataset."""
    print("=== Bayesian Network Dataset Generation ===")
    print("Using refactored code from src/bn/generation/")
    print()

    # Parameter grids
    ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    treewidths = {
        4: [2],
        6: [2, 4],
        8: [2, 4, 6],
        10: [2, 4, 6, 8],
        12: [2, 4, 6, 8, 10],
        14: [2, 4, 6, 8, 10, 12],
        16: [2, 4, 6, 8, 10, 12],
        18: [2, 4, 6, 8, 10, 12],
        20: [2, 4, 6, 8, 10, 12],
    }
    arity_specs = [
        {"type": "fixed", "fixed": 2},
        # {"type": "range", "min": 2, "max": 3},
    ]
    dirichlet_alphas = [1.0, 0.5]
    determinism_fracs = [0.0]
    variants_per_combo = 1
    base_seed = 42

    # Step 1-2: Generate Bayesian networks with SIMPLE naming
    all_bn_dag_pairs = _generate_bayesian_networks(
        ns=ns,
        treewidths=treewidths,
        arity_specs=arity_specs,
        dirichlet_alphas=dirichlet_alphas,
        determinism_fracs=determinism_fracs,
        variants_per_combo=variants_per_combo,
        base_seed=base_seed,
    )

    # Step 3-4: Compute additional metadata and create comprehensive dataframe
    # This adds UUIDs, computes metrics, and serializes objects
    df = _compute_additional_metadata(all_bn_dag_pairs)

    # Step 5: Export complete dataset to parquet file
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    _export_results(df, data_dir)

    print()
    print("=== Generation Complete ===")
    print(f"✅ Successfully generated {len(all_bn_dag_pairs)} Bayesian networks")
    print(f"✅ Exported to {data_dir / 'bns.parquet'}")
    print()
    print("Dataset structure:")
    print(
        "   • Metadata columns: n, treewidth, naming_strategy, alpha, etc.",
    )
    print("   • bn_uuid: Unique identifier for each BN")
    print("   • bn_pickle: Serialized pgmpy BayesianNetwork")
    print("   • dag_pickle: Serialized networkx DiGraph")


if __name__ == "__main__":
    main()
