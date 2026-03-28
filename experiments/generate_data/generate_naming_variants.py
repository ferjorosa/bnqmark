#!/usr/bin/env python3
"""
Combined Naming Variant Generation Script.

This script generates naming variants for both Bayesian Networks and queries,
creating variants with different naming strategies while preserving structure
and probabilities for ablation testing.

Steps:
1. Generate BN variants
2. Generate query variants
3. Save updated datasets with variants appended
"""

import json
import pickle
from pathlib import Path

import pandas as pd

from src.dag import NamingStrategy
from src.naming_variants import (
    create_bn_naming_variant,
    create_dag_naming_variant,
    create_name_mapping_from_strategy,
)


def _load_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load existing dataset."""
    print(f"Loading dataset from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.\n")
    return df


def _create_bn_naming_variants(
    df: pd.DataFrame,
    naming_strategies: list[NamingStrategy],
    base_seed: int = 42,
) -> pd.DataFrame:
    """Generate naming variants for BN dataset."""
    print(f"Creating BN naming variants for {len(df)} rows ...")
    print(f"Strategies: {[s.value for s in naming_strategies]}")
    variant_rows = []

    for idx, row in df.iterrows():
        bn = pickle.loads(row["bn_pickle"])
        dag = pickle.loads(row["dag_pickle"])
        original_naming = row.get("naming_strategy")

        for strategy in naming_strategies:
            if original_naming == strategy.value:
                continue

            strategy_hash = abs(hash(strategy.value)) % (2**32)
            variant_seed = (base_seed + idx * 1000 + strategy_hash) % (2**32)

            bn_variant = create_bn_naming_variant(bn, strategy, seed=variant_seed)
            dag_variant = create_dag_naming_variant(dag, strategy, seed=variant_seed)

            # Copy all data from original row (structure/metrics remain identical)
            # Only update: naming_strategy and serialized objects
            new_row = row.copy()
            new_row.update(
                {
                    "naming_strategy": strategy.value,
                    "bn_pickle": pickle.dumps(bn_variant),
                    "dag_pickle": pickle.dumps(dag_variant),
                },
            )
            variant_rows.append(new_row)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows")

    print(f"Created {len(variant_rows)} BN variant rows.\n")
    return pd.DataFrame(variant_rows) if variant_rows else pd.DataFrame()


def _create_query_naming_variants(
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
    naming_strategies: list[NamingStrategy],
    base_seed: int = 42,
) -> pd.DataFrame:
    """Generate naming variants for queries."""
    print(f"Creating query naming variants for {len(queries_df)} queries...")
    print(f"Strategies: {[s.value for s in naming_strategies]}\n")

    # Build BN lookup: uuid -> (bn, row_idx)
    bn_lookup = {
        row["bn_uuid"]: (pickle.loads(row["bn_pickle"]), idx)
        for idx, row in bns_df.iterrows()
        if row.get("bn_uuid")
    }

    variant_rows = []
    for idx, qrow in queries_df.iterrows():
        bn_uuid = qrow.get("bn_uuid")
        if bn_uuid not in bn_lookup:
            continue

        bn, bn_idx = bn_lookup[bn_uuid]
        nodes = list(bn.nodes())
        original_naming = qrow["naming_strategy"]

        for strategy in naming_strategies:
            if original_naming == strategy.value:
                continue

            # Same seed calculation as BN variants
            seed = (base_seed + bn_idx * 1000 + abs(hash(strategy.value)) % (2**32)) % (
                2**32
            )
            mapping = create_name_mapping_from_strategy(nodes, strategy, seed=seed)

            # Get query columns (targets always exist, evidence may be empty)
            target_dict = qrow["target"]
            target_nodes = list(qrow["target_nodes"])
            evidence_dict = qrow.get("evidence", {})
            evidence_nodes = list(qrow.get("evidence_nodes", []))

            # Apply mapping
            new_row = qrow.copy()
            new_row.update(
                {
                    "naming_strategy": strategy.value,
                    "target": {mapping.get(k, k): v for k, v in target_dict.items()},
                    "evidence": {
                        mapping.get(k, k): v for k, v in evidence_dict.items()
                    },
                    "target_nodes": [mapping.get(x, x) for x in target_nodes],
                    "evidence_nodes": [mapping.get(x, x) for x in evidence_nodes],
                },
            )
            variant_rows.append(new_row)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(queries_df)} queries")

    print(f"Created {len(variant_rows)} query variant rows.\n")
    return pd.DataFrame(variant_rows) if variant_rows else pd.DataFrame()


def _combine_and_save_bn_dataset(
    original_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Combine and save BN datasets."""
    if variants_df.empty:
        print("No BN variants created. Keeping original dataset.")
        combined_df = original_df
    else:
        combined_df = pd.concat([original_df, variants_df], ignore_index=True)
        print(
            f"Combined BN dataset: {len(original_df)} original + "
            f"{len(variants_df)} variants = {len(combined_df)} total rows.",
        )

    combined_df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved BN dataset to {output_path} ({size_mb:.2f} MB)\n")

    if "naming_strategy" in combined_df.columns:
        print("BN naming strategy distribution:")
        for strategy, count in (
            combined_df["naming_strategy"].value_counts().sort_index().items()
        ):
            print(f"  {strategy}: {count}")
        print()


def _combine_and_save_queries_dataset(
    original_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Combine and save query datasets."""
    if variants_df.empty:
        print("No query variants created. Keeping original dataset.")
        combined_df = original_df
    else:
        combined_df = pd.concat([original_df, variants_df], ignore_index=True)
        print(
            f"Combined query dataset: {len(original_df)} original + "
            f"{len(variants_df)} variants = {len(combined_df)} total rows.",
        )

    # Serialize all dictionaries to JSON strings before saving
    combined_df["target"] = combined_df["target"].apply(json.dumps)
    combined_df["evidence"] = combined_df["evidence"].apply(
        lambda x: json.dumps(x) if x else json.dumps({}),
    )

    combined_df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved query dataset to {output_path} ({size_mb:.2f} MB)\n")

    if "naming_strategy" in combined_df.columns:
        print("Query naming strategy distribution:")
        for strategy, count in (
            combined_df["naming_strategy"].value_counts().sort_index().items()
        ):
            print(f"  {strategy}: {count}")
        print()


def main():
    """Main entry point."""
    print("=== Combined Naming Variant Generation ===\n")

    naming_strategies = [
        NamingStrategy.CONFUSING,
        # NamingStrategy.SEMANTIC,
        # NamingStrategy.MIXED,
    ]

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    bns_path = data_dir / "bns.parquet"
    queries_path = data_dir / "queries.parquet"
    bns_df = _load_dataset(bns_path)
    queries_df = _load_dataset(queries_path)
    base_seed = 42

    # Generate BN variants
    print("--- Processing BN Dataset ---")
    bn_variants_df = _create_bn_naming_variants(bns_df, naming_strategies, base_seed)
    _combine_and_save_bn_dataset(bns_df, bn_variants_df, bns_path)
    print(f"BN variants: Added {len(bn_variants_df)} new variants.\n")

    # Generate query variants
    print("--- Processing Query Dataset ---")
    # Deserialize JSON strings back to dictionaries for target and evidence
    queries_df["target"] = queries_df["target"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x,
    )
    queries_df["evidence"] = queries_df["evidence"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if x else {}),
    )

    query_variants_df = _create_query_naming_variants(
        queries_df,
        bns_df,
        naming_strategies,
        base_seed,
    )
    _combine_and_save_queries_dataset(queries_df, query_variants_df, queries_path)
    print(f"Query variants: Added {len(query_variants_df)} new variants.\n")

    print("=== Generation Complete ===")


if __name__ == "__main__":
    main()
