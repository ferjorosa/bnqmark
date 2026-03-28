#!/usr/bin/env python3
"""
Script to insert discrete BN and queries datasets into PostgreSQL database.

Loads data/bns.parquet and data/queries.parquet and inserts into respective tables.

This is not mandatory to run as experiments use the data directly from the parquet
files, but it helps keeping the database consistent.
"""

from pathlib import Path

import pandas as pd

from src.database.discrete_bn_db import (
    get_existing_bn_identifiers,
    initialize_discrete_bns_db,
    insert_bn_batch,
)
from src.database.discrete_queries_db import (
    get_existing_query_identifiers,
    initialize_discrete_queries_db,
    insert_query_batch,
)


def insert_bns(data_dir: Path):
    """Insert BN dataset."""
    print("=== Discrete BN Database Insertion ===")
    initialize_discrete_bns_db()

    df = pd.read_parquet(data_dir / "bns.parquet")
    print(f"Loaded {len(df)} BN records")

    # Get existing BN identifiers
    existing = get_existing_bn_identifiers()
    print(f"Found {len(existing)} existing BN records in database")

    # Filter out existing records
    df["key"] = df.apply(lambda row: (row["bn_uuid"], row["naming_strategy"]), axis=1)
    df_new = df[~df["key"].isin(existing)].copy()
    df_new = df_new.drop(columns=["key"])

    if len(df_new) == 0:
        print("✅ All BN records already exist in database")
        return 0

    print(f"Filtered to {len(df_new)} new BN records to insert")

    inserted = insert_bn_batch(df_new.to_dict("records"), debug=True)
    print(f"✅ Inserted {inserted} BN records")
    return inserted


def insert_queries(data_dir: Path):
    """Insert queries dataset."""
    print("\n=== Discrete Queries Database Insertion ===")
    initialize_discrete_queries_db()

    df = pd.read_parquet(data_dir / "queries.parquet")
    print(f"Loaded {len(df)} query records")

    # Get existing query identifiers
    existing = get_existing_query_identifiers()
    print(f"Found {len(existing)} existing query records in database")

    # Filter out existing records
    df["key"] = df.apply(
        lambda row: (row["query_uuid"], row["naming_strategy"]), axis=1
    )
    df_new = df[~df["key"].isin(existing)].copy()
    df_new = df_new.drop(columns=["key"])

    if len(df_new) == 0:
        print("✅ All query records already exist in database")
        return 0

    print(f"Filtered to {len(df_new)} new query records to insert")

    inserted = insert_query_batch(df_new.to_dict("records"), debug=True)
    print(f"✅ Inserted {inserted} query records")
    return inserted


def main():
    """Insert both datasets."""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    insert_bns(data_dir)
    insert_queries(data_dir)
    print("\n✅ All datasets stored successfully in DB")


if __name__ == "__main__":
    main()
