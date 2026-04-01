#!/usr/bin/env python3
"""
Script to insert discrete experiments dataset into PostgreSQL database.

Loads data/experiments.parquet and inserts into the discrete_experiments table.

This is not mandatory to run as experiments use the data directly from the parquet
files, but it helps keeping the database consistent.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.discrete_experiments_db import (
    get_existing_experiment_identifiers,
    initialize_discrete_experiments_db,
    insert_experiment_batch,
)


def insert_experiments(data_dir: Path):
    """Insert experiments dataset."""
    print("=== Discrete Experiments Database Insertion ===")
    initialize_discrete_experiments_db()

    df = pd.read_parquet(data_dir / "experiments.parquet")
    print(f"Loaded {len(df)} experiment records")

    # Get existing experiment identifiers
    existing = get_existing_experiment_identifiers()
    print(f"Found {len(existing)} existing experiment records in database")

    # Filter out existing records
    df["key"] = df.apply(
        lambda row: (
            row["query_uuid"],
            row["naming_strategy"],
            row["run"],
            row["model_name"],
            row["experiment_type"],
        ),
        axis=1,
    )
    df_new = df[~df["key"].isin(existing)].copy()
    df_new = df_new.drop(columns=["key"])

    if len(df_new) == 0:
        print("✅ All experiment records already exist in database")
        return 0

    print(f"Filtered to {len(df_new)} new experiment records to insert")

    inserted = insert_experiment_batch(df_new.to_dict("records"), debug=True)
    print(f"✅ Inserted {inserted} experiment records")
    return inserted


def main():
    """Insert experiments dataset."""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    insert_experiments(data_dir)
    print("\n✅ Experiments dataset stored successfully in DB")


if __name__ == "__main__":
    main()
