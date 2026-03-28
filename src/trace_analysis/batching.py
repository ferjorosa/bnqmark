"""
Batch creation utilities for parallel trace analysis.

This module provides functions to split experiment DataFrames into batches
for parallel trace analysis.
"""

import pandas as pd


def create_experiment_batches(
    experiments_df: pd.DataFrame, batch_size: int
) -> list[pd.DataFrame]:
    """
    Split an experiments DataFrame into batches of a specified size.

    Args:
        experiments_df: DataFrame containing experiments to batch.
        batch_size: Number of experiments per batch.

    Returns:
        List of DataFrames, each containing up to batch_size experiments.
        Returns empty list if experiments_df is empty.
    """
    if experiments_df.empty:
        return []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # If batch_size is larger than the DataFrame, return single batch
    if batch_size >= len(experiments_df):
        return [experiments_df.copy()]

    batches = []
    total_rows = len(experiments_df)

    # Split into batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = experiments_df.iloc[start_idx:end_idx].copy()
        batches.append(batch)

    return batches
