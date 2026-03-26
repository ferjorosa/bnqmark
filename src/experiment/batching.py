"""
Batch creation utilities for parallel query processing.

This module provides functions to split query DataFrames into batches
for parallel processing.
"""

import pandas as pd


def create_query_batches(
    queries_df: pd.DataFrame, batch_size: int
) -> list[pd.DataFrame]:
    """
    Split a queries DataFrame into batches of a specified size.

    Args:
        queries_df: DataFrame containing queries to batch.
        batch_size: Number of queries per batch.

    Returns:
        List of DataFrames, each containing up to batch_size queries.
        Returns empty list if queries_df is empty.
    """
    if queries_df.empty:
        return []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # If batch_size is larger than the DataFrame, return single batch
    if batch_size >= len(queries_df):
        return [queries_df.copy()]

    batches = []
    total_rows = len(queries_df)

    # Split into batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = queries_df.iloc[start_idx:end_idx].copy()
        batches.append(batch)

    return batches
