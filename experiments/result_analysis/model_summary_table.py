#!/usr/bin/env python3
"""Generate a summary table with MAE and std error per model."""

import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

# Configuration - Edit these to filter experiments
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive"
EXPERIMENT_TYPE = "raw_reasoning"  # Options: "raw_reasoning", "code_generation"


def load_experiments(parquet_path: Path | None = None) -> pd.DataFrame:
    """Load experiments from parquet file."""
    if parquet_path is None:
        parquet_path = repo_root / "data" / "experiments.parquet"
    return pd.read_parquet(parquet_path)


def load_queries(parquet_path: Path | None = None) -> pd.DataFrame:
    """Load queries with true probabilities from parquet file."""
    if parquet_path is None:
        parquet_path = repo_root / "data" / "queries.parquet"
    df = pd.read_parquet(parquet_path)
    return df[["query_uuid", "naming_strategy", "probability"]].copy()


def generate_model_summary(
    naming_strategy: str = NAMING_STRATEGY,
    experiment_type: str = EXPERIMENT_TYPE,
    accuracy_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a summary table with one row per model.

    Columns:
    - model_name: Name of the model
    - supported_queries: Number of queries without llm_probability = -1000
    - mae: Mean Absolute Error (valid queries only)
    - std_error: Standard deviation of absolute errors (valid queries only)
    - accuracy: Percentage of queries (excluding -1000) with abs_error <= threshold
                   (null counts as failure, -1000 is excluded from denominator)

    MAE and std_error are computed only on valid queries (not None and not -1000).
    Accuracy excludes -1000 (context limit exceeded) but counts null as failure.
    """
    experiments_df = load_experiments()
    queries_df = load_queries()

    # Filter by naming_strategy and experiment_type
    experiments_df = experiments_df[
        (experiments_df["naming_strategy"] == naming_strategy)
        & (experiments_df["experiment_type"] == experiment_type)
    ]
    queries_df = queries_df[queries_df["naming_strategy"] == naming_strategy]

    # Get all unique models
    all_models = experiments_df["model_name"].unique().tolist()

    # Count supported queries (llm_probability != -1000)
    supported_counts = (
        experiments_df[experiments_df["llm_probability"] != -1000]
        .groupby("model_name")
        .size()
        .to_dict()
    )

    # Filter to valid probabilities for MAE calculation
    valid_df = experiments_df[
        (experiments_df["llm_probability"].notna())
        & (experiments_df["llm_probability"] != -1000)
    ].copy()

    # Merge with queries to get true probabilities
    merged_df = valid_df.merge(
        queries_df[["query_uuid", "probability"]],
        on="query_uuid",
        how="inner",
    )

    # Calculate absolute error
    merged_df["abs_error"] = (
        merged_df["probability"] - merged_df["llm_probability"]
    ).abs()

    # Compute MAE and std per model
    stats = (
        merged_df.groupby("model_name")["abs_error"]
        .agg(["mean", "std"])
        .to_dict("index")
    )

    # Compute accuracy on all queries except -1000 (nulls count as failures)
    non_error_df = experiments_df[experiments_df["llm_probability"] != -1000].copy()
    non_error_df = non_error_df.merge(
        queries_df[["query_uuid", "probability"]],
        on="query_uuid",
        how="inner",
    )
    # nulls automatically fail the threshold check
    non_error_df["abs_error"] = (
        (non_error_df["probability"] - non_error_df["llm_probability"])
        .abs()
        .where(non_error_df["llm_probability"].notna())
    )
    non_error_df["within_threshold"] = non_error_df["abs_error"] <= accuracy_threshold
    accuracy_stats = (
        non_error_df.groupby("model_name")["within_threshold"].mean().to_dict()
    )

    # Build summary table
    rows = []
    for model in all_models:
        model_stats = stats.get(
            model, {"mean": float("nan"), "std": float("nan"), "count": 0}
        )
        rows.append(
            {
                "model_name": model,
                "supported_queries": supported_counts.get(model, 0),
                "mae": model_stats["mean"],
                "std_error": model_stats["std"],
                "accuracy": accuracy_stats.get(model, float("nan")),
            }
        )

    return pd.DataFrame(rows)


def main():
    """Generate and display the model summary table."""
    print(
        f"Generating model summary (naming_strategy='{NAMING_STRATEGY}', "
        f"experiment_type='{EXPERIMENT_TYPE}')\n"
    )

    summary_df = generate_model_summary()

    # Format for display
    display_df = summary_df.copy()
    display_df["mae"] = display_df["mae"].apply(
        lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
    )
    display_df["std_error"] = display_df["std_error"].apply(
        lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
    )
    display_df["accuracy"] = display_df["accuracy"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )

    print(display_df.to_string(index=False))
    print(f"\nTotal models: {len(summary_df)}")


if __name__ == "__main__":
    main()
