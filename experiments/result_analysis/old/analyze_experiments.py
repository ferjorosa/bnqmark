#!/usr/bin/env python3
"""Simple script to load and display discrete_experiments data from parquet files."""

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure src is importable
repo_root = Path(__file__).resolve().parents[2]
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

# Configuration - Edit these values to filter experiments
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive", etc.
EXPERIMENT_TYPE = "raw_reasoning"  # Options: "raw_reasoning", "code_generation"


def load_bn_dataset(parquet_path: Path | None = None) -> pd.DataFrame:
    """
    Load BN dataset from parquet file.

    Args:
        parquet_path: Optional path to the parquet file. If None, uses default location.

    Returns:
        DataFrame containing Bayesian networks
    """
    if parquet_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        parquet_path = repo_root / "data" / "bns.parquet"

    print(f"Loading BN dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"✓ Loaded {len(df)} Bayesian networks\n")
    return df


def load_query_dataset(parquet_path: Path | None = None) -> pd.DataFrame:
    """
    Load query dataset from parquet file.

    Args:
        parquet_path: Optional path to the parquet file. If None, uses default location.

    Returns:
        DataFrame containing queries with deserialized target and evidence columns
    """
    if parquet_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        parquet_path = repo_root / "data" / "queries.parquet"

    print(f"Loading query dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Deserialize JSON strings back to dictionaries for target and evidence
    df["target"] = df["target"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x,
    )
    df["evidence"] = df["evidence"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if x else {}),
    )

    print(f"✓ Loaded {len(df)} queries\n")
    return df


def load_experiments(parquet_path: Path | None = None) -> pd.DataFrame:
    """
    Load all discrete_experiments rows from parquet file.

    Args:
        parquet_path: Optional path to the parquet file. If None, uses default location.

    Returns:
        DataFrame containing all experiment rows
    """
    if parquet_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        parquet_path = repo_root / "data" / "experiments.parquet"

    print(f"Loading experiments from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"✓ Loaded {len(df)} experiments\n")
    return df


def load_queries_from_parquet(parquet_path: Path | None = None) -> pd.DataFrame:
    """
    Load all discrete_queries rows from parquet file.

    Args:
        parquet_path: Optional path to the parquet file. If None, uses default location.

    Returns:
        DataFrame containing all query rows with true probabilities
    """
    if parquet_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        parquet_path = repo_root / "data" / "queries.parquet"

    print(f"Loading queries from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    # Select only the columns we need
    df = df[["query_uuid", "naming_strategy", "bn_uuid", "probability"]].copy()
    print(f"✓ Loaded {len(df)} queries\n")
    return df


def generate_model_dataframes(
    naming_strategy: str = NAMING_STRATEGY,
    experiment_type: str = EXPERIMENT_TYPE,
    run: int | None = None,
    experiments_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate per-model dataframes with valid llm_probability.

    Filters to queries where all models have non-null llm_probability and not -1000.

    This method:
    1. Loads experiments and queries from parquet files (or uses provided
       experiments_df)
    2. Filters to queries where all models have non-null llm_probability and not -1000
    3. Creates a separate dataframe for each model with the requested columns

    Args:
        naming_strategy: Filter by naming strategy (default: NAMING_STRATEGY global)
        experiment_type: Filter by experiment type (default: EXPERIMENT_TYPE global)
        run: Optional run number to filter by. If None, uses all runs.
        experiments_df: Optional pre-loaded experiments dataframe. If None, loads from
            parquet file.

    Returns:
        Dictionary mapping model_name to DataFrame with columns:
        - query_uuid
        - bn_uuid
        - probability (true probability from queries table)
        - model_name
        - input_tokens
        - output_tokens
        - llm_probability
        - started_at
        - finished_at
        - probability_diff_abs (probability - llm_probability)
    """
    if experiments_df is None:
        experiments_df = load_experiments()
    else:
        print("Using provided experiments dataframe...")

    queries_df = load_queries_from_parquet()

    # Filter by naming_strategy
    experiments_df = experiments_df[
        experiments_df["naming_strategy"] == naming_strategy
    ]
    queries_df = queries_df[queries_df["naming_strategy"] == naming_strategy]

    # Filter by experiment_type
    experiments_df = experiments_df[
        experiments_df["experiment_type"] == experiment_type
    ]

    # Filter by run if specified
    if run is not None:
        experiments_df = experiments_df[experiments_df["run"] == run]

    print(f"Loaded {len(experiments_df)} experiment rows")
    print(f"Loaded {len(queries_df)} query rows")

    # Get all unique models from the actual data
    all_models = experiments_df["model_name"].unique().tolist()

    print(f"Found {len(all_models)} models: {list(all_models)}")

    # Show row counts per model at different filtering stages
    print("\n" + "=" * 60)
    print("Row counts per model (filtering stages):")
    print("=" * 60)

    # 1. Total rows per model (no filtering)
    print("\n1. Total rows per model (no filtering):")
    print("-" * 60)
    total_counts = experiments_df["model_name"].value_counts()
    for model_name in all_models:
        count = total_counts.get(model_name, 0)
        print(f"  {model_name}: {count} rows")

    # 2. Rows per model minus -1000 values (but keeping nulls)
    print("\n2. Rows per model (excluding -1000, keeping nulls):")
    print("-" * 60)
    experiments_no_minus_1000 = experiments_df[
        experiments_df["llm_probability"] != -1000
    ].copy()
    no_minus_1000_counts = experiments_no_minus_1000["model_name"].value_counts()
    for model_name in all_models:
        count = no_minus_1000_counts.get(model_name, 0)
        print(f"  {model_name}: {count} rows")

    # 3. Rows per model without -1000 and without nulls
    print("\n3. Rows per model (excluding -1000 and nulls):")
    print("-" * 60)
    experiments_with_prob = experiments_df[
        (experiments_df["llm_probability"].notna())
        & (experiments_df["llm_probability"] != -1000)
    ].copy()
    model_counts = experiments_with_prob["model_name"].value_counts()
    for model_name in all_models:
        count = model_counts.get(model_name, 0)
        print(f"  {model_name}: {count} rows")
    print()

    # Find queries where all models have non-null llm_probability and not -1000

    # Group by query_uuid and check if all models have non-null llm_probability
    # and not -1000
    query_llm_prob_counts = (
        experiments_with_prob.groupby("query_uuid")["model_name"]
        .nunique()
        .reset_index(name="num_models_with_prob")
    )

    # Get queries where all models have non-null llm_probability
    complete_queries_with_prob = query_llm_prob_counts[
        query_llm_prob_counts["num_models_with_prob"] == len(all_models)
    ]["query_uuid"].unique()

    print(
        f"Found {len(complete_queries_with_prob)} queries where all {len(all_models)} "
        f"models have non-null llm_probability and not -1000"
    )

    # Filter experiments to only these queries (but keep all experiments, not just
    # those with prob). This ensures we get all columns even if some rows don't have
    # llm_probability. However, we've already filtered to queries where all models
    # have prob, so this should be fine
    filtered_experiments = experiments_with_prob[
        experiments_with_prob["query_uuid"].isin(complete_queries_with_prob)
    ].copy()

    # Merge with queries to get true probability and bn_uuid
    merged_df = filtered_experiments.merge(
        queries_df[["query_uuid", "bn_uuid", "probability"]],
        on="query_uuid",
        how="inner",
    )

    # Calculate probability_diff_abs
    merged_df["probability_diff_abs"] = (
        merged_df["probability"] - merged_df["llm_probability"]
    ).abs()

    # Select only the requested columns
    columns = [
        "query_uuid",
        "bn_uuid",
        "probability",
        "model_name",
        "input_tokens",
        "output_tokens",
        "llm_probability",
        "started_at",
        "finished_at",
        "probability_diff_abs",
    ]

    # Create a dataframe for each model
    model_dfs = {}
    for model_name in all_models:
        model_df = merged_df[merged_df["model_name"] == model_name][columns].copy()
        model_dfs[model_name] = model_df
        print(f"  {model_name}: {len(model_df)} queries")

    return model_dfs


def analyze_probability_differences(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    """
    Analyze probability differences in a model dataframe.

    Args:
        df: DataFrame with probability_diff_abs column (from generate_model_dataframes)
        threshold: Threshold for counting queries with small/large differences

    Returns:
        Dictionary with:
        - num_queries: Total number of queries
        - num_below_threshold: Number of queries with probability_diff_abs <= threshold
        - pct_below_threshold: Percentage of queries below or equal to threshold
        - num_above_threshold: Number of queries with probability_diff_abs > threshold
        - pct_above_threshold: Percentage of queries above threshold
        - avg_diff: Average probability_diff_abs
        - std_diff: Standard deviation of probability_diff_abs
        - min_diff: Minimum probability_diff_abs
        - max_diff: Maximum probability_diff_abs
    """
    if "probability_diff_abs" not in df.columns:
        raise ValueError("DataFrame must contain 'probability_diff_abs' column")

    num_queries = len(df)
    num_below_threshold = len(df[df["probability_diff_abs"] <= threshold])
    pct_below_threshold = (
        (num_below_threshold / num_queries * 100) if num_queries > 0 else 0.0
    )
    num_above_threshold = len(df[df["probability_diff_abs"] > threshold])
    pct_above_threshold = (
        (num_above_threshold / num_queries * 100) if num_queries > 0 else 0.0
    )

    avg_diff = df["probability_diff_abs"].mean()
    std_diff = df["probability_diff_abs"].std()
    min_diff = df["probability_diff_abs"].min()
    max_diff = df["probability_diff_abs"].max()

    return {
        "num_queries": num_queries,
        "num_below_threshold": num_below_threshold,
        "pct_below_threshold": pct_below_threshold,
        "num_above_threshold": num_above_threshold,
        "pct_above_threshold": pct_above_threshold,
        "avg_diff": avg_diff,
        "std_diff": std_diff,
        "min_diff": min_diff,
        "max_diff": max_diff,
    }


def main():
    """Load and display experiment data, then analyze probability differences."""
    print("=" * 60)
    print("Loading discrete_experiments from parquet file...")
    print("=" * 60)
    df = load_experiments()

    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Generate model dataframes and analyze probability differences
    print("\n" + "=" * 60)
    print("Generating model dataframes and analyzing probability differences")
    print(f"  Naming strategy: {NAMING_STRATEGY}")
    print(f"  Experiment type: {EXPERIMENT_TYPE}")
    print("=" * 60)
    model_dfs = generate_model_dataframes(experiments_df=df)

    if not model_dfs:
        print("\nNo model dataframes generated. Exiting.")
        return

    # Analyze each model with thresholds 0.01 and 0.03
    for threshold in [0.01, 0.03]:
        print("\n" + "=" * 60)
        print(f"Probability Difference Analysis (threshold = {threshold})")
        print("=" * 60)

        for model_name, model_df in model_dfs.items():
            print(f"\n{model_name}:")
            print("-" * 60)
            stats = analyze_probability_differences(model_df, threshold)
            print(
                f"  Queries below threshold: {stats['num_below_threshold']}/"
                f"{stats['num_queries']} ({stats['pct_below_threshold']:.1f}%)"
            )
            print(f"  Avg diff: {stats['avg_diff']:.6f}")
            print(f"  Std diff: {stats['std_diff']:.6f}")
            print(f"  Min diff: {stats['min_diff']:.6f}")
            print(f"  Max diff: {stats['max_diff']:.6f}")


if __name__ == "__main__":
    main()
