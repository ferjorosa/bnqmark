#!/usr/bin/env python3
"""Summarize token and cost efficiency on the common-correct query set."""

import json
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]

# Configuration - Edit these to filter experiments
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive"
EXPERIMENT_TYPES = ("raw_reasoning", "code_generation")
ACCURACY_THRESHOLD = 0.01


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


def parse_usage_metadata(value) -> dict:
    """Parse usage metadata from JSON string or dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def compute_common_correct_efficiency_summary(
    naming_strategy: str = NAMING_STRATEGY,
    experiment_type: str = "raw_reasoning",
    accuracy_threshold: float = ACCURACY_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    """Build a matched-efficiency table on the common-correct query set."""
    experiments_df = load_experiments()
    queries_df = load_queries()

    experiments_df = experiments_df[
        (experiments_df["naming_strategy"] == naming_strategy)
        & (experiments_df["experiment_type"] == experiment_type)
    ].copy()
    queries_df = queries_df[queries_df["naming_strategy"] == naming_strategy].copy()

    all_models = sorted(experiments_df["model_name"].unique().tolist())

    valid_df = experiments_df[
        (experiments_df["llm_probability"].notna())
        & (experiments_df["llm_probability"] != -1000)
    ].copy()
    merged_df = valid_df.merge(
        queries_df[["query_uuid", "probability"]],
        on="query_uuid",
        how="inner",
    )
    merged_df["abs_error"] = (
        merged_df["probability"] - merged_df["llm_probability"]
    ).abs()
    merged_df["within_threshold"] = merged_df["abs_error"] <= accuracy_threshold

    success_matrix = (
        merged_df.pivot(
            index="query_uuid", columns="model_name", values="within_threshold"
        )
        .reindex(columns=all_models)
        .fillna(False)
    )
    common_query_uuids = success_matrix.index[success_matrix.all(axis=1)]
    common_df = merged_df[merged_df["query_uuid"].isin(common_query_uuids)].copy()

    usage_df = common_df["usage_metadata"].apply(parse_usage_metadata).apply(pd.Series)
    common_df["reasoning_tokens"] = usage_df.get("reasoning_tokens")
    common_df["cost"] = usage_df.get("upstream_inference_cost")
    common_df["non_reasoning_output_tokens"] = (
        common_df["output_tokens"] - common_df["reasoning_tokens"]
    ).where(common_df["output_tokens"].notna() & common_df["reasoning_tokens"].notna())

    rows = []
    for model in all_models:
        model_df = common_df[common_df["model_name"] == model]
        rows.append(
            {
                "model_name": model,
                "common_correct_queries": len(common_query_uuids),
                "median_cost": model_df["cost"].median(),
                "median_input_tokens": model_df["input_tokens"].median(),
                "median_output_tokens": model_df["output_tokens"].median(),
                "median_reasoning_tokens": model_df["reasoning_tokens"].median(),
                "median_non_reasoning_output_tokens": model_df[
                    "non_reasoning_output_tokens"
                ].median(),
            }
        )

    return pd.DataFrame(rows), len(common_query_uuids)


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for display."""
    display_df = summary_df.copy()

    token_columns = [
        "median_input_tokens",
        "median_output_tokens",
        "median_reasoning_tokens",
        "median_non_reasoning_output_tokens",
    ]

    display_df["median_cost"] = display_df["median_cost"].apply(
        lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
    )

    for column in token_columns:
        display_df[column] = display_df[column].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

    return display_df


def main():
    """Generate and display one table per experiment type."""
    print(
        f"Generating common-correct efficiency summary "
        f"(naming_strategy='{NAMING_STRATEGY}', "
        f"accuracy_threshold={ACCURACY_THRESHOLD})\n"
    )

    for experiment_type in EXPERIMENT_TYPES:
        summary_df, common_count = compute_common_correct_efficiency_summary(
            naming_strategy=NAMING_STRATEGY,
            experiment_type=experiment_type,
            accuracy_threshold=ACCURACY_THRESHOLD,
        )
        display_df = format_summary_table(summary_df)

        print(f"{experiment_type}:\n")
        print(display_df.to_string(index=False))
        print(f"\nCommon correct queries across all models: {common_count}\n")


if __name__ == "__main__":
    main()
