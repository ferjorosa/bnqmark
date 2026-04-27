#!/usr/bin/env python3
"""Summarize BN library usage in code-generation responses."""

import re
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]

# Configuration - Edit these to filter experiments
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive"
EXPERIMENT_TYPE = "code_generation"  # This script is intended for code generation
ACCURACY_THRESHOLD = 0.01

LIBRARY_BUCKETS = ["pgmpy", "pyagrum", "both", "none"]

CODE_BLOCK_PATTERN = re.compile(
    r"<code>(.*?)</code>|```(?:python)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
PGMPY_PATTERN = re.compile(
    r"\b(?:from\s+pgmpy\b|import\s+pgmpy\b|pgmpy\.)",
    re.IGNORECASE,
)
PYAGRUM_PATTERN = re.compile(
    r"\b(?:from\s+pyagrum\b|import\s+pyagrum\b|pyagrum\.)",
    re.IGNORECASE,
)


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


def extract_code_text(response: str) -> str:
    """Extract code blocks when present; otherwise fall back to full response."""
    if not isinstance(response, str):
        return ""

    matches = CODE_BLOCK_PATTERN.findall(response)
    if matches:
        code_blocks = [(tagged or fenced).strip() for tagged, fenced in matches]
        return "\n\n".join(block for block in code_blocks if block)

    return response


def classify_library_usage(response: str) -> str:
    """Bucket a response by apparent BN library usage."""
    code_text = extract_code_text(response)
    has_pgmpy = bool(PGMPY_PATTERN.search(code_text))
    has_pyagrum = bool(PYAGRUM_PATTERN.search(code_text))

    if has_pgmpy and has_pyagrum:
        return "both"
    if has_pgmpy:
        return "pgmpy"
    if has_pyagrum:
        return "pyagrum"
    return "none"


def generate_library_usage_summary(
    naming_strategy: str = NAMING_STRATEGY,
    experiment_type: str = EXPERIMENT_TYPE,
    accuracy_threshold: float = ACCURACY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate one accuracy table and one MAE/SD table per model.

    Percentages and per-bucket metrics are computed only on valid
    code-generation rows, where llm_probability is neither null nor -1000.
    """
    experiments_df = load_experiments()
    queries_df = load_queries()

    experiments_df = experiments_df[
        (experiments_df["naming_strategy"] == naming_strategy)
        & (experiments_df["experiment_type"] == experiment_type)
    ].copy()
    queries_df = queries_df[queries_df["naming_strategy"] == naming_strategy].copy()

    all_models = experiments_df["model_name"].unique().tolist()

    valid_df = experiments_df[
        (experiments_df["llm_probability"].notna())
        & (experiments_df["llm_probability"] != -1000)
    ].copy()
    valid_df["library_usage"] = valid_df["response"].apply(classify_library_usage)

    merged_df = valid_df.merge(
        queries_df[["query_uuid", "probability"]],
        on="query_uuid",
        how="inner",
    )
    merged_df["abs_error"] = (
        merged_df["probability"] - merged_df["llm_probability"]
    ).abs()
    merged_df["within_threshold"] = merged_df["abs_error"] <= accuracy_threshold

    bucket_counts = (
        merged_df.groupby(["model_name", "library_usage"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LIBRARY_BUCKETS, fill_value=0)
    )
    mae_by_bucket = (
        merged_df.groupby(["model_name", "library_usage"])["abs_error"]
        .mean()
        .unstack()
        .reindex(columns=LIBRARY_BUCKETS)
    )
    sd_by_bucket = (
        merged_df.groupby(["model_name", "library_usage"])["abs_error"]
        .std()
        .unstack()
        .reindex(columns=LIBRARY_BUCKETS)
    )
    accuracy_by_bucket = (
        merged_df.groupby(["model_name", "library_usage"])["within_threshold"]
        .mean()
        .unstack()
        .reindex(columns=LIBRARY_BUCKETS)
    )

    accuracy_rows = []
    mae_sd_rows = []
    for model in all_models:
        valid_count = (
            int(bucket_counts.loc[model].sum()) if model in bucket_counts.index else 0
        )

        accuracy_row = {
            "model_name": model,
            "valid_queries": valid_count,
        }
        mae_sd_row = {
            "model_name": model,
            "valid_queries": valid_count,
        }

        for bucket in LIBRARY_BUCKETS:
            count = (
                int(bucket_counts.loc[model, bucket])
                if model in bucket_counts.index
                else 0
            )
            pct_value = count / valid_count if valid_count > 0 else float("nan")
            accuracy_row[f"pct_{bucket}"] = pct_value
            mae_sd_row[f"pct_{bucket}"] = pct_value

            accuracy_row[f"accuracy_{bucket}"] = (
                accuracy_by_bucket.loc[model, bucket]
                if model in accuracy_by_bucket.index
                else float("nan")
            )
            mae_sd_row[f"mae_{bucket}"] = (
                mae_by_bucket.loc[model, bucket]
                if model in mae_by_bucket.index
                else float("nan")
            )
            mae_sd_row[f"sd_{bucket}"] = (
                sd_by_bucket.loc[model, bucket]
                if model in sd_by_bucket.index
                else float("nan")
            )

        accuracy_rows.append(accuracy_row)
        mae_sd_rows.append(mae_sd_row)

    return pd.DataFrame(accuracy_rows), pd.DataFrame(mae_sd_rows)


def main():
    """Generate and display the library usage summary tables."""
    print(
        f"Generating library usage summary (naming_strategy='{NAMING_STRATEGY}', "
        f"experiment_type='{EXPERIMENT_TYPE}')\n"
    )

    accuracy_df, mae_sd_df = generate_library_usage_summary()

    display_accuracy_df = accuracy_df.copy()
    display_mae_sd_df = mae_sd_df.copy()
    percent_columns = [f"pct_{bucket}" for bucket in LIBRARY_BUCKETS]
    accuracy_columns = [f"accuracy_{bucket}" for bucket in LIBRARY_BUCKETS]
    mae_columns = [f"mae_{bucket}" for bucket in LIBRARY_BUCKETS]
    sd_columns = [f"sd_{bucket}" for bucket in LIBRARY_BUCKETS]

    for column in [*percent_columns, *accuracy_columns]:
        display_accuracy_df[column] = display_accuracy_df[column].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    for column in percent_columns:
        display_mae_sd_df[column] = display_mae_sd_df[column].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    for column in [*mae_columns, *sd_columns]:
        display_mae_sd_df[column] = display_mae_sd_df[column].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
        )

    print("Accuracy table:\n")
    print(display_accuracy_df.to_string(index=False))
    print("\nMAE + SD table:\n")
    print(display_mae_sd_df.to_string(index=False))
    print(f"\nTotal models: {len(accuracy_df)}")


if __name__ == "__main__":
    main()
