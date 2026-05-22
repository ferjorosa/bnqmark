#!/usr/bin/env python3
"""Summarize arithmetic complexity in generated code responses."""

import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.code_analysis_utils import (
    analyze_response_arithmetic,
    extract_code_text,
)

# Configuration - edit these to filter experiments.
NAMING_STRATEGY = "simple"
EXPERIMENT_TYPE = "code_generation"
MANUAL_ONLY = True

PGMPY_PATTERNS = ("import pgmpy", "from pgmpy", "pgmpy.")
PYAGRUM_PATTERNS = ("import pyagrum", "from pyagrum", "pyagrum.")


def load_experiments(parquet_path: Path | None = None) -> pd.DataFrame:
    """Load experiments from parquet file."""
    if parquet_path is None:
        parquet_path = repo_root / "data" / "experiments.parquet"
    return pd.read_parquet(parquet_path)


def classify_code_style(response: str) -> str:
    """Classify generated code as BN-library based or manual/static Python."""
    code_text = extract_code_text(response).lower()
    uses_pgmpy = any(pattern in code_text for pattern in PGMPY_PATTERNS)
    uses_pyagrum = any(pattern in code_text for pattern in PYAGRUM_PATTERNS)

    if uses_pgmpy and uses_pyagrum:
        return "both"
    if uses_pgmpy:
        return "pgmpy"
    if uses_pyagrum:
        return "pyagrum"
    return "manual"


def add_arithmetic_metrics(experiments_df: pd.DataFrame) -> pd.DataFrame:
    """Add static arithmetic metrics to experiment rows."""
    df = experiments_df.copy()
    metrics = df["response"].apply(analyze_response_arithmetic)

    df["code_style"] = df["response"].apply(classify_code_style)
    df["arithmetic_operator_count"] = metrics.apply(
        lambda metric: metric.arithmetic_operator_count
    )
    df["largest_factor_size"] = metrics.apply(lambda metric: metric.largest_factor_size)
    df["code_parse_error"] = metrics.apply(lambda metric: metric.parse_error)

    for operator in ["+", "-", "*", "/", "//", "%", "**", "@", "unary+", "unary-"]:
        df[f"op_{operator}"] = metrics.apply(
            lambda metric, operator=operator: metric.operator_counts.get(operator, 0)
        )

    return df


def generate_arithmetic_summary(
    naming_strategy: str = NAMING_STRATEGY,
    experiment_type: str = EXPERIMENT_TYPE,
    manual_only: bool = MANUAL_ONLY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate row-level arithmetic metrics and a per-model summary."""
    experiments_df = load_experiments()
    filtered_df = experiments_df[
        (experiments_df["naming_strategy"] == naming_strategy)
        & (experiments_df["experiment_type"] == experiment_type)
    ].copy()
    metrics_df = add_arithmetic_metrics(filtered_df)

    if manual_only:
        metrics_df = metrics_df[metrics_df["code_style"] == "manual"].copy()

    grouped = metrics_df.groupby("model_name", dropna=False)
    summary_df = grouped.agg(
        responses=("query_uuid", "count"),
        parse_failures=("code_parse_error", lambda values: values.notna().sum()),
        mean_ops=("arithmetic_operator_count", "mean"),
        median_ops=("arithmetic_operator_count", "median"),
        max_ops=("arithmetic_operator_count", "max"),
        mean_largest_factor=("largest_factor_size", "mean"),
        median_largest_factor=("largest_factor_size", "median"),
        max_largest_factor=("largest_factor_size", "max"),
    )

    return metrics_df, summary_df.reset_index()


def main() -> None:
    """Print arithmetic metrics for generated code responses."""
    scope = "manual code-generation" if MANUAL_ONLY else "all code-generation"
    print(
        f"Generating arithmetic summary for {scope} rows "
        f"(naming_strategy='{NAMING_STRATEGY}')\n"
    )

    metrics_df, summary_df = generate_arithmetic_summary()
    print(f"Rows analyzed: {len(metrics_df)}\n")

    display_df = summary_df.copy()
    for column in [
        "mean_ops",
        "median_ops",
        "mean_largest_factor",
        "median_largest_factor",
    ]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")

    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
