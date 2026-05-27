#!/usr/bin/env python3
"""Plot query-complexity metric distributions by treewidth."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

repo_root = Path(__file__).resolve().parents[2]

METRICS = [
    ("induced_width", "Query induced width", "Width", False),
    ("total_factor_size", "Total factor size", "Count (log scale)", True),
    ("scalar_additions", "Scalar additions", "Count (log scale)", True),
    ("scalar_multiplications", "Scalar multiplications", "Count (log scale)", True),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root
        / "data"
        / "result_analysis"
        / "query_complexity_metrics_by_query.csv",
        help="Query-complexity CSV generated from the benchmark parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "plots"
        / "result_analysis"
        / "query_complexity_boxplots_by_treewidth.pdf",
        help="Output figure path. The suffix controls the Matplotlib format.",
    )
    parser.add_argument(
        "--treewidth-column",
        choices=["target_tw", "achieved_tw"],
        default="target_tw",
        help="BN metadata column used to group query-complexity distributions.",
    )
    parser.add_argument(
        "--naming-strategy",
        default="simple",
        help="Naming strategy to use when the input contains multiple prompt variants.",
    )
    return parser.parse_args()


def load_plot_data(
    input_path: Path,
    treewidth_column: str,
    naming_strategy: str,
) -> pd.DataFrame:
    """Load query-complexity metrics and select one row per benchmark query."""
    df = pd.read_csv(input_path)
    required_columns = {
        "query_uuid",
        "naming_strategy",
        treewidth_column,
        *(metric for metric, *_ in METRICS),
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"{input_path} is missing required columns: {missing}")

    df = df[df["naming_strategy"] == naming_strategy].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for naming_strategy={naming_strategy!r} in {input_path}."
        )

    df = df.drop_duplicates(subset=["query_uuid"]).copy()
    df["treewidth"] = df[treewidth_column].astype(int)
    return df


def plot_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    """Create and save the four-panel query-complexity boxplot figure."""
    sns.set_theme(style="whitegrid", context="paper")
    treewidth_order = sorted(df["treewidth"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(7.32, 5.87), sharex=False)
    axes_flat = axes.ravel()

    for ax, (metric, title, ylabel, log_scale) in zip(
        axes_flat,
        METRICS,
        strict=True,
    ):
        sns.boxplot(
            data=df,
            x="treewidth",
            y=metric,
            order=treewidth_order,
            color="#4C72B0",
            linewidth=0.9,
            fliersize=2,
            ax=ax,
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Treewidth", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if log_scale:
            ax.set_yscale("log")

    fig.suptitle(
        "Query-complexity distributions by treewidth",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the query-complexity boxplot figure."""
    args = parse_args()
    df = load_plot_data(args.input, args.treewidth_column, args.naming_strategy)
    plot_boxplots(df, args.output)
    treewidths = ", ".join(str(tw) for tw in sorted(df["treewidth"].unique()))
    print(f"Treewidth column: {args.treewidth_column}")
    print(f"Rows plotted: {len(df)}")
    print(f"Treewidth values: {treewidths}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
