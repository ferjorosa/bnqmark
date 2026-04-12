#!/usr/bin/env python3
"""
Script to generate bar plots showing accuracy per model and treewidth.

For a given network size, this script:
1. Loads experiments, queries, and BNs from parquet files
2. Filters by network size, naming strategy, and experiment type
3. Calculates accuracy per model and treewidth using 0.01 threshold
4. Creates a bar plot with treewidth on X-axis and accuracy on Y-axis
5. Saves the plot to the plots directory with a unique filename
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration - Edit these values to customize the plot
NETWORK_SIZE = 20  # Network size (n) to analyze
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive", etc.
EXPERIMENT_TYPE = "code_generation"  # Options: "raw_reasoning", "code_generation"
ACCURACY_THRESHOLD = 0.01  # Threshold for considering a prediction accurate

# Plotting configuration
FIGURE_SIZE = (16, 8)  # Wider to fit 9 models across 6 treewidths
BAR_WIDTH = 0.08  # Narrower bars to fit all models
DPI = 300


def load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load experiments, queries, and BNs from parquet files."""
    experiments_path = repo_root / "data" / "experiments.parquet"
    queries_path = repo_root / "data" / "queries.parquet"
    bns_path = repo_root / "data" / "bns.parquet"

    print(f"Loading experiments from {experiments_path}...")
    experiments_df = pd.read_parquet(experiments_path)
    print(f"✓ Loaded {len(experiments_df)} experiments")

    print(f"Loading queries from {queries_path}...")
    queries_df = pd.read_parquet(queries_path)
    print(f"✓ Loaded {len(queries_df)} queries")

    print(f"Loading BNs from {bns_path}...")
    bns_df = pd.read_parquet(bns_path)
    print(f"✓ Loaded {len(bns_df)} BNs\n")

    return experiments_df, queries_df, bns_df


def filter_data(
    experiments_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter data by network size, naming strategy, and experiment type."""
    # Filter BNs by network size
    bns_filtered = bns_df[bns_df["n"] == NETWORK_SIZE].copy()
    print(f"Filtered to {len(bns_filtered)} BNs with n={NETWORK_SIZE}")

    # Get valid bn_uuids
    valid_bn_uuids = set(bns_filtered["bn_uuid"].unique())

    # Filter queries to only those from BNs with the target network size
    queries_filtered = queries_df[queries_df["bn_uuid"].isin(valid_bn_uuids)].copy()
    print(f"Filtered to {len(queries_filtered)} queries for n={NETWORK_SIZE}")

    # Get valid query_uuids
    valid_query_uuids = set(queries_filtered["query_uuid"].unique())

    # Filter experiments by naming_strategy, experiment_type, and valid queries
    experiments_filtered = experiments_df[
        (experiments_df["naming_strategy"] == NAMING_STRATEGY)
        & (experiments_df["experiment_type"] == EXPERIMENT_TYPE)
        & (experiments_df["query_uuid"].isin(valid_query_uuids))
    ].copy()
    print(f"Filtered to {len(experiments_filtered)} experiments")
    print(f"  • Naming strategy: {NAMING_STRATEGY}")
    print(f"  • Experiment type: {EXPERIMENT_TYPE}\n")

    return experiments_filtered, queries_filtered, bns_filtered


def calculate_accuracy(
    experiments_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
) -> dict[str, dict[int, float]]:
    """
    Calculate accuracy per model and treewidth.

    Returns:
        Dictionary mapping model_name -> {treewidth -> accuracy}
    """
    # Create bn_uuid -> treewidth mapping (column is 'achieved_tw' in discrete_bns)
    bn_to_treewidth = bns_df.set_index("bn_uuid")["achieved_tw"].to_dict()

    # Create query_uuid -> bn_uuid mapping
    query_to_bn = queries_df.set_index("query_uuid")["bn_uuid"].to_dict()

    # Add treewidth column to experiments
    experiments_df = experiments_df.copy()
    experiments_df["bn_uuid"] = experiments_df["query_uuid"].map(query_to_bn)
    experiments_df["treewidth"] = experiments_df["bn_uuid"].map(bn_to_treewidth)

    # Get all unique models
    models = experiments_df["model_name"].unique()
    treewidths = sorted(experiments_df["treewidth"].dropna().unique())

    print(f"Found {len(models)} models: {list(models)}")
    print(f"Found {len(treewidths)} treewidths: {treewidths}\n")

    # Calculate accuracy per model and treewidth
    accuracy_data: dict[str, dict[int, float]] = {}

    for model in models:
        accuracy_data[model] = {}
        model_df = experiments_df[experiments_df["model_name"] == model]

        for tw in treewidths:
            tw_df = model_df[model_df["treewidth"] == tw]

            if len(tw_df) == 0:
                accuracy_data[model][tw] = 0.0
                continue

            # Total unique queries for this (model, treewidth) combination
            total_queries = tw_df["query_uuid"].nunique()

            # Successful predictions:
            # - llm_probability is not null
            # - llm_probability != -1000
            # - |probability - llm_probability| <= ACCURACY_THRESHOLD
            # Join with queries to get true probability
            merged = tw_df.merge(
                queries_df[["query_uuid", "probability"]],
                on="query_uuid",
                how="inner",
            )

            # Filter out -1000 values (failures) and nulls
            valid = merged[
                (merged["llm_probability"].notna())
                & (merged["llm_probability"] != -1000)
            ].copy()

            # Count unique queries that have valid predictions
            queries_with_valid = valid["query_uuid"].nunique()

            if queries_with_valid == 0:
                accuracy_data[model][tw] = 0.0
                print(
                    f"  {model} (tw={tw}): 0/{total_queries} = 0.0% "
                    "(no valid predictions)"
                )
                continue

            # Calculate absolute difference per query (taking min if multiple runs)
            valid["diff"] = (valid["probability"] - valid["llm_probability"]).abs()

            # For each query, check if at least one prediction is accurate
            query_accurate = (
                valid.groupby("query_uuid")["diff"].min() <= ACCURACY_THRESHOLD
            )
            accurate_count = query_accurate.sum()

            # Accuracy as percentage (out of total unique queries)
            accuracy = (accurate_count / total_queries) * 100
            accuracy_data[model][tw] = accuracy

            print(
                f"  {model} (tw={tw}): {accurate_count}/{total_queries} = "
                f"{accuracy:.1f}%"
            )

    return accuracy_data, treewidths


def create_bar_plot(
    accuracy_data: dict[str, dict[int, float]],
    treewidths: list[int],
) -> plt.Figure:
    """Create a bar plot showing accuracy per model and treewidth."""
    models = list(accuracy_data.keys())
    num_models = len(models)
    num_treewidths = len(treewidths)

    # Set up the figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # X positions for treewidth groups
    x = np.arange(num_treewidths)

    # Calculate bar width based on number of models
    width = BAR_WIDTH
    total_width = width * num_models
    offset_start = -total_width / 2 + width / 2

    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    # Plot bars for each model
    for i, model in enumerate(models):
        offset = offset_start + i * width
        accuracies = [accuracy_data[model].get(tw, 0.0) for tw in treewidths]
        bars = ax.bar(x + offset, accuracies, width, label=model, color=colors[i])

        # Add value labels on top of bars (only for bars with sufficient height)
        for bar, acc in zip(bars, accuracies, strict=True):
            height = bar.get_height()
            if height > 5:  # Only label if accuracy > 5%
                ax.annotate(
                    f"{acc:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    # Set labels and title
    ax.set_xlabel("Treewidth", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Accuracy per Model and Treewidth for n = {NETWORK_SIZE}\n"
        f"({NAMING_STRATEGY}, {EXPERIMENT_TYPE}, threshold = {ACCURACY_THRESHOLD})",
        fontsize=16,
        fontweight="bold",
    )

    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(treewidths, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # Set y-axis limits
    ax.set_ylim(0, 105)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend with smaller font to fit all models
    ax.legend(
        title="Model",
        loc="upper right",
        bbox_to_anchor=(1.25, 1),
        fontsize=8,
        ncol=1,
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def save_plot(fig: plt.Figure, repo_root: Path) -> Path:
    """Save the plot to the plots directory with a unique filename."""
    plots_dir = repo_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"accuracy_n{NETWORK_SIZE}_{NAMING_STRATEGY}_{EXPERIMENT_TYPE}_{timestamp}.png"
    )
    plot_path = plots_dir / filename

    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_path}")

    return plot_path


def main():
    """Main function to generate the accuracy plot."""
    print("=" * 70)
    print("Generating Accuracy Plot by Treewidth")
    print("=" * 70)
    print("Configuration:")
    print(f"  • Network size (n): {NETWORK_SIZE}")
    print(f"  • Naming strategy: {NAMING_STRATEGY}")
    print(f"  • Experiment type: {EXPERIMENT_TYPE}")
    print(f"  • Accuracy threshold: {ACCURACY_THRESHOLD}")
    print("=" * 70)
    print()

    # Get repo root
    repo_root = Path(__file__).resolve().parents[2]

    # Load data
    experiments_df, queries_df, bns_df = load_data(repo_root)

    # Filter data
    experiments_filtered, queries_filtered, bns_filtered = filter_data(
        experiments_df, queries_df, bns_df
    )

    if len(experiments_filtered) == 0:
        print("No experiments found matching the criteria. Exiting.")
        return

    # Calculate accuracy
    print("\nCalculating accuracy per model and treewidth...")
    print("-" * 70)
    accuracy_data, treewidths = calculate_accuracy(
        experiments_filtered, queries_filtered, bns_filtered
    )

    if not accuracy_data or not treewidths:
        print("No data to plot. Exiting.")
        return

    # Create plot
    print("\nCreating bar plot...")
    fig = create_bar_plot(accuracy_data, treewidths)

    # Save plot
    save_plot(fig, repo_root)

    # Show plot (non-blocking)
    plt.show()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
