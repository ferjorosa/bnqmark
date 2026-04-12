#!/usr/bin/env python3
"""
Script to generate heatmaps showing accuracy per network size and treewidth.

Creates one heatmap per (model, experiment_type, naming_strategy) combination.
Each heatmap shows:
- Y-axis: network size (n)
- X-axis: achieved treewidth
- Cell values: accuracy percentage (with "N/A" for missing data)

Color scheme: warm colors where 0% and N/A are white.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configuration - Edit these values to customize the output
ACCURACY_THRESHOLD = 0.01  # Threshold for considering a prediction accurate

# Heatmap configuration
FIGURE_SIZE = (10, 8)
DPI = 300
ANNOT_FONT_SIZE = 10


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


def calculate_accuracy_matrix(
    experiments_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate accuracy matrix with network sizes as rows and treewidths as columns.

    Returns:
        DataFrame with n as index, treewidth as columns, accuracy as values
    """
    # Get the naming_strategy from experiments (already filtered)
    naming_strategy = experiments_df["naming_strategy"].iloc[0]

    # Filter BNs by the same naming_strategy
    bns_filtered = bns_df[bns_df["naming_strategy"] == naming_strategy].copy()

    # Create bn_uuid -> (n, achieved_tw) mapping for this naming_strategy
    bns_unique = bns_filtered.drop_duplicates(subset=["bn_uuid"])
    bn_metadata = bns_unique.set_index("bn_uuid")[["n", "achieved_tw"]].to_dict("index")

    # Create query_uuid -> bn_uuid mapping
    query_to_bn = queries_df.set_index("query_uuid")["bn_uuid"].to_dict()

    # Add n and treewidth columns to experiments
    experiments_df = experiments_df.copy()
    experiments_df["bn_uuid"] = experiments_df["query_uuid"].map(query_to_bn)
    experiments_df["n"] = experiments_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get("n")
    )
    experiments_df["achieved_tw"] = experiments_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get("achieved_tw")
    )

    # Get all unique network sizes and treewidths
    network_sizes = sorted(experiments_df["n"].dropna().unique())
    treewidths = sorted(experiments_df["achieved_tw"].dropna().unique())

    print(f"Network sizes: {network_sizes}")
    print(f"Treewidths: {treewidths}\n")

    # Initialize accuracy matrix (rows = n, columns = treewidth)
    accuracy_matrix = pd.DataFrame(
        index=[int(n) for n in network_sizes],
        columns=[int(tw) for tw in treewidths],
        dtype=float,
    )

    # Calculate accuracy for each (n, treewidth) combination
    for n in network_sizes:
        for tw in treewidths:
            # Filter experiments for this (n, treewidth) combination
            subset = experiments_df[
                (experiments_df["n"] == n) & (experiments_df["achieved_tw"] == tw)
            ]

            if len(subset) == 0:
                # No experiments at all for this combination -> use sentinel -1
                accuracy_matrix.loc[int(n), int(tw)] = -1
                continue

            # Total unique queries for this combination
            total_queries = subset["query_uuid"].nunique()

            # Merge with queries to get true probability
            merged = subset.merge(
                queries_df[["query_uuid", "probability"]],
                on="query_uuid",
                how="inner",
            )

            # Check if ALL predictions are -1000 (complete failure = N/A)
            all_predictions = merged["llm_probability"]
            valid_predictions = all_predictions[
                all_predictions.notna() & (all_predictions != -1000)
            ]

            if len(valid_predictions) == 0:
                # All predictions are -1000 or null -> N/A (black cell)
                accuracy_matrix.loc[int(n), int(tw)] = np.nan
                continue

            # Filter out -1000 values (failures) and nulls for accuracy calculation
            valid = merged[
                (merged["llm_probability"].notna())
                & (merged["llm_probability"] != -1000)
            ].copy()

            # Calculate absolute difference per query
            valid["diff"] = (valid["probability"] - valid["llm_probability"]).abs()

            # For each query, check if at least one prediction is accurate
            query_min_diff = valid.groupby("query_uuid")["diff"].min()
            accurate_count = (query_min_diff <= ACCURACY_THRESHOLD).sum()

            # Accuracy as proportion 0-1 (out of total unique queries)
            accuracy = accurate_count / total_queries
            accuracy_matrix.loc[int(n), int(tw)] = accuracy

    return accuracy_matrix


def create_heatmap(
    accuracy_matrix: pd.DataFrame,
    model: str,
    experiment_type: str,
    naming_strategy: str,
) -> plt.Figure:
    """Create a heatmap for a specific model/experiment_type/naming_strategy combo."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create custom colormap: white for 0, warm colors for higher values
    # White -> Yellow -> Orange -> Red
    colors = ["#FFFFFF", "#FFE5B4", "#FFA500", "#FF4500", "#8B0000"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("warm", colors, N=n_bins)

    # Prepare annotation matrix and identify cell types
    annot_matrix = accuracy_matrix.copy().astype(object)
    na_cells = []  # (row, col) positions for N/A (black cells)
    dash_cells = []  # (row, col) positions for "-" (white cells)

    for i, row_idx in enumerate(accuracy_matrix.index):
        for j, col_idx in enumerate(accuracy_matrix.columns):
            val = accuracy_matrix.loc[row_idx, col_idx]
            if pd.isna(val):
                # Check if subset was empty (no data at all) or all -1000
                # We need to track this during calculation - use a sentinel value
                annot_matrix.loc[row_idx, col_idx] = "N/A"
                na_cells.append((i, j))
            elif val == -1:
                # Sentinel for no data at all
                annot_matrix.loc[row_idx, col_idx] = "-"
                dash_cells.append((i, j))
            else:
                annot_matrix.loc[row_idx, col_idx] = f"{val:.2f}"

    # Fill NaN with 0 for visualization, we'll overlay black/white cells
    plot_matrix = accuracy_matrix.copy()
    plot_matrix = plot_matrix.fillna(0)
    for i, j in dash_cells:
        plot_matrix.iloc[i, j] = 0

    # Create heatmap
    sns.heatmap(
        plot_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": "Accuracy",
            "shrink": 0.8,
        },
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": ANNOT_FONT_SIZE, "color": "black"},
    )

    # Overlay black cells for N/A (all -1000)
    for i, j in na_cells:
        ax.add_patch(
            plt.Rectangle(
                (j, i),
                1,
                1,
                fill=True,
                facecolor="black",
                edgecolor="gray",
                linewidth=0.5,
                zorder=10,
            )
        )
        # Add white text for N/A
        ax.text(
            j + 0.5,
            i + 0.5,
            "N/A",
            ha="center",
            va="center",
            fontsize=ANNOT_FONT_SIZE,
            color="white",
            fontweight="bold",
            zorder=11,
        )

    # Set labels
    ax.set_xlabel("Treewidth", fontsize=14, fontweight="bold")
    ax.set_ylabel("Network Size (n)", fontsize=14, fontweight="bold")

    # Format title (clean up model name)
    model_short = model.split("/")[-1] if "/" in model else model
    ax.set_title(
        f"Accuracy Heatmap\nModel: {model_short}\n"
        f"({naming_strategy}, {experiment_type}, threshold = {ACCURACY_THRESHOLD})",
        fontsize=14,
        fontweight="bold",
    )

    # Ensure y-axis labels are integers
    ax.set_yticklabels([int(x) for x in accuracy_matrix.index], rotation=0)

    plt.tight_layout()

    return fig


def save_plot(
    fig: plt.Figure,
    model: str,
    experiment_type: str,
    naming_strategy: str,
    plots_dir: Path,
) -> Path:
    """Save the heatmap to the plots directory."""
    # Clean model name for filename
    model_clean = model.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"heatmap_{model_clean}_{naming_strategy}_{experiment_type}_{timestamp}.png"
    )
    plot_path = plots_dir / filename

    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Heatmap saved to: {plot_path}")

    return plot_path


def main():
    """Main function to generate accuracy heatmaps."""
    print("=" * 70)
    print("Generating Accuracy Heatmaps")
    print("=" * 70)
    print("Configuration:")
    print(f"  • Accuracy threshold: {ACCURACY_THRESHOLD}")
    print("=" * 70)
    print()

    # Get repo root
    repo_root = Path(__file__).resolve().parents[2]

    # Create plots directory
    plots_dir = repo_root / "plots" / "accuracy_heatmap"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    experiments_df, queries_df, bns_df = load_data(repo_root)

    # Get all unique combinations of (model, experiment_type, naming_strategy)
    combinations = (
        experiments_df[["model_name", "experiment_type", "naming_strategy"]]
        .drop_duplicates()
        .values.tolist()
    )

    print(f"Found {len(combinations)} unique combinations to plot\n")
    print("-" * 70)

    # Generate one heatmap per combination
    for i, (model, exp_type, naming_strat) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Processing:")
        print(f"  • Model: {model}")
        print(f"  • Experiment type: {exp_type}")
        print(f"  • Naming strategy: {naming_strat}")

        # Filter experiments for this combination
        filtered_exp = experiments_df[
            (experiments_df["model_name"] == model)
            & (experiments_df["experiment_type"] == exp_type)
            & (experiments_df["naming_strategy"] == naming_strat)
        ].copy()

        if len(filtered_exp) == 0:
            print("  ✗ No data for this combination, skipping...")
            continue

        print(f"  • {len(filtered_exp)} experiment rows")

        # Calculate accuracy matrix
        accuracy_matrix = calculate_accuracy_matrix(filtered_exp, queries_df, bns_df)

        if accuracy_matrix.empty or accuracy_matrix.isna().all().all():
            print("  ✗ No valid accuracy data, skipping...")
            continue

        print(f"  • Matrix shape: {accuracy_matrix.shape}")

        # Create heatmap
        fig = create_heatmap(accuracy_matrix, model, exp_type, naming_strat)

        # Save plot
        save_plot(fig, model, exp_type, naming_strat, plots_dir)

        plt.close(fig)

    print("\n" + "=" * 70)
    print("✓ All heatmaps generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
