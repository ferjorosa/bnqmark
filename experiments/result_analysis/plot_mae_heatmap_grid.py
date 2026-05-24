#!/usr/bin/env python3
"""
Script to generate a grid of MAE (Mean Absolute Error) heatmaps (3x3).

MAE = average of |pmodel - ptrue| over all evaluated queries.

Creates one figure with 3x3 subplots, each showing MAE heatmap for a model.
Each heatmap shows:
- Y-axis: network size (n)
- X-axis: configurable treewidth column (`achieved_tw` or `target_tw`)
- Cell values: MAE (0-1, where 0 = perfect predictions, higher = worse)

Requires global configuration for naming_strategy and experiment_type.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configuration - Edit these values to customize the output
NAMING_STRATEGY = "simple"  # Options: "simple", "descriptive", etc.
EXPERIMENT_TYPE = "code_generation"  # Options: "raw_reasoning", "code_generation"
TREEWIDTH_COLUMN = "target_tw"  # Options: "achieved_tw", "target_tw"
OUTPUT_FORMAT = "pdf"  # Options: "png", "pdf"

# Grid configuration
FIGURE_SIZE = (16, 16)  # Large figure to accommodate 3x3 grid
ANNOT_FONT_SIZE = 8  # Cell value font size
AXIS_LABEL_FONT_SIZE = 10  # X/Y axis label font size
AXIS_TICK_FONT_SIZE = 8  # X/Y axis tick label font size
TITLE_FONT_SIZE = 12  # Subplot title font size
LABEL_PAD = 10  # Padding between axis labels and tick labels
TITLE_PAD = 15  # Padding between subplot title and plot area


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


def calculate_mae_matrix(
    experiments_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate MAE matrix with network sizes as rows and treewidths as columns.

    MAE = mean of |pmodel - ptrue| over all valid predictions.

    Returns:
        DataFrame with n as index, treewidth as columns, MAE as values
    """
    # Filter BNs by the naming_strategy
    bns_filtered = bns_df[bns_df["naming_strategy"] == NAMING_STRATEGY].copy()

    # Create bn_uuid -> (n, selected treewidth) mapping
    bns_unique = bns_filtered.drop_duplicates(subset=["bn_uuid"])
    bn_metadata = bns_unique.set_index("bn_uuid")[["n", TREEWIDTH_COLUMN]].to_dict(
        "index"
    )

    # Create query_uuid -> bn_uuid mapping
    query_to_bn = queries_df.set_index("query_uuid")["bn_uuid"].to_dict()

    # Add n and treewidth columns to experiments
    experiments_df = experiments_df.copy()
    experiments_df["bn_uuid"] = experiments_df["query_uuid"].map(query_to_bn)
    experiments_df["n"] = experiments_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get("n")
    )
    experiments_df[TREEWIDTH_COLUMN] = experiments_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get(TREEWIDTH_COLUMN)
    )

    # Get all unique network sizes and treewidths
    network_sizes = sorted(experiments_df["n"].dropna().unique())
    treewidths = sorted(experiments_df[TREEWIDTH_COLUMN].dropna().unique())

    # Initialize MAE matrix (rows = n, columns = treewidth)
    mae_matrix = pd.DataFrame(
        index=[int(n) for n in network_sizes],
        columns=[int(tw) for tw in treewidths],
        dtype=float,
    )

    # Calculate MAE for each (n, treewidth) combination
    for n in network_sizes:
        for tw in treewidths:
            # Filter experiments for this (n, treewidth) combination
            subset = experiments_df[
                (experiments_df["n"] == n) & (experiments_df[TREEWIDTH_COLUMN] == tw)
            ]

            if len(subset) == 0:
                # No experiments at all for this combination -> use sentinel -1
                mae_matrix.loc[int(n), int(tw)] = -1
                continue

            # Merge with queries to get true probability
            merged = subset.merge(
                queries_df[["query_uuid", "probability"]],
                on="query_uuid",
                how="inner",
            )

            # Filter out -1000 values (failures) and nulls
            valid = merged[
                (merged["llm_probability"].notna())
                & (merged["llm_probability"] != -1000)
            ].copy()

            # Mark as N/A only when ALL non-null predictions are the -1000 sentinel
            # used for context-length failures. Nulls are excluded from MAE calc
            # (not counted), but don't trigger N/A.
            all_predictions = subset["llm_probability"]
            non_null_predictions = all_predictions[all_predictions.notna()]

            if len(non_null_predictions) > 0 and (non_null_predictions == -1000).all():
                # All non-null predictions are -1000 -> N/A (black cell, context limit)
                mae_matrix.loc[int(n), int(tw)] = np.nan
                continue

            if len(non_null_predictions) == 0:
                # All predictions are NULL -> N/A (red cell, no parseable answer)
                mae_matrix.loc[int(n), int(tw)] = -2
                continue

            # Calculate absolute differences
            valid["diff"] = (valid["probability"] - valid["llm_probability"]).abs()

            # Calculate MAE (mean of all absolute differences)
            mae = valid["diff"].mean()
            mae_matrix.loc[int(n), int(tw)] = mae

    return mae_matrix


def create_heatmap_subplot(
    ax: plt.Axes,
    mae_matrix: pd.DataFrame,
    model: str,
) -> None:
    """Create a single heatmap subplot."""
    # Create colormap for MAE: white (good/low error) -> light green
    # -> dark green (bad/high error)
    colors = ["#FFFFFF", "#C8E6C9", "#81C784", "#4CAF50", "#1B5E20"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("mae", colors, N=n_bins)

    # Prepare annotation matrix and identify cell types
    annot_matrix = mae_matrix.copy().astype(object)
    na_cells_black = []  # (row, col) positions for N/A (black cells: all -1000)
    na_cells_red = []  # (row, col) positions for N/A (red cells: all NULL)
    dash_cells = []  # (row, col) positions for "-" (white cells)

    for i, row_idx in enumerate(mae_matrix.index):
        for j, col_idx in enumerate(mae_matrix.columns):
            val = mae_matrix.loc[row_idx, col_idx]
            if pd.isna(val):
                annot_matrix.loc[row_idx, col_idx] = "N/A"
                na_cells_black.append((i, j))
            elif val == -2:
                annot_matrix.loc[row_idx, col_idx] = "N/A"
                na_cells_red.append((i, j))
            elif val == -1:
                annot_matrix.loc[row_idx, col_idx] = "-"
                dash_cells.append((i, j))
            else:
                annot_matrix.loc[row_idx, col_idx] = f"{val:.2f}"

    # Fill NaN and -2 with 0 for visualization
    plot_matrix = mae_matrix.copy()
    plot_matrix = plot_matrix.fillna(0)
    for i, j in dash_cells:
        plot_matrix.iloc[i, j] = 0
    for i, j in na_cells_red:
        plot_matrix.iloc[i, j] = 0

    # Create heatmap with fixed range 0-1
    sns.heatmap(
        plot_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=1,  # Fixed range 0-1 for MAE
        cbar=False,  # No colorbar for individual subplots
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": ANNOT_FONT_SIZE, "color": "black"},
    )

    # Overlay black cells for N/A (all -1000, context limit)
    for i, j in na_cells_black:
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

    # Overlay red cells for N/A (all NULL, no parseable answer)
    for i, j in na_cells_red:
        ax.add_patch(
            plt.Rectangle(
                (j, i),
                1,
                1,
                fill=True,
                facecolor="red",
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

    # Set labels with padding
    ax.set_xlabel(
        "Treewidth",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        labelpad=LABEL_PAD,
    )
    ax.set_ylabel(
        "Network Size (n)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        labelpad=LABEL_PAD,
    )

    # Format title (clean up model name) with padding
    model_short = model.split("/")[-1] if "/" in model else model
    ax.set_title(
        model_short, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=TITLE_PAD
    )

    # Ensure y-axis labels are integers
    ax.set_yticklabels(
        [int(x) for x in mae_matrix.index], rotation=0, fontsize=AXIS_TICK_FONT_SIZE
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=AXIS_TICK_FONT_SIZE)


def create_grid_figure(
    all_matrices: dict[str, pd.DataFrame],
) -> plt.Figure:
    """Create a 3x3 grid figure with all model heatmaps."""
    models = list(all_matrices.keys())
    n_models = len(models)

    # Create figure with 3x3 grid and space for colorbar
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

    # Plot each model
    for i, model in enumerate(models):
        ax = axes[i]
        mae_matrix = all_matrices[model]
        create_heatmap_subplot(ax, mae_matrix, model)

    # Hide unused subplots if less than 9 models
    for i in range(n_models, 9):
        axes[i].set_visible(False)

    # Add shared colorbar at the figure level
    # Create colormap for colorbar: white -> green
    colors = ["#FFFFFF", "#C8E6C9", "#81C784", "#4CAF50", "#1B5E20"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("mae", colors, N=n_bins)

    # Add colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("MAE", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Format title based on experiment type
    if EXPERIMENT_TYPE == "raw_reasoning":
        title_text = "MAE - Raw Reasoning"
    elif EXPERIMENT_TYPE == "code_generation":
        title_text = "MAE - Code Generation"
    else:
        title_text = f"MAE - {EXPERIMENT_TYPE}"

    # Add big title for entire figure
    fig.suptitle(
        title_text,
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    return fig


def save_plot(fig: plt.Figure, plots_dir: Path) -> Path:
    """Save the grid figure to the plots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"mae_heatmap_grid_{NAMING_STRATEGY}_{EXPERIMENT_TYPE}_{timestamp}."
        f"{OUTPUT_FORMAT}"
    )
    plot_path = plots_dir / filename

    if OUTPUT_FORMAT.lower() == "pdf":
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Grid heatmap saved to: {plot_path}")

    return plot_path


def main():
    """Main function to generate MAE heatmap grid."""
    print("=" * 70)
    print("Generating MAE Heatmap Grid (3x3)")
    print("=" * 70)
    print("Configuration:")
    print(f"  • Naming strategy: {NAMING_STRATEGY}")
    print(f"  • Experiment type: {EXPERIMENT_TYPE}")
    print(f"  • Treewidth column: {TREEWIDTH_COLUMN}")
    print("=" * 70)
    print()

    # Get repo root
    repo_root = Path(__file__).resolve().parents[2]

    # Create plots directory
    plots_dir = repo_root / "plots" / "mae_heatmap"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    experiments_df, queries_df, bns_df = load_data(repo_root)

    # Filter experiments by naming_strategy and experiment_type
    filtered_exp = experiments_df[
        (experiments_df["naming_strategy"] == NAMING_STRATEGY)
        & (experiments_df["experiment_type"] == EXPERIMENT_TYPE)
    ].copy()

    print(f"Filtered to {len(filtered_exp)} experiments")
    print(f"  • Naming strategy: {NAMING_STRATEGY}")
    print(f"  • Experiment type: {EXPERIMENT_TYPE}\n")

    if len(filtered_exp) == 0:
        print("No experiments found matching the criteria. Exiting.")
        return

    # Get all unique models
    models = filtered_exp["model_name"].unique()
    print(f"Found {len(models)} models: {list(models)}\n")

    # Calculate MAE matrix for each model
    print("Calculating MAE matrices for each model...")
    print("-" * 70)
    all_matrices = {}

    for model in models:
        print(f"  Processing {model}...")
        model_df = filtered_exp[filtered_exp["model_name"] == model]
        mae_matrix = calculate_mae_matrix(model_df, queries_df, bns_df)
        all_matrices[model] = mae_matrix

    print(f"\n✓ Calculated {len(all_matrices)} MAE matrices\n")

    # Create grid figure
    print("Creating 3x3 grid figure...")
    fig = create_grid_figure(all_matrices)

    # Save plot
    save_plot(fig, plots_dir)

    plt.close(fig)

    print("\n" + "=" * 70)
    print("✓ Grid heatmap generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
