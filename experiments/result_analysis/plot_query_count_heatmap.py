#!/usr/bin/env python3
"""
Script to generate a heatmap showing query count per network size and treewidth.

This is a general data overview showing:
- Y-axis: network size (n)
- X-axis: achieved treewidth
- Cell values: number of unique queries

Uses only queries.parquet and bns.parquet (no experiments data).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Heatmap configuration
FIGURE_SIZE = (12, 10)  # Slightly larger to accommodate bigger fonts
DPI = 300
ANNOT_FONT_SIZE = 14  # Increased cell value font size
X_AXIS_FONT_SIZE = 12  # X-axis tick label font size
OUTPUT_FORMAT = "pdf"  # Options: "png", "pdf"


def load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load queries and BNs from parquet files."""
    queries_path = repo_root / "data" / "queries.parquet"
    bns_path = repo_root / "data" / "bns.parquet"

    print(f"Loading queries from {queries_path}...")
    queries_df = pd.read_parquet(queries_path)
    print(f"✓ Loaded {len(queries_df)} queries")

    print(f"Loading BNs from {bns_path}...")
    bns_df = pd.read_parquet(bns_path)
    print(f"✓ Loaded {len(bns_df)} BNs\n")

    return queries_df, bns_df


def calculate_query_count_matrix(
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate query count matrix with network sizes as rows and treewidths as columns.

    Returns:
        DataFrame with n as index, treewidth as columns, query count as values
    """
    # Create bn_uuid -> (n, achieved_tw) mapping
    bns_unique = bns_df.drop_duplicates(subset=["bn_uuid"])
    bn_metadata = bns_unique.set_index("bn_uuid")[["n", "achieved_tw"]].to_dict("index")

    # Add n and treewidth columns to queries
    queries_df = queries_df.copy()
    queries_df["n"] = queries_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get("n")
    )
    queries_df["achieved_tw"] = queries_df["bn_uuid"].map(
        lambda x: bn_metadata.get(x, {}).get("achieved_tw")
    )

    # Get all unique network sizes and treewidths
    network_sizes = sorted(queries_df["n"].dropna().unique())
    treewidths = sorted(queries_df["achieved_tw"].dropna().unique())

    print(f"Network sizes: {network_sizes}")
    print(f"Treewidths: {treewidths}\n")

    # Initialize query count matrix (rows = n, columns = treewidth)
    query_count_matrix = pd.DataFrame(
        index=[int(n) for n in network_sizes],
        columns=[int(tw) for tw in treewidths],
        dtype=float,
    )

    # Calculate query count for each (n, treewidth) combination
    for n in network_sizes:
        for tw in treewidths:
            # Filter queries for this (n, treewidth) combination
            subset = queries_df[
                (queries_df["n"] == n) & (queries_df["achieved_tw"] == tw)
            ]

            if len(subset) == 0:
                # No queries for this combination
                query_count_matrix.loc[int(n), int(tw)] = np.nan
                continue

            # Count unique queries for this combination
            query_count = subset["query_uuid"].nunique()
            query_count_matrix.loc[int(n), int(tw)] = query_count

    return query_count_matrix


def create_heatmap(query_count_matrix: pd.DataFrame) -> plt.Figure:
    """Create a heatmap showing query count."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create grayscale colormap: white for 0/N/A, darker gray for higher counts
    colors = ["#FFFFFF", "#E0E0E0", "#BDBDBD", "#757575", "#424242"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("grayscale", colors, N=n_bins)

    # Get max value for color scaling
    max_count = query_count_matrix.max().max()
    if pd.isna(max_count) or max_count == 0:
        max_count = 1

    # Prepare annotation matrix
    annot_matrix = query_count_matrix.copy().astype(object)
    na_cells = []  # (row, col) positions for N/A cells

    for i, row_idx in enumerate(query_count_matrix.index):
        for j, col_idx in enumerate(query_count_matrix.columns):
            val = query_count_matrix.loc[row_idx, col_idx]
            if pd.isna(val):
                annot_matrix.loc[row_idx, col_idx] = "-"
                na_cells.append((i, j))
            else:
                annot_matrix.loc[row_idx, col_idx] = f"{int(val)}"

    # Fill NaN with 0 for visualization
    plot_matrix = query_count_matrix.fillna(0)

    # Create heatmap (no colorbar since values are shown in cells)
    sns.heatmap(
        plot_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=max_count,
        cbar=False,  # No colorbar
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": ANNOT_FONT_SIZE, "color": "black"},
    )

    # Overlay white cells for N/A (no data)
    for i, j in na_cells:
        ax.add_patch(
            plt.Rectangle(
                (j, i),
                1,
                1,
                fill=True,
                facecolor="white",
                edgecolor="gray",
                linewidth=0.5,
                zorder=10,
            )
        )

    # Set labels with padding to separate from tick labels
    ax.set_xlabel("Achieved Treewidth", fontsize=14, fontweight="bold", labelpad=15)
    ax.set_ylabel("Network Size (n)", fontsize=14, fontweight="bold", labelpad=15)
    ax.set_title(
        "Number of Queries per Network Size and Treewidth",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Ensure y-axis labels are integers and set font sizes
    ax.set_yticklabels(
        [int(x) for x in query_count_matrix.index], rotation=0, fontsize=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=X_AXIS_FONT_SIZE)

    plt.tight_layout()

    return fig


def save_plot(fig: plt.Figure, plots_dir: Path) -> Path:
    """Save the heatmap to the plots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"query_count_heatmap_{timestamp}.{OUTPUT_FORMAT}"
    plot_path = plots_dir / filename

    if OUTPUT_FORMAT.lower() == "pdf":
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Heatmap saved to: {plot_path}")

    return plot_path


def main():
    """Main function to generate query count heatmap."""
    print("=" * 70)
    print("Generating Query Count Heatmap")
    print("=" * 70)
    print()

    # Get repo root
    repo_root = Path(__file__).resolve().parents[2]

    # Create plots directory
    plots_dir = repo_root / "plots" / "query_count_heatmap"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    queries_df, bns_df = load_data(repo_root)

    # Calculate query count matrix
    print("Calculating query count matrix...")
    query_count_matrix = calculate_query_count_matrix(queries_df, bns_df)

    if query_count_matrix.empty or query_count_matrix.isna().all().all():
        print("✗ No valid query count data. Exiting.")
        return

    print(f"Matrix shape: {query_count_matrix.shape}\n")

    # Create heatmap
    print("Creating heatmap...")
    fig = create_heatmap(query_count_matrix)

    # Save plot
    save_plot(fig, plots_dir)

    plt.close(fig)

    print("\n" + "=" * 70)
    print("✓ Query count heatmap generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
