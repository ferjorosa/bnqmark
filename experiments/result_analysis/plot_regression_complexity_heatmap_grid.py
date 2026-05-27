#!/usr/bin/env python3
"""Generate model heatmaps over regression-selected complexity axes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

MODEL_DISPLAY = {
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "deepseek/deepseek-v3.2-speciale": "DeepSeek V3.2-Speciale",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "z-ai/glm-5": "GLM-5",
    "openai/gpt-5.4": "GPT-5.4",
    "x-ai/grok-4.20": "Grok 4.20",
    "moonshotai/kimi-k2.5": "Kimi K2.5",
    "minimax/minimax-m2.7": "MiniMax M2.7",
    "qwen/qwen3-max-thinking": "Qwen3-Max-Thinking",
}

MODEL_ORDER = sorted(MODEL_DISPLAY, key=lambda model: MODEL_DISPLAY[model].lower())

PROTOCOL_LABELS = {
    "raw_reasoning": "Raw Reasoning",
    "code_generation": "Code Generation",
}

METRIC_LABELS = {
    "answerability": "Answerability",
    "accuracy": "Accuracy",
    "mae": "MAE",
}

FACTOR_SIZE_BINS = [0, 16, 32, 64, 128, 256, 512, 1024, 4096, 10**12]
FACTOR_SIZE_LABELS = [
    "<=16",
    "17-32",
    "33-64",
    "65-128",
    "129-256",
    "257-512",
    "513-1k",
    "1k-4k",
    ">4k",
]

FIGURE_SIZE = (16, 16)
ANNOT_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 10
AXIS_TICK_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
ACCURACY_THRESHOLD = 0.01
TREEWIDTH_COLUMN = "target_tw"
TREEWIDTH_LABEL = "Treewidth"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "data" / "hf_bnqmark_20" / "regression"
    default_output_dir = repo_root / "plots" / "regression_complexity_heatmap"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=(
            "Directory containing experiments.parquet, queries.parquet, "
            "and bns.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where PDF heatmaps will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["answerability", "accuracy", "mae", "all"],
        default="all",
    )
    parser.add_argument(
        "--protocol",
        choices=["raw_reasoning", "code_generation", "all"],
        default="all",
    )
    parser.add_argument("--naming-strategy", default="simple")
    parser.add_argument(
        "--treewidth-column",
        choices=["target_tw", "achieved_tw"],
        default=TREEWIDTH_COLUMN,
        help="BN metadata column used for the treewidth axis.",
    )
    return parser.parse_args()


def load_data(data_dir: Path, treewidth_column: str) -> pd.DataFrame:
    """Load and merge experiment, query, and BN metadata."""
    experiments = pd.read_parquet(
        data_dir / "experiments.parquet",
        columns=[
            "query_uuid",
            "naming_strategy",
            "experiment_type",
            "model_name",
            "llm_probability",
        ],
    )
    queries = pd.read_parquet(
        data_dir / "queries.parquet",
        columns=[
            "query_uuid",
            "bn_uuid",
            "probability",
            "total_factor_size",
        ],
    )
    bns = pd.read_parquet(
        data_dir / "bns.parquet",
        columns=["bn_uuid", treewidth_column],
    ).drop_duplicates(subset=["bn_uuid"])
    bns = bns.rename(columns={treewidth_column: "treewidth"})

    merged = experiments.merge(queries, on="query_uuid", how="inner")
    merged = merged.merge(bns, on="bn_uuid", how="inner")
    merged["factor_size_bin"] = pd.cut(
        merged["total_factor_size"],
        bins=FACTOR_SIZE_BINS,
        labels=FACTOR_SIZE_LABELS,
        include_lowest=True,
    )
    return merged


def calculate_matrix(model_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Calculate one metric matrix for a single model."""
    treewidths = sorted(model_df["treewidth"].dropna().unique())
    matrix = pd.DataFrame(
        index=pd.CategoricalIndex(
            FACTOR_SIZE_LABELS,
            categories=FACTOR_SIZE_LABELS,
            ordered=True,
            name="factor_size_bin",
        ),
        columns=[int(tw) for tw in treewidths],
        dtype=float,
    )

    for factor_bin in FACTOR_SIZE_LABELS:
        for tw in treewidths:
            subset = model_df[
                (model_df["factor_size_bin"] == factor_bin)
                & (model_df["treewidth"] == tw)
            ]
            if subset.empty:
                matrix.loc[factor_bin, int(tw)] = -1
                continue

            non_null = subset["llm_probability"][subset["llm_probability"].notna()]
            if len(non_null) > 0 and (non_null == -1000).all():
                matrix.loc[factor_bin, int(tw)] = np.nan
                continue

            valid = subset[
                subset["llm_probability"].notna() & (subset["llm_probability"] != -1000)
            ].copy()

            if metric == "answerability":
                total_queries = subset["query_uuid"].nunique()
                answered_queries = valid["query_uuid"].nunique()
                matrix.loc[factor_bin, int(tw)] = answered_queries / total_queries
            elif metric == "accuracy":
                total_queries = subset["query_uuid"].nunique()
                valid["abs_error"] = (
                    valid["llm_probability"] - valid["probability"]
                ).abs()
                min_error = valid.groupby("query_uuid")["abs_error"].min()
                accurate_queries = (min_error <= ACCURACY_THRESHOLD).sum()
                matrix.loc[factor_bin, int(tw)] = accurate_queries / total_queries
            elif metric == "mae":
                if valid.empty:
                    matrix.loc[factor_bin, int(tw)] = -2
                    continue
                valid["abs_error"] = (
                    valid["llm_probability"] - valid["probability"]
                ).abs()
                matrix.loc[factor_bin, int(tw)] = valid["abs_error"].mean()
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return matrix


def metric_colormap(metric: str) -> tuple[LinearSegmentedColormap, float, float]:
    """Return a colormap and plotting range for the metric."""
    if metric == "answerability":
        colors = ["#FFFFFF", "#BBDEFB", "#64B5F6", "#2196F3", "#0D47A1"]
        return LinearSegmentedColormap.from_list("answerability", colors), 0, 1
    if metric == "accuracy":
        colors = ["#FFFFFF", "#FFE5B4", "#FFA500", "#FF4500", "#8B0000"]
        return LinearSegmentedColormap.from_list("accuracy", colors), 0, 1
    if metric == "mae":
        colors = ["#FFFFFF", "#C8E6C9", "#81C784", "#4CAF50", "#1B5E20"]
        return LinearSegmentedColormap.from_list("mae", colors), 0, 1
    raise ValueError(f"Unknown metric: {metric}")


def create_subplot(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    model_name: str,
    metric: str,
) -> None:
    """Draw a single model heatmap."""
    cmap, vmin, vmax = metric_colormap(metric)

    annot = matrix.copy().astype(object)
    black_na_cells = []
    red_na_cells = []
    dash_cells = []

    for i, row_idx in enumerate(matrix.index):
        for j, col_idx in enumerate(matrix.columns):
            val = matrix.loc[row_idx, col_idx]
            if pd.isna(val):
                annot.loc[row_idx, col_idx] = "N/A"
                black_na_cells.append((i, j))
            elif val == -2:
                annot.loc[row_idx, col_idx] = "N/A"
                red_na_cells.append((i, j))
            elif val == -1:
                annot.loc[row_idx, col_idx] = "--"
                dash_cells.append((i, j))
            else:
                annot.loc[row_idx, col_idx] = f"{val:.2f}"

    plot_matrix = matrix.copy().fillna(0)
    for i, j in dash_cells + red_na_cells:
        plot_matrix.iloc[i, j] = 0

    sns.heatmap(
        plot_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": ANNOT_FONT_SIZE, "color": "black"},
    )

    for cells, color in ((black_na_cells, "black"), (red_na_cells, "red")):
        for i, j in cells:
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=True,
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=10,
                )
            )
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

    ax.set_xlabel(
        TREEWIDTH_LABEL,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Total factor size",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_title(
        MODEL_DISPLAY.get(model_name, model_name),
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=AXIS_TICK_FONT_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=AXIS_TICK_FONT_SIZE)


def create_grid(
    matrices: dict[str, pd.DataFrame],
    metric: str,
    protocol: str,
) -> plt.Figure:
    """Create a 3x3 grid of model heatmaps."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(3, 3, hspace=0.32, wspace=0.35)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

    models = list(matrices)
    for index, model_name in enumerate(models):
        create_subplot(axes[index], matrices[model_name], model_name, metric)

    for index in range(len(models), 9):
        axes[index].set_visible(False)

    cmap, vmin, vmax = metric_colormap(metric)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    title = f"{METRIC_LABELS[metric]} - {PROTOCOL_LABELS[protocol]}"
    if metric == "accuracy":
        title = f"{title} (threshold = {ACCURACY_THRESHOLD})"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    return fig


def generate_one(
    data: pd.DataFrame,
    metric: str,
    protocol: str,
    naming_strategy: str,
    output_dir: Path,
) -> Path:
    """Generate one metric/protocol figure."""
    filtered = data[
        (data["naming_strategy"] == naming_strategy)
        & (data["experiment_type"] == protocol)
    ].copy()

    matrices = {}
    for model_name in MODEL_ORDER:
        model_df = filtered[filtered["model_name"] == model_name]
        if model_df.empty:
            continue
        matrices[model_name] = calculate_matrix(model_df, metric)

    fig = create_grid(matrices, metric, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric}_heatmap_grid_{naming_strategy}_{protocol}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    """Generate requested heatmaps."""
    args = parse_args()
    metrics = (
        ["answerability", "accuracy", "mae"] if args.metric == "all" else [args.metric]
    )
    protocols = (
        ["raw_reasoning", "code_generation"]
        if args.protocol == "all"
        else [args.protocol]
    )

    print(f"Treewidth column: {args.treewidth_column}")
    data = load_data(args.data_dir, args.treewidth_column)
    for metric in metrics:
        for protocol in protocols:
            generate_one(data, metric, protocol, args.naming_strategy, args.output_dir)


if __name__ == "__main__":
    main()
