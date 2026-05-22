#!/usr/bin/env python3
"""Plot query-by-model manual arithmetic operation counts."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.result_analysis.code_generation_arithmetic_summary import (
    add_arithmetic_metrics,
)

NAMING_STRATEGY = "simple"
EXPERIMENT_TYPE = "code_generation"
ACCURACY_THRESHOLD = 0.01

MODEL_LABELS = {
    "anthropic/claude-sonnet-4.6": "Claude\nSonnet 4.6",
    "deepseek/deepseek-v3.2-speciale": "DeepSeek\nV3.2",
    "google/gemini-3.1-pro-preview": "Gemini\n3.1 Pro",
    "minimax/minimax-m2.7": "MiniMax\nM2.7",
    "moonshotai/kimi-k2.5": "Kimi\nK2.5",
    "openai/gpt-5.4": "GPT-5.4",
    "qwen/qwen3-max-thinking": "Qwen3 Max\nThinking",
    "x-ai/grok-4.20": "Grok\n4.20",
    "z-ai/glm-5": "GLM-5",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        type=Path,
        default=repo_root / "data" / "experiments.parquet",
        help="Path to experiments parquet.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=repo_root / "data" / "queries.parquet",
        help="Path to queries parquet.",
    )
    parser.add_argument(
        "--bns",
        type=Path,
        default=repo_root / "data" / "bns.parquet",
        help="Path to Bayesian network metadata parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "plots" / "result_analysis",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=ACCURACY_THRESHOLD,
        help="Absolute-error threshold used for correctness markers.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Percentile used to cap heatmap color scale.",
    )
    return parser.parse_args()


def load_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load experiment, query, and BN metadata."""
    experiments_df = pd.read_parquet(args.experiments)
    queries_df = pd.read_parquet(args.queries)
    bns_df = pd.read_parquet(args.bns)
    return experiments_df, queries_df, bns_df


def build_plot_tables(
    experiments_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
    accuracy_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build operation, correctness, and sorted query metadata tables."""
    code_df = experiments_df[
        (experiments_df["naming_strategy"] == NAMING_STRATEGY)
        & (experiments_df["experiment_type"] == EXPERIMENT_TYPE)
    ].copy()
    metrics_df = add_arithmetic_metrics(code_df)

    query_cols = [
        "query_uuid",
        "bn_uuid",
        "target",
        "evidence",
        "probability",
        "induced_width",
        "num_eliminated",
        "num_target_nodes",
        "num_evidence_nodes",
    ]
    query_df = queries_df[queries_df["naming_strategy"] == NAMING_STRATEGY][
        query_cols
    ].copy()

    bn_cols = ["bn_uuid", "n", "achieved_tw", "num_edges", "avg_markov_blanket_size"]
    bn_df = (
        bns_df[bns_df["naming_strategy"] == NAMING_STRATEGY][bn_cols]
        .drop_duplicates("bn_uuid")
        .copy()
    )
    query_df = query_df.merge(bn_df, on="bn_uuid", how="left")

    metrics_df = metrics_df.merge(
        query_df[["query_uuid", "probability"]],
        on="query_uuid",
        how="left",
    )
    metrics_df["is_manual_parseable"] = (
        metrics_df["code_style"] == "manual"
    ) & metrics_df["code_parse_error"].isna()
    metrics_df["abs_error"] = (
        metrics_df["llm_probability"] - metrics_df["probability"]
    ).abs()
    metrics_df["is_correct"] = metrics_df["abs_error"] <= accuracy_threshold
    metrics_df["operation_count"] = metrics_df["arithmetic_operator_count"].where(
        metrics_df["is_manual_parseable"]
    )
    metrics_df["correctness_marker"] = metrics_df["is_correct"].where(
        metrics_df["is_manual_parseable"]
    )

    model_order = [
        model for model in MODEL_LABELS if model in set(metrics_df["model_name"])
    ]
    remaining_models = sorted(set(metrics_df["model_name"]) - set(model_order))
    model_order.extend(remaining_models)

    ops = metrics_df.pivot_table(
        index="query_uuid",
        columns="model_name",
        values="operation_count",
        aggfunc="first",
    ).reindex(columns=model_order)
    correctness = metrics_df.pivot_table(
        index="query_uuid",
        columns="model_name",
        values="correctness_marker",
        aggfunc="first",
    ).reindex(columns=model_order)

    query_df = query_df.set_index("query_uuid")
    query_df["mean_manual_ops"] = ops.mean(axis=1, skipna=True)
    query_df["manual_model_count"] = ops.notna().sum(axis=1)
    query_df = query_df.sort_values(
        [
            "n",
            "achieved_tw",
            "induced_width",
            "num_eliminated",
            "num_target_nodes",
            "num_evidence_nodes",
            "mean_manual_ops",
        ],
        ascending=[True, True, True, True, True, True, True],
        na_position="last",
    )

    sorted_queries = query_df.index.tolist()
    return (
        ops.loc[sorted_queries],
        correctness.loc[sorted_queries],
        query_df,
        model_order,
    )


def plot_heatmap(
    ops: pd.DataFrame,
    correctness: pd.DataFrame,
    query_df: pd.DataFrame,
    model_order: list[str],
    output_dir: Path,
    accuracy_threshold: float,
    vmax_percentile: float,
) -> tuple[Path, Path]:
    """Create and save the operation-count heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)

    values = ops.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite_values, vmax_percentile))
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(13, 20))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E6E6E6")
    masked_values = np.ma.masked_invalid(values)
    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
    )

    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(
        [MODEL_LABELS.get(model, model) for model in model_order],
        fontsize=9,
        rotation=0,
    )

    n_groups = query_df.reset_index().groupby("n", sort=False).size()
    starts = n_groups.cumsum().shift(fill_value=0)
    middles = starts + (n_groups / 2) - 0.5
    ax.set_yticks(middles.to_numpy())
    ax.set_yticklabels([f"n={int(n)}" for n in n_groups.index], fontsize=9)
    ax.set_ylabel("Queries sorted by network size, treewidth, and query structure")
    ax.set_xlabel("Model")

    for boundary in starts.iloc[1:]:
        ax.axhline(boundary - 0.5, color="#222222", linewidth=0.6)

    tw_groups = query_df.reset_index().groupby(["n", "achieved_tw"], sort=False).size()
    tw_starts = tw_groups.cumsum().shift(fill_value=0)
    for boundary in tw_starts.iloc[1:]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.25, alpha=0.6)

    correct_mask = correctness.to_numpy(dtype=object) == True  # noqa: E712
    incorrect_mask = correctness.to_numpy(dtype=object) == False  # noqa: E712
    correct_y, correct_x = np.where(correct_mask)
    incorrect_y, incorrect_x = np.where(incorrect_mask)

    ax.scatter(
        correct_x,
        correct_y,
        s=5,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=0.25,
        label=f"Correct (abs. error <= {accuracy_threshold:g})",
    )
    ax.scatter(
        incorrect_x,
        incorrect_y,
        s=8,
        marker="x",
        color="#D62728",
        linewidths=0.45,
        label="Incorrect",
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(
        f"Arithmetic operations in parseable manual code "
        f"(color capped at p{vmax_percentile:g}={vmax:.0f})"
    )

    legend_handles = [
        Patch(facecolor="#E6E6E6", edgecolor="none", label="Not manual / parse failed"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markersize=5,
            linewidth=0,
            label=f"Correct (abs. error <= {accuracy_threshold:g})",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#D62728",
            markersize=5,
            linewidth=0,
            label="Incorrect",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Manual Arithmetic Operations by Query and Model",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.968,
        "Rows are grouped by network size n, then achieved treewidth, induced width, "
        "eliminated variables, target/evidence counts, and mean operation count.",
        fontsize=9,
        ha="center",
        va="top",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    png_path = output_dir / "manual_ops_heatmap_by_query.png"
    pdf_path = output_dir / "manual_ops_heatmap_by_query.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    """Generate the manual operation-count heatmap."""
    args = parse_args()
    experiments_df, queries_df, bns_df = load_inputs(args)
    ops, correctness, query_df, model_order = build_plot_tables(
        experiments_df,
        queries_df,
        bns_df,
        args.accuracy_threshold,
    )
    png_path, pdf_path = plot_heatmap(
        ops,
        correctness,
        query_df,
        model_order,
        args.output_dir,
        args.accuracy_threshold,
        args.vmax_percentile,
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
