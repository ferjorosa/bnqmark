#!/usr/bin/env python3
"""Plot pooled accuracy AND answerability on shared panels (4 lines total).

Single-row, two-panel figure: columns are the complexity axes (treewidth,
total factor size). Each panel draws four lines --- {accuracy, answerability} x
{raw reasoning, code generation} --- pooled over all supported (query, model)
rows within each bin, with Wilson 95% confidence intervals.

Metric is encoded by line style/marker (accuracy = solid circles,
answerability = dashed triangles); protocol is encoded by color (raw
reasoning = red, code generation = blue). This is a single-row, space-saving
version that stacks the two metrics on shared panels instead of using two rows.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]

ACCURACY_THRESHOLD = 0.01
PROTOCOLS = ["raw_reasoning", "code_generation"]
PROTOCOL_LABELS = {
    "raw_reasoning": "Raw reasoning",
    "code_generation": "Code generation",
}
PROTOCOL_COLORS = {"raw_reasoning": "#C44E52", "code_generation": "#4C72B0"}

METRICS = ["accuracy", "answerability"]
METRIC_STYLES = {
    "accuracy": {"linestyle": "-", "marker": "o", "label": "Accuracy"},
    "answerability": {"linestyle": "--", "marker": "^", "label": "Answerability"},
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

FIGURE_SIZE = (7.32, 2.5)
TITLE_FONT_SIZE = 8
LABEL_FONT_SIZE = 7
TICK_FONT_SIZE = 6
LEGEND_FONT_SIZE = 8
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
JPEG_DPI = 200


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory containing experiments.parquet, queries.parquet, bns.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "plots"
        / "result_analysis"
        / "complexity_performance_correlation.pdf",
        help="Output figure path. The suffix controls the Matplotlib format.",
    )
    parser.add_argument(
        "--naming-strategy",
        default="simple",
        help="Naming strategy used in the experiment rows.",
    )
    parser.add_argument(
        "--treewidth-column",
        choices=["target_tw", "achieved_tw"],
        default="target_tw",
        help="BN metadata column used for the treewidth axis.",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=ACCURACY_THRESHOLD,
        help="Absolute-error threshold for accuracy.",
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
        columns=["query_uuid", "bn_uuid", "probability", "total_factor_size"],
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


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half_width = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def binned_metrics(
    df: pd.DataFrame,
    bin_column: str,
    bins: list,
    accuracy_threshold: float,
) -> pd.DataFrame:
    """Aggregate answerability and accuracy within each bin, pooled over models."""
    rows = []
    for bin_value in bins:
        subset = df[df[bin_column] == bin_value]
        total = len(subset)
        if total == 0:
            rows.append({"bin": bin_value, "n": 0})
            continue
        answerable = subset["llm_probability"].notna()
        absolute_error = (subset["llm_probability"] - subset["probability"]).abs()
        correct = answerable & (absolute_error <= accuracy_threshold)
        answerability = float(answerable.mean())
        accuracy = float(correct.mean())
        ans_low, ans_high = wilson_ci(int(answerable.sum()), total)
        acc_low, acc_high = wilson_ci(int(correct.sum()), total)
        rows.append(
            {
                "bin": bin_value,
                "n": total,
                "answerability": answerability,
                "answerability_low": ans_low,
                "answerability_high": ans_high,
                "accuracy": accuracy,
                "accuracy_low": acc_low,
                "accuracy_high": acc_high,
            }
        )
    return pd.DataFrame(rows)


def plot_panel(
    ax: plt.Axes,
    summary: dict[tuple[str, str], pd.DataFrame],
    bins: list,
    *,
    categorical: bool,
) -> None:
    """Draw one panel: 4 lines (metric x protocol) vs the complexity axis."""
    positions: dict = {}
    if categorical:
        positions = {bin_value: i for i, bin_value in enumerate(bins)}
        x_axis_values = list(range(len(bins)))
        x_axis_labels = list(bins)
    else:
        for bin_value in bins:
            try:
                positions[bin_value] = float(bin_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Non-numeric bin {bin_value!r} requires categorical=True."
                )
        x_axis_values = [float(b) for b in bins]
        x_axis_labels = x_axis_values
    x_positions = np.array([positions[b] for b in bins], dtype=float)

    for metric in METRICS:
        style = METRIC_STYLES[metric]
        for protocol in PROTOCOLS:
            df = summary[(metric, protocol)]
            x = df["bin"].map(positions).to_numpy(dtype=float)
            y = df[metric].to_numpy(dtype=float)
            low = df[f"{metric}_low"].to_numpy(dtype=float)
            high = df[f"{metric}_high"].to_numpy(dtype=float)
            valid = df["n"].to_numpy() > 0
            _, _, bars = ax.errorbar(
                x[valid],
                y[valid],
                yerr=[y[valid] - low[valid], high[valid] - y[valid]],
                marker=style["marker"],
                markersize=3,
                linestyle=style["linestyle"],
                linewidth=1.2,
                capsize=2,
                color=PROTOCOL_COLORS[protocol],
                label=f"{style['label']} ({PROTOCOL_LABELS[protocol]})",
            )
            for bar in bars:
                bar.set_alpha(0.3)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    if categorical:
        ax.set_xticks(x_axis_values)
        ax.set_xticklabels(x_axis_labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(bins) - 0.5)
    else:
        ax.set_xticks(x_axis_values)
        ax.set_xticklabels([f"{v:g}" for v in x_axis_labels])
        ax.set_xlim(min(x_axis_values) - 1, max(x_axis_values) + 1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(Y_TICKS)
    ax.tick_params(labelsize=TICK_FONT_SIZE)


def main() -> None:
    """Generate the combined complexity-performance figure."""
    args = parse_args()
    data = load_data(args.data_dir, args.treewidth_column)
    data = data[data["naming_strategy"] == args.naming_strategy].copy()
    if data.empty:
        raise ValueError(f"No rows for naming_strategy={args.naming_strategy!r}")

    supported = data[data["llm_probability"] != -1000]
    treewidths = sorted(supported["treewidth"].dropna().unique())

    summaries: dict[tuple[str, str, str], pd.DataFrame] = {}
    for metric in METRICS:
        for protocol in PROTOCOLS:
            protocol_df = supported[supported["experiment_type"] == protocol]
            summaries[(metric, protocol, "treewidth")] = binned_metrics(
                protocol_df, "treewidth", treewidths, args.accuracy_threshold
            )
            summaries[(metric, protocol, "factor")] = binned_metrics(
                protocol_df,
                "factor_size_bin",
                FACTOR_SIZE_LABELS,
                args.accuracy_threshold,
            )

    fig, axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, sharey=True, sharex=False
    )
    fig.subplots_adjust(wspace=0.25)

    panels = [
        ("treewidth", "Treewidth", False),
        ("factor", "Total factor size", True),
    ]
    for (axis, xlabel, categorical), ax in zip(panels, axes, strict=True):
        bins = treewidths if axis == "treewidth" else FACTOR_SIZE_LABELS
        summary = {
            (metric, protocol): summaries[(metric, protocol, axis)]
            for metric in METRICS
            for protocol in PROTOCOLS
        }
        plot_panel(ax, summary, bins, categorical=categorical)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE, fontweight="bold")

    axes[0].set_ylabel("Proportion", fontsize=LABEL_FONT_SIZE, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    jpeg_path = args.output.with_suffix(".jpg")
    fig.savefig(jpeg_path, dpi=JPEG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Wrote {jpeg_path}")


if __name__ == "__main__":
    main()
