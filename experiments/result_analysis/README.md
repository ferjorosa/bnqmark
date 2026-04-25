# Result Analysis

Generate publication-quality plots and summary tables from the experiment results.

These scripts visualize LLM performance across different network sizes, treewidths, and model configurations, producing figures for the paper. All scripts read from the database and output to `plots/`.

## Scripts

### `plot_accuracy_heatmap_grid.py`

Generate a grid of accuracy heatmaps showing all models side-by-side.

**Key functionality:**
- Creates multi-panel figure with one subplot per model
- X-axis: achieved treewidth, Y-axis: network size
- Color intensity represents accuracy (fraction within threshold)
- Marks N/A cells where context limits exceeded
- Primary figure for the paper results section

**Main function:**
- `create_grid_figure(...)` — Create multi-model heatmap grid

### `plot_answerability_heatmap_grid.py`

Generate answerability heatmaps showing valid response rates.

**Key functionality:**
- Similar grid layout to accuracy heatmaps
- Shows fraction of queries that received parseable answers
- Distinguishes between context-limit failures vs. format errors

### `plot_mae_heatmap_grid.py`

Generate Mean Absolute Error heatmaps.

**Key functionality:**
- Shows `|p_model - p_true|` averaged over valid responses
- Reveals systematic over/under-estimation patterns
- Complements accuracy plots with continuous error metric

### `plot_query_count_heatmap.py`

Generate query distribution heatmap.

**Key functionality:**
- Shows number of benchmark queries per (treewidth, network size) cell
- Validates benchmark coverage and balance
- Used in the dataset description section

### `model_summary_table.py`

Generate summary statistics tables.

**Key functionality:**
- Aggregates accuracy, answerability, MAE per model
- Produces LaTeX-ready tables for the paper
- Supports both raw reasoning and code generation breakdowns

### `old/` — Deprecated Scripts

Previous plotting scripts replaced by the grid-based visualizations:
- `plot_accuracy_heatmap.py` — Single heatmap per model
- `plot_accuracy_by_treewidth.py` — Bar charts by treewidth
- `analyze_experiments.py` — Early analysis script

## Output

All plots are saved to `plots/` directory in PDF and PNG formats.
