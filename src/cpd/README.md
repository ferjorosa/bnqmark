# CPD Utilities

Utilities for formatting Conditional Probability Distributions (CPDs) from Bayesian Networks.

## Module

- **`formatting`** – Format CPDs as ASCII tables for display and debugging.

## Main Function

- `cpd_to_ascii_table(cpd)` – Convert a `TabularCPD` to a formatted ASCII table string.

## Features

- Handles CPDs with or without parent nodes.
- Automatically formats tables with proper column widths.
- Displays probabilities with 4 decimal precision.
- Works with any number of parent nodes and state cardinalities.
