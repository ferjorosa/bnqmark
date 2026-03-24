"""Formatting utilities for Conditional Probability Distributions (CPDs)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import cast

from pgmpy.factors.discrete import TabularCPD


def _format_table(rows: list[list[str]]) -> str:
    widths: list[int] = []
    for row in rows:
        for i, cell in enumerate(row):
            if i >= len(widths):
                widths.append(len(cell))
            else:
                widths[i] = max(widths[i], len(cell))

    def horiz() -> str:
        parts = ["+" + "-" * (w + 2) for w in widths]
        return "".join(parts) + "+"

    def fmt_row(row: list[str]) -> str:
        cells = [f" {cell.ljust(w)} " for cell, w in zip(row, widths, strict=False)]
        return "|" + "|".join(cells) + "|"

    out: list[str] = []
    out.append(horiz())
    for r in rows:
        out.append(fmt_row(r))
        out.append(horiz())
    return "\n".join(out)


def _parent_assignments(
    parents: Sequence[str],
    state_names: Mapping[str, Sequence[str]],
) -> list[tuple[str, ...]]:
    domains = [state_names[p] for p in parents]
    return list(product(*domains)) if parents else []


def cpd_to_ascii_table(cpd: TabularCPD) -> str:
    """
    Convert a TabularCPD to an ASCII table representation.

    Args:
        cpd: The conditional probability distribution to format.

    Returns:
        A formatted ASCII table string representing the CPD.
    """
    var = cpd.variable

    # Narrow type for state_names (pgmpy is not well typed here)
    state_names = cast(Mapping[str, Sequence[str]], cpd.state_names)

    var_states = list(state_names[var])

    parents = (
        list(cpd.variables[1:])
        if hasattr(cpd, "variables")
        else list(cpd.evidence or [])
    )

    rows: list[list[str]] = []

    if not parents:
        rows.append(["Node(Value)", "Probability"])  # header
        for s_idx, s in enumerate(var_states):
            prob = float(cpd.values[s_idx])
            rows.append([f"{var}({s})", f"{prob:.4f}"])
        return _format_table(rows)

    parent_assigns = _parent_assignments(parents, state_names)

    # Precompute parent index positions (avoid repeated .index calls)
    parent_pos = {p: i for i, p in enumerate(parents)}

    # Header rows listing parent assignments as columns
    for p in parents:
        header = [p]
        p_idx = parent_pos[p]
        for assign in parent_assigns:
            val = assign[p_idx]
            header.append(f"{p}({val})")
        rows.append(header)

    # Build index maps for parent state -> integer index
    parent_state_to_idx: dict[str, dict[str, int]] = {
        p: {name: idx for idx, name in enumerate(state_names[p])} for p in parents
    }

    # Now child rows: fetch probability using multi-index over parent axes
    for s_idx, s in enumerate(var_states):
        row = [f"{var}({s})"]
        for assign in parent_assigns:
            idx_tuple = tuple(
                parent_state_to_idx[p][assign[parent_pos[p]]] for p in parents
            )
            prob = float(cpd.values[(s_idx,) + idx_tuple])
            row.append(f"{prob:.4f}")
        rows.append(row)

    return _format_table(rows)


__all__ = ["cpd_to_ascii_table"]
