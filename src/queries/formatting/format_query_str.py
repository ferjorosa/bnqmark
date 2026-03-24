"""Formatting utilities for queries."""

from pgmpy.factors.discrete import TabularCPD

from src.cpd.formatting import cpd_to_ascii_table


def format_probability_query(
    target: dict[str, str],
    evidence: dict[str, str] | None = None,
) -> str:
    """Generate formatted query string like P(Rain=true | Cloudy=true)."""
    target_str = ", ".join([f"{k}={v}" for k, v in target.items()])
    if evidence:
        evidence_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
        return f"P({target_str} | {evidence_str})"
    return f"P({target_str})"


def format_discrete_cpds(cpts: list[TabularCPD]) -> str:
    """Format a list of TabularCPD objects into a single string."""
    cpd_strings = []
    for cpd in cpts:
        cpd_strings.append(cpd_to_ascii_table(cpd))
    return "\n\n".join(cpd_strings)
