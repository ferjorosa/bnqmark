"""Bayesian Network utility functions."""

from __future__ import annotations

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def get_cpds_list(bn: DiscreteBayesianNetwork) -> list[TabularCPD]:
    """
    Get all CPDs from a Bayesian Network as a list.

    pgmpy's `get_cpds()` method returns `TabularCPD | list[TabularCPD]` - it returns
    a single CPD when there's only one, or a list when there are multiple. This
    function normalizes the return value to always be a list, making it easier
    to iterate over CPDs without type checker errors.

    Args:
        bn: A DiscreteBayesianNetwork instance.

    Returns:
        A list of TabularCPD objects.

    Raises:
        ValueError: If the model has no CPDs.

    Example:
        >>> cpds = get_cpds_list(bn)
        >>> for cpd in cpds:
        ...     print(cpd.variable)
    """
    cpds = bn.get_cpds()
    if cpds is None:
        raise ValueError("Bayesian network has no CPDs.")
    if isinstance(cpds, TabularCPD):
        return [cpds]
    return list(cpds)


__all__ = ["get_cpds_list"]
