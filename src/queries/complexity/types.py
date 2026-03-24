"""
Data types for query complexity analysis.

This module defines the data structures returned by complexity analysis functions.
"""

from dataclasses import dataclass


@dataclass
class ComplexityMetrics:
    """
    Comprehensive complexity metrics for a probabilistic query.

    This dataclass encapsulates all metrics computed during query complexity analysis,
    including network reduction statistics, elimination order information, and
    computational cost estimates.

    Attributes:
        original_num_vars: Number of variables in the original network
        reduced_num_vars: Number of variables after reduction
        num_independent_vars: Number of conditionally independent variables removed
        num_barren_vars: Number of barren variables removed
        num_vars_removed: Total number of variables removed (independent + barren)
        num_edges: Number of edges in the reduced network
        num_target_vars: Number of target/query variables
        num_evidence_vars: Number of evidence variables
        num_eliminated_vars: Number of variables eliminated during inference
        elimination_order: Order in which variables are eliminated
        complete_elimination_order: Full elimination order including target variables
        induced_width: Maximum clique size in the elimination graph minus 1
        total_cost: Sum of all intermediate factor sizes during elimination
        max_factor_size: Largest intermediate factor encountered
        avg_factor_size: Average size of factors during elimination
        factor_sizes: List of factor sizes for each elimination step
        log_total_cost: Log2 of total cost
        log_max_factor_size: Log2 of max factor size
        keep_vars: Sorted list of variables kept (targets)
        eliminate_vars: Sorted list of variables to eliminate

    Example:
        >>> metrics = ComplexityMetrics(
        ...     original_num_vars=10,
        ...     reduced_num_vars=5,
        ...     num_independent_vars=3,
        ...     num_barren_vars=2,
        ...     # ... other fields
        ... )
        >>> print(f"Complexity reduction: {metrics.num_vars_removed} vars removed")
        >>> print(f"Induced width: {metrics.induced_width}")
    """

    # Network reduction statistics
    original_num_vars: int
    reduced_num_vars: int
    num_independent_vars: int
    num_barren_vars: int
    num_vars_removed: int
    num_edges: int

    # Query structure
    num_target_vars: int
    num_evidence_vars: int
    num_eliminated_vars: int

    # Elimination order
    elimination_order: list[str]
    complete_elimination_order: list[str]

    # Complexity metrics
    induced_width: int
    total_cost: int
    max_factor_size: int
    avg_factor_size: float
    factor_sizes: list[int]
    log_total_cost: float
    log_max_factor_size: float

    # Variable sets
    keep_vars: list[str]
    eliminate_vars: list[str]

    def __repr__(self) -> str:
        """Provide a concise summary of complexity metrics."""
        return (
            f"ComplexityMetrics("
            f"vars={self.original_num_vars}→{self.reduced_num_vars}, "
            f"removed={self.num_vars_removed}, "
            f"induced_width={self.induced_width}, "
            f"total_cost={self.total_cost:,}, "
            f"max_factor={self.max_factor_size:,})"
        )
