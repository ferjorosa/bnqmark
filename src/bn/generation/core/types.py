"""
Type definitions for Bayesian Network generation.

This module provides all the dataclasses and type definitions used throughout
the BN generation pipeline, including strategy configurations and metadata structures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ------------------------------
# Strategy Types
# ------------------------------


@dataclass
class ArityStrategy:
    """
    Strategy for determining variable cardinalities (number of states per variable).

    This class defines how the cardinality (number of possible states) is assigned
    to each variable in a Bayesian Network. Two strategies are supported:

    1. **Fixed**: All variables have the same cardinality
    2. **Range**: Each variable's cardinality is randomly sampled from a range

    Examples:
        >>> # Fixed arity: all variables have 3 states
        >>> strategy = ArityStrategy(type="fixed", fixed=3)
        >>> # Range arity: each variable has 2-4 states (inclusive)
        >>> strategy = ArityStrategy(type="range", min=2, max=4)
    """

    type: str = field(
        metadata={
            "description": "Type of arity strategy. Must be either 'fixed' or 'range'. "
            "'fixed' assigns the same cardinality to all variables, "
            "'range' randomly samples cardinalities from [min, max] for each variable.",
        },
    )

    fixed: int | None = field(
        default=None,
        metadata={
            "description": "Fixed cardinality value. Required when type='fixed'. "
            "Must be >= 2. All variables will have this same number of states.",
        },
    )

    min: int | None = field(
        default=None,
        metadata={
            "description": "Minimum cardinality value. Required when type='range'. "
            "Must be >= 2 and <= max. Each variable's cardinality will be "
            "randomly sampled from the range [min, max] (inclusive).",
        },
    )

    max: int | None = field(
        default=None,
        metadata={
            "description": "Maximum cardinality value. Required when type='range'. "
            "Must be >= min. Each variable's cardinality will be "
            "randomly sampled from the range [min, max] (inclusive).",
        },
    )

    def _validate(self) -> None:
        """
        Validate strategy parameters.

        Raises:
            ValueError: If strategy parameters are invalid
        """
        if self.type == "fixed":
            if not self.fixed or self.fixed < 2:
                raise ValueError("fixed arity must be >= 2")
        elif self.type == "range":
            if not self.min or not self.max or self.min < 2 or self.max < self.min:
                raise ValueError("range arity requires 2 <= min <= max")
        else:
            raise ValueError("Unsupported arity strategy; use 'fixed' or 'range'")

    def draw_cardinalities(
        self,
        nodes: Sequence[Any],
        rng: np.random.Generator,
    ) -> dict[Any, int]:
        """
        Draw cardinalities for each node according to the strategy.

        Args:
            nodes: Sequence of node identifiers (e.g., ['V0', 'V1', 'V2'])
            rng: Random number generator for sampling (used when type='range')

        Returns:
            Dictionary mapping each node identifier to its cardinality
            (number of states)

        Raises:
            ValueError: If strategy parameters are invalid

        Examples:
            >>> import numpy as np
            >>> rng = np.random.default_rng(42)
            >>> # Fixed strategy
            >>> strategy = ArityStrategy(type="fixed", fixed=3)
            >>> cards = strategy.draw_cardinalities(["V0", "V1"], rng)
            >>> print(cards)
            {'V0': 3, 'V1': 3}
            >>> # Range strategy
            >>> strategy = ArityStrategy(type="range", min=2, max=4)
            >>> cards = strategy.draw_cardinalities(["V0", "V1"], rng)
            >>> print(cards)  # Random values in [2, 4]
            {'V0': 3, 'V1': 2}
        """
        self._validate()

        if self.type == "fixed":
            assert self.fixed is not None, "fixed must be set when type='fixed'"
            return {n: int(self.fixed) for n in nodes}
        else:  # type == "range"
            assert self.min is not None and self.max is not None, (
                "min and max must be set when type='range'"
            )
            return {n: int(rng.integers(self.min, self.max + 1)) for n in nodes}


# ------------------------------
# Metadata Types
# ------------------------------


@dataclass
class BNGenerationMetadata:
    """
    Metadata returned from BN generation.

    This metadata captures the essential parameters and results from generating
    a Bayesian Network from a DAG, including cardinality assignments and CPT
    generation parameters.
    """

    node_cardinalities: dict[Any, int] = field(
        metadata={
            "description": (
                "Mapping from node identifiers to their cardinality "
                "(number of states). "
                "For example, {'V0': 2, 'V1': 3} means V0 has 2 states "
                "and V1 has 3 states."
            ),
        },
    )

    dirichlet_alpha: float = field(
        metadata={
            "description": "Dirichlet concentration parameter used for CPT sampling. "
            "Values < 1.0 create skewed distributions, 1.0 creates uniform, "
            "and > 1.0 creates flatter distributions.",
        },
    )

    determinism_fraction: float = field(
        metadata={
            "description": (
                "Fraction of CPT columns set deterministically (0/1 values). "
                "0.0 means all columns are probabilistic, higher values "
                "introduce deterministic relationships (0% recommended by "
                "default)."
            ),
        },
    )

    seed: int | None = field(
        default=None,
        metadata={
            "description": "Random seed used for generating this BN. "
            "None if no seed was specified, ensuring reproducibility.",
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation of the metadata
        """
        return asdict(self)


@dataclass
class BaseBNMetadata:
    """
    Base metadata for a BN-DAG pair before naming variants are applied.

    This metadata tracks the full experimental configuration for generating a
    Bayesian Network, including graph structure parameters, CPT parameters,
    and network properties. Used internally in the sweep pipeline before
    naming variants are applied.
    """

    n: int = field(
        metadata={
            "description": "Number of variables (nodes) in the Bayesian Network.",
        },
    )

    target_tw: int = field(
        metadata={
            "description": "Target treewidth that was requested during DAG generation. "
            "Treewidth correlates with inference complexity.",
        },
    )

    achieved_tw: int = field(
        metadata={
            "description": "Actual treewidth achieved in the generated DAG. "
            "May differ slightly from target_tw due to approximation methods.",
        },
    )

    naming: str = field(
        metadata={
            "description": (
                "Node naming strategy used ('simple', 'confusing', "
                "'semantic', 'mixed'). "
                "This value will be overridden when naming variants are "
                "applied."
            ),
        },
    )

    arity: str = field(
        metadata={
            "description": (
                "Arity specification as a string, e.g., 'fixed:2' or "
                "'range:2-4'. "
                "Describes how variable cardinalities were assigned."
            ),
        },
    )

    alpha: float = field(
        metadata={
            "description": "Dirichlet alpha parameter used for CPT column sampling. "
            "Controls the skewness of probability distributions.",
        },
    )

    determinism: float = field(
        metadata={
            "description": (
                "Fraction of CPT columns that are deterministic (0/1 values). "
                "0.0 means all relationships are probabilistic."
            ),
        },
    )

    variant_index: int = field(
        metadata={
            "description": "Index of this variant within the set of variants generated "
            "for the same DAG and parameter combination.",
        },
    )

    num_edges: int = field(
        metadata={
            "description": (
                "Number of directed edges in the Bayesian Network structure."
            ),
        },
    )

    num_nodes: int = field(
        metadata={
            "description": "Number of nodes in the Bayesian Network (equal to n).",
        },
    )

    base_sample_counter: int = field(
        metadata={
            "description": "Sample counter tracking the original BN generation. "
            "Used to identify which base BN this metadata corresponds to "
            "when multiple naming variants are created.",
        },
    )

    seed: int | None = field(
        default=None,
        metadata={
            "description": "Random seed used for BN generation. "
            "Ensures reproducibility when specified.",
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for compatibility with naming_variants functions.

        Returns:
            Dictionary representation of the metadata
        """
        return asdict(self)

    def update_naming(self, naming_strategy: str) -> BaseBNMetadata:
        """
        Create a new metadata instance with updated naming strategy.

        Args:
            naming_strategy: New naming strategy to apply

        Returns:
            New BaseBNMetadata instance with updated naming field
        """
        return BaseBNMetadata(
            n=self.n,
            target_tw=self.target_tw,
            achieved_tw=self.achieved_tw,
            naming=naming_strategy,
            arity=self.arity,
            alpha=self.alpha,
            determinism=self.determinism,
            seed=self.seed,
            variant_index=self.variant_index,
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            base_sample_counter=self.base_sample_counter,
        )


@dataclass
class OutputRowMetadata:
    """
    Metadata row for DataFrame creation in sweep results.

    This is a simplified metadata structure optimized for creating pandas DataFrames
    from sweep results. Contains only the essential fields needed for analysis,
    without internal tracking fields like base_sample_counter.
    """

    n: int = field(
        metadata={
            "description": "Number of variables (nodes) in the Bayesian Network.",
        },
    )

    target_tw: int = field(
        metadata={
            "description": "Target treewidth that was requested during DAG generation.",
        },
    )

    achieved_tw: int = field(
        metadata={"description": "Actual treewidth achieved in the generated DAG."},
    )

    naming: str = field(
        metadata={
            "description": (
                "Final node naming strategy applied ('simple', 'confusing', "
                "'semantic', 'mixed'). "
                "This reflects the naming after variants are applied."
            ),
        },
    )

    arity: str = field(
        metadata={
            "description": (
                "Arity specification as a string, e.g., 'fixed:2' or 'range:2-4'."
            ),
        },
    )

    alpha: float = field(
        metadata={
            "description": "Dirichlet alpha parameter used for CPT column sampling.",
        },
    )

    determinism: float = field(
        metadata={
            "description": (
                "Fraction of CPT columns that are deterministic (0/1 values)."
            ),
        },
    )

    variant_index: int = field(
        metadata={
            "description": "Index of this variant within the set of variants generated "
            "for the same DAG and parameter combination.",
        },
    )

    num_edges: int = field(
        metadata={
            "description": (
                "Number of directed edges in the Bayesian Network structure."
            ),
        },
    )

    num_nodes: int = field(
        metadata={
            "description": "Number of nodes in the Bayesian Network (equal to n).",
        },
    )

    seed: int | None = field(
        default=None,
        metadata={"description": "Random seed used for BN generation."},
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for DataFrame creation.

        Returns:
            Dictionary representation suitable for pandas DataFrame
        """
        return asdict(self)
