"""
Type definitions for Query generation.

This module provides all the dataclasses and type definitions used throughout
the query generation pipeline, including query specifications and metadata structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork


@dataclass
class QueryGenerationContext:
    """Internal context for query generation."""

    model: DiscreteBayesianNetwork
    G: nx.DiGraph
    nodes: list[str]
    state_labels: dict[str, list[str]]
    rng: np.random.Generator
    distance_bucket: tuple[int, int]

    @property
    def dmin(self) -> int:
        """Minimum distance for evidence placement."""
        return self.distance_bucket[0]

    @property
    def dmax(self) -> int:
        """Maximum distance for evidence placement."""
        return self.distance_bucket[1]


@dataclass
class QueryGenerationMetadata:
    """
    Metadata about a generated query's difficulty characteristics.

    This metadata captures the essential difficulty metrics for a query,
    including the number of query/evidence variables and their spatial
    relationship in the network graph.

    Attributes:
        num_query_nodes: Number of variables being queried (1 or 2)
        num_evidence_nodes: Number of evidence variables (0 or more)
        distance_bucket: (min, max) tuple of the desired distance range for
            evidence placement
        min_target_evidence_distance: Actual minimum distance between any query variable
                                     and any evidence variable in the undirected graph
    """

    num_query_nodes: int = field(
        metadata={"description": "Number of query variables (1 or 2)."},
    )

    num_evidence_nodes: int = field(
        metadata={"description": "Number of evidence variables (0 or more)."},
    )

    distance_bucket: tuple[int, int] = field(
        metadata={
            "description": "Desired distance range (min, max) for evidence "
            "placement in the graph.",
        },
    )

    min_target_evidence_distance: int = field(
        metadata={
            "description": "Actual minimum distance between any query variable "
            "and any evidence variable in the undirected graph. 0 if no evidence.",
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation of the metadata
        """
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like access for backward compatibility.

        Allows accessing metadata fields using dict-style `.get()` method.

        Args:
            key: Field name to access
            default: Default value if key not found

        Returns:
            Field value or default if key doesn't exist
        """
        return getattr(self, key, default)


@dataclass
class QuerySpec:
    """
    Specification for a probabilistic query with evidence and optional probabilities.

    A QuerySpec defines a complete probabilistic query that can be posed to a
    Bayesian Network, including the target variables to query, their desired states,
    any evidence assignments, metadata about the query's difficulty characteristics,
    and optionally the computed posterior and prior probabilities.

    Attributes:
        targets: Dictionary mapping query variable names to their desired states.
                For example: {"Rain": "true", "Sprinkler": "false"}
        evidence: Dictionary mapping evidence variable names to their observed states.
                 For example: {"Cloudy": "true", "WetGrass": "false"}
        meta: QueryGenerationMetadata containing difficulty metrics and
            generation parameters.
        posterior_probability: Computed posterior probability P(targets | evidence).
                              None if not computed yet.
        prior_probability: Computed prior probability P(targets).
                          None if not computed yet.
    """

    # One or two query nodes with chosen states (state labels)
    targets: dict[str, str]
    # Evidence assignments as mapping node -> state label
    evidence: dict[str, str]
    # Metadata about difficulty dimensions
    meta: QueryGenerationMetadata
    # Computed probabilities (optional, populated by sweep module)
    posterior_probability: float | None = field(default=None)
    prior_probability: float | None = field(default=None)
