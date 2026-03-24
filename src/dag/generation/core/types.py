"""
Type definitions for DAG generation.

This module provides dataclasses and type definitions used throughout
the DAG generation pipeline, including configuration and metadata structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# ------------------------------
# Strategy Types
# ------------------------------


class NamingStrategy(str, Enum):
    """
    Enumeration of valid node naming strategies for DAG generation.

    Different naming strategies can help test how node names affect downstream
    tasks like LLM probabilistic reasoning.

    Values:
        - SIMPLE: V0, V1, V2, ... (clear and systematic)
        - CONFUSING: X_445aFa, S_af3a34, ... (random alphanumeric)
        - SEMANTIC: Rain, Sprinkler, WetGrass, ... (meaningful names)
        - MIXED: combination of different strategies
        - DEFAULT: numeric labels 0, 1, 2, ...

    Examples:
        >>> strategy = NamingStrategy.SIMPLE
        >>> print(strategy.value)
        'simple'

        >>> # Can also use string for backward compatibility
        >>> strategy = NamingStrategy("simple")
        >>> print(strategy)
        NamingStrategy.SIMPLE
    """

    SIMPLE = "simple"
    CONFUSING = "confusing"
    SEMANTIC = "semantic"
    MIXED = "mixed"
    DEFAULT = "default"


# ------------------------------
# Metadata Types
# ------------------------------


@dataclass
class DAGGenerationMetadata:
    """
    Metadata returned from DAG generation.

    This metadata captures the essential parameters and results from generating
    a DAG with target treewidth, including structural properties and generation
    method details.
    """

    n_nodes: int = field(
        metadata={"description": "Number of nodes in the generated DAG."},
    )

    target_treewidth: int = field(
        metadata={
            "description": "Target treewidth requested during generation. "
            "Treewidth correlates with exact inference complexity.",
        },
    )

    achieved_treewidth: int = field(
        metadata={
            "description": "Actual treewidth achieved in the generated DAG. "
            "May differ from target due to approximation methods.",
        },
    )

    treewidth_difference: int = field(
        metadata={
            "description": "Absolute difference between target and achieved treewidth. "
            "0 means exact match.",
        },
    )

    exact_treewidth: bool = field(
        metadata={
            "description": "Whether the target treewidth was achieved exactly. "
            "True if treewidth_difference == 0.",
        },
    )

    dag_method: str = field(
        metadata={
            "description": (
                "DAG conversion method used ('random', 'topological', 'bfs', 'dfs'). "
                "'random' and 'topological' preserve treewidth, "
                "'bfs' and 'dfs' create spanning trees (treewidth=1)."
            ),
        },
    )

    node_naming: str = field(
        metadata={
            "description": (
                "Node naming strategy used (stored as string value from "
                "NamingStrategy enum)."
            ),
        },
    )

    max_iterations: int = field(
        metadata={"description": "Maximum iterations used during treewidth search."},
    )

    base_graph_edges: int = field(
        metadata={
            "description": (
                "Number of edges in the base undirected graph before DAG conversion."
            ),
        },
    )

    dag_edges: int = field(
        metadata={
            "description": (
                "Number of edges in the final DAG (should equal base_graph_edges)."
            ),
        },
    )

    final_treewidth: int = field(
        metadata={
            "description": "Final treewidth of the DAG's underlying undirected graph. "
            "Should equal achieved_treewidth.",
        },
    )

    node_names: list[str] | None = field(
        default=None,
        metadata={
            "description": "List of node names if naming strategy was applied. "
            "None if default numeric labels were kept.",
        },
    )

    seed: int | None = field(
        default=None,
        metadata={
            "description": "Random seed used for generation. "
            "None if no seed was specified.",
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the metadata
        """
        return asdict(self)


@dataclass
class GraphGenerationMetadata:
    """
    Metadata from undirected graph generation with target treewidth.

    This is an internal metadata structure used during the iterative
    graph generation process before DAG conversion.
    """

    n_nodes: int = field(metadata={"description": "Number of nodes in the graph."})

    target_treewidth: int = field(
        metadata={"description": "Target treewidth for the graph."},
    )

    achieved_treewidth: int = field(
        metadata={"description": "Actual achieved treewidth."},
    )

    treewidth_difference: int = field(
        metadata={"description": "Difference from target treewidth."},
    )

    iterations_used: int = field(
        metadata={"description": "Number of iterations used in generation."},
    )

    n_edges: int = field(
        metadata={"description": "Number of edges in the generated graph."},
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation of the metadata
        """
        return asdict(self)
