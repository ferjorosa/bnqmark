"""
Core DAG generation implementation.

This package contains the low-level implementation details for generating
DAGs with controlled treewidth. Most users should use the high-level API
from generator.py instead of importing directly from this package.

Internal modules:
    - generation: Core graph and DAG generation algorithms
    - types: Type definitions and metadata dataclasses
    - naming: Node naming strategies
"""

# Re-export key functions for internal use
from .generation import (
    generate_graph_with_target_treewidth,
    undirected_to_dag,
)
from .naming import (
    generate_node_names,
    relabel_graph_nodes,
)
from .types import (
    DAGGenerationMetadata,
    GraphGenerationMetadata,
    NamingStrategy,
)

__all__ = [
    # Generation
    "generate_graph_with_target_treewidth",
    "undirected_to_dag",
    # Types
    "DAGGenerationMetadata",
    "GraphGenerationMetadata",
    "NamingStrategy",
    # Naming
    "generate_node_names",
    "relabel_graph_nodes",
]
