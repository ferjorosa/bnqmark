"""
Query Generation Core Module.

This module provides the fundamental building block for query generation:
generating a SINGLE query with strict constraint enforcement.

For generating MULTIPLE queries, use the sweep module:
    from src.queries.sweep import generate_queries_with_sampling
"""

# Main entry point - use this!
# Re-export key types for type hints and advanced usage
from .generator import generate_single_query
from .types import (
    QueryGenerationMetadata,
    QuerySpec,
)

__all__ = [
    # Main API
    "generate_single_query",
    # Types (for type hints)
    "QuerySpec",
    "QueryGenerationMetadata",
]
