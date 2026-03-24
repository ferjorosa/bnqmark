"""
DAG Analysis Module.

This module provides functions for analyzing structural and complexity
properties of generated graphs and DAGs.

Main Functions:
    analyze_graph_properties() - Analyze various graph properties
    verify_dag_properties() - Verify DAG meets expected properties

Example:
    >>> from src.dag import generate_single_dag, analyze_graph_properties
    >>> dag, tw, _ = generate_single_dag(10, 3, seed=42)
    >>> props = analyze_graph_properties(dag)
    >>> print(f"Treewidth: {props['treewidth']}, Density: {props['density']:.2f}")
"""

from .analysis import (
    analyze_graph_properties,
    verify_dag_properties,
)

__all__ = [
    "analyze_graph_properties",
    "verify_dag_properties",
]
