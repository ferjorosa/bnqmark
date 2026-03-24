"""
Query Sweep Module.

This module provides functionality for generating multiple queries from Bayesian
Networks with systematic parameter sweeps and threshold-based sampling.
"""

from .sweep import generate_queries

__all__ = ["generate_queries"]
