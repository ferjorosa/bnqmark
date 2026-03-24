"""
Database utilities specifically for the *discrete-queries* experiment.

This module encapsulates all SQL helpers that touch the ``discrete_queries`` table.
It is intentionally independent from the generic helpers that live in
:pyfile:`src.database.database` so that each experiment can evolve
without stepping on each other's toes.
"""

import json
import logging

import numpy as np

from src.database.database import (
    execute_many,
    execute_query,
    query_db,
)

# Constants
TABLE_NAME = "discrete_queries"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


def initialize_discrete_queries_db() -> None:
    """
    Initialise the SQLite table for the *discrete-queries* experiment.

    The function will create the ``discrete_queries`` table if it
    does not exist yet but will *not* overwrite an existing database.
    """
    # Check if table exists
    table_exists_query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """

    result_df = query_db(table_exists_query, [TABLE_NAME])

    if result_df.empty:
        create_table_query = f"""
            CREATE TABLE {TABLE_NAME} (
                bn_uuid TEXT NOT NULL,
                query_uuid TEXT NOT NULL,
                naming_strategy TEXT NOT NULL,
                target TEXT NOT NULL,
                evidence TEXT NOT NULL,
                num_evidence INTEGER NOT NULL,
                num_target INTEGER NOT NULL,
                probability REAL NOT NULL,
                prior_probability REAL NOT NULL,
                induced_width INTEGER NOT NULL,
                num_eliminated INTEGER NOT NULL,
                avg_markov_blanket_size_target REAL NOT NULL,
                avg_markov_blanket_size_evidence REAL NOT NULL,
                min_distance_target_evidence INTEGER NOT NULL,
                min_distance_target_target REAL,
                min_distance_evidence_evidence REAL,
                evidence_distances TEXT NOT NULL,
                all_target_are_roots INTEGER NOT NULL,
                all_target_are_leaves INTEGER NOT NULL,
                all_evidence_are_roots INTEGER NOT NULL,
                all_evidence_are_leaves INTEGER NOT NULL,
                num_target_nodes INTEGER NOT NULL,
                num_evidence_nodes INTEGER NOT NULL,
                target_nodes TEXT NOT NULL,
                evidence_nodes TEXT NOT NULL,
                abs_diff REAL NOT NULL,
                rel_diff REAL NOT NULL,
                avg_distance_target_evidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (query_uuid, naming_strategy),
                FOREIGN KEY (bn_uuid, naming_strategy) REFERENCES
                    discrete_bns(bn_uuid, naming_strategy)
            );
        """
        execute_query(create_table_query)
        print(f"Table {TABLE_NAME} created successfully")

        # Create indexes for fast lookups
        index_queries = [
            f"CREATE INDEX idx_{TABLE_NAME}_query_uuid ON {TABLE_NAME} (query_uuid);",
            f"CREATE INDEX idx_{TABLE_NAME}_bn_uuid ON {TABLE_NAME} (bn_uuid);",
            f"CREATE INDEX idx_{TABLE_NAME}_naming_strategy ON {TABLE_NAME} (naming_strategy);",  # noqa: E501
        ]

        for index_query in index_queries:
            execute_query(index_query)

        print("Indexes created successfully")

    else:
        print(f"Table {TABLE_NAME} already exists")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_existing_query_identifiers() -> set[tuple[str, str]]:
    """
    Get all existing (query_uuid, naming_strategy) pairs from the database.

    Returns:
        Set of tuples (query_uuid, naming_strategy)
    """
    query = f"""
        SELECT query_uuid, naming_strategy
        FROM {TABLE_NAME}
    """

    result_df = query_db(query)
    if result_df.empty:
        return set()

    return set(zip(result_df["query_uuid"], result_df["naming_strategy"], strict=False))


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------


def _convert_array_to_json(value) -> str:
    """Convert numpy array to JSON string for database storage."""
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    elif isinstance(value, list):
        return json.dumps(value)
    else:
        return str(value)


def _handle_nan_values(value):
    """Convert NaN values to None for database storage."""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def insert_query_row(
    bn_uuid: str,
    query_uuid: str,
    naming_strategy: str,
    target: str,
    evidence: str,
    num_evidence: int,
    num_target: int,
    probability: float,
    prior_probability: float,
    induced_width: int,
    num_eliminated: int,
    avg_markov_blanket_size_target: float,
    avg_markov_blanket_size_evidence: float,
    min_distance_target_evidence: int,
    min_distance_target_target: float | None,
    min_distance_evidence_evidence: float | None,
    evidence_distances: str,
    all_target_are_roots: bool,
    all_target_are_leaves: bool,
    all_evidence_are_roots: bool,
    all_evidence_are_leaves: bool,
    num_target_nodes: int,
    num_evidence_nodes: int,
    target_nodes: str,
    evidence_nodes: str,
    abs_diff: float,
    rel_diff: float,
    avg_distance_target_evidence: float,
    debug: bool = False,
) -> None:
    """
    Insert a row into the discrete_queries table.

    Args:
        bn_uuid: UUID of the associated Bayesian network
        query_uuid: Unique identifier for the query
        naming_strategy: Naming strategy used
        target: Target variables (JSON string)
        evidence: Evidence variables (JSON string)
        num_evidence: Number of evidence variables
        num_target: Number of target variables
        probability: Query probability
        prior_probability: Prior probability
        induced_width: Induced width
        num_eliminated: Number of eliminated variables
        avg_markov_blanket_size_target: Average Markov blanket size for target
        avg_markov_blanket_size_evidence: Average Markov blanket size for evidence
        min_distance_target_evidence: Minimum distance between target and evidence
        min_distance_target_target: Minimum distance between target variables
        min_distance_evidence_evidence: Minimum distance between evidence variables
        evidence_distances: Evidence distances (JSON string)
        all_target_are_roots: Whether all target variables are roots
        all_target_are_leaves: Whether all target variables are leaves
        all_evidence_are_roots: Whether all evidence variables are roots
        all_evidence_are_leaves: Whether all evidence variables are leaves
        num_target_nodes: Number of target nodes
        num_evidence_nodes: Number of evidence nodes
        target_nodes: Target node names (JSON string)
        evidence_nodes: Evidence node names (JSON string)
        abs_diff: Absolute difference
        rel_diff: Relative difference
        avg_distance_target_evidence: Average distance between target and evidence
        debug: Flag to enable debug output
    """
    query = f"""
        INSERT INTO {TABLE_NAME} (
            bn_uuid, query_uuid, naming_strategy, target, evidence,
            num_evidence, num_target, probability, prior_probability,
            induced_width, num_eliminated,
            avg_markov_blanket_size_target,
            avg_markov_blanket_size_evidence,
            min_distance_target_evidence,
            min_distance_target_target,
            min_distance_evidence_evidence,
            evidence_distances, all_target_are_roots,
            all_target_are_leaves, all_evidence_are_roots,
            all_evidence_are_leaves, num_target_nodes,
            num_evidence_nodes, target_nodes, evidence_nodes,
            abs_diff, rel_diff, avg_distance_target_evidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        bn_uuid,
        query_uuid,
        naming_strategy,
        target,
        evidence,
        num_evidence,
        num_target,
        probability,
        prior_probability,
        induced_width,
        num_eliminated,
        avg_markov_blanket_size_target,
        avg_markov_blanket_size_evidence,
        min_distance_target_evidence,
        min_distance_target_target,
        min_distance_evidence_evidence,
        evidence_distances,
        int(all_target_are_roots),
        int(all_target_are_leaves),
        int(all_evidence_are_roots),
        int(all_evidence_are_leaves),
        num_target_nodes,
        num_evidence_nodes,
        target_nodes,
        evidence_nodes,
        abs_diff,
        rel_diff,
        avg_distance_target_evidence,
    )

    execute_query(query, params)

    if debug:
        print(f"Query row inserted: {query_uuid}")


def insert_query_batch(query_rows: list[dict], debug: bool = False) -> int:
    """
    Insert multiple query rows in a single transaction.

    Args:
        query_rows: List of dictionaries containing query data
        debug: Flag to enable debug output

    Returns:
        Number of rows inserted
    """
    if not query_rows:
        return 0

    # Process the data to handle arrays and NaN values
    processed_rows = []
    for row in query_rows:
        processed_row = row.copy()

        # Convert array columns to JSON strings
        processed_row["evidence_distances"] = _convert_array_to_json(
            row["evidence_distances"],
        )
        processed_row["target_nodes"] = _convert_array_to_json(row["target_nodes"])
        processed_row["evidence_nodes"] = _convert_array_to_json(row["evidence_nodes"])

        # Handle NaN values
        processed_row["min_distance_target_target"] = _handle_nan_values(
            row["min_distance_target_target"],
        )
        processed_row["min_distance_evidence_evidence"] = _handle_nan_values(
            row["min_distance_evidence_evidence"],
        )

        processed_rows.append(processed_row)

    query = f"""
        INSERT INTO {TABLE_NAME} (
            bn_uuid, query_uuid, naming_strategy, target, evidence,
            num_evidence, num_target, probability, prior_probability,
            induced_width, num_eliminated,
            avg_markov_blanket_size_target,
            avg_markov_blanket_size_evidence,
            min_distance_target_evidence,
            min_distance_target_target,
            min_distance_evidence_evidence,
            evidence_distances, all_target_are_roots,
            all_target_are_leaves, all_evidence_are_roots,
            all_evidence_are_leaves, num_target_nodes,
            num_evidence_nodes, target_nodes, evidence_nodes,
            abs_diff, rel_diff, avg_distance_target_evidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params_list = []
    for row in processed_rows:
        params = (
            row["bn_uuid"],
            row["query_uuid"],
            row["naming_strategy"],
            row["target"],
            row["evidence"],
            row["num_evidence"],
            row["num_target"],
            row["probability"],
            row["prior_probability"],
            row["induced_width"],
            row["num_eliminated"],
            row["avg_markov_blanket_size_target"],
            row["avg_markov_blanket_size_evidence"],
            row["min_distance_target_evidence"],
            row["min_distance_target_target"],
            row["min_distance_evidence_evidence"],
            row["evidence_distances"],
            int(row["all_target_are_roots"]),
            int(row["all_target_are_leaves"]),
            int(row["all_evidence_are_roots"]),
            int(row["all_evidence_are_leaves"]),
            row["num_target_nodes"],
            row["num_evidence_nodes"],
            row["target_nodes"],
            row["evidence_nodes"],
            row["abs_diff"],
            row["rel_diff"],
            row["avg_distance_target_evidence"],
        )
        params_list.append(params)

    execute_many(query, params_list)

    if debug:
        print(f"Batch inserted {len(query_rows)} query rows")

    return len(query_rows)
