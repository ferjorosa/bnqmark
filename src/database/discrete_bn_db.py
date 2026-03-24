"""
Database utilities specifically for the *discrete-bn* experiment.

This module encapsulates all SQL helpers that touch the ``discrete_bns`` table.
It is intentionally independent from the generic helpers that live in
:pyfile:`src.database.database` so that each experiment can evolve
without stepping on each other's toes.
"""

import logging

from src.database.database import (
    execute_many,
    execute_query,
    query_db,
)

# Constants
TABLE_NAME = "discrete_bns"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


def initialize_discrete_bns_db() -> None:
    """
    Initialise the SQLite table for the *discrete-bn* experiment.

    The function will create the ``discrete_bns`` table if it
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
                n INTEGER NOT NULL,
                target_tw INTEGER NOT NULL,
                achieved_tw INTEGER NOT NULL,
                arity TEXT NOT NULL,
                alpha REAL NOT NULL,
                determinism REAL NOT NULL,
                variant_index INTEGER NOT NULL,
                num_edges INTEGER NOT NULL,
                num_nodes INTEGER NOT NULL,
                base_sample_counter INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                edge_density REAL NOT NULL,
                avg_markov_blanket_size REAL NOT NULL,
                naming_strategy TEXT NOT NULL,
                bn_pickle BLOB NOT NULL,
                dag_pickle BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (bn_uuid, naming_strategy)
            );
        """
        execute_query(create_table_query)
        print(f"Table {TABLE_NAME} created successfully")

        # Create index on bn_uuid for fast lookups
        index_query = f"""
            CREATE INDEX idx_{TABLE_NAME}_bn_uuid ON {TABLE_NAME} (bn_uuid);
        """
        execute_query(index_query)
        print("Index on bn_uuid created successfully")

    else:
        print(f"Table {TABLE_NAME} already exists")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_existing_bn_identifiers() -> set[tuple[str, str]]:
    """
    Get all existing (bn_uuid, naming_strategy) pairs from the database.

    Returns:
        Set of tuples (bn_uuid, naming_strategy)
    """
    query = f"""
        SELECT bn_uuid, naming_strategy
        FROM {TABLE_NAME}
    """

    result_df = query_db(query)
    if result_df.empty:
        return set()

    return set(zip(result_df["bn_uuid"], result_df["naming_strategy"], strict=False))


def get_existing_bn(bn_uuid: str) -> dict | None:
    """
    Retrieve an existing Bayesian network from the database.

    Args:
        bn_uuid: The unique identifier for the Bayesian network

    Returns:
        Dictionary with the BN data, or None if not found
    """
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE bn_uuid = ?
        LIMIT 1
    """

    result_df = query_db(query, [bn_uuid])
    result = result_df.to_dict("records")

    if result:
        return result[0]

    return None


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def insert_bn_row(
    bn_uuid: str,
    n: int,
    target_tw: int,
    achieved_tw: int,
    arity: str,
    alpha: float,
    determinism: float,
    variant_index: int,
    num_edges: int,
    num_nodes: int,
    base_sample_counter: int,
    seed: int,
    edge_density: float,
    avg_markov_blanket_size: float,
    naming_strategy: str,
    bn_pickle: bytes,
    dag_pickle: bytes,
    debug: bool = False,
) -> None:
    """
    Insert a row into the discrete_bns table.

    Args:
        bn_uuid: Unique identifier for the Bayesian network
        n: Network size parameter
        target_tw: Target treewidth
        achieved_tw: Achieved treewidth
        arity: Arity specification
        alpha: Dirichlet alpha parameter
        determinism: Determinism fraction
        variant_index: Variant index for this parameter combination
        num_edges: Number of edges in the network
        num_nodes: Number of nodes in the network
        base_sample_counter: Base sample counter
        seed: Random seed used
        edge_density: Edge density of the network
        avg_markov_blanket_size: Average Markov blanket size
        naming_strategy: Naming strategy used
        bn_pickle: Serialized Bayesian network object
        dag_pickle: Serialized DAG object
        debug: Flag to enable debug output
    """
    query = f"""
        INSERT INTO {TABLE_NAME} (
            bn_uuid, n, target_tw, achieved_tw, arity, alpha, determinism,
            variant_index, num_edges, num_nodes, base_sample_counter, seed,
            edge_density, avg_markov_blanket_size, naming_strategy,
            bn_pickle, dag_pickle
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        bn_uuid,
        n,
        target_tw,
        achieved_tw,
        arity,
        alpha,
        determinism,
        variant_index,
        num_edges,
        num_nodes,
        base_sample_counter,
        seed,
        edge_density,
        avg_markov_blanket_size,
        naming_strategy,
        bn_pickle,
        dag_pickle,
    )

    execute_query(query, params)

    if debug:
        print(f"Bayesian network row inserted: {bn_uuid}")


def insert_bn_batch(bn_rows: list[dict], debug: bool = False) -> int:
    """
    Insert multiple Bayesian network rows in a single transaction.

    Args:
        bn_rows: List of dictionaries containing BN data
        debug: Flag to enable debug output

    Returns:
        Number of rows inserted
    """
    if not bn_rows:
        return 0

    query = f"""
        INSERT INTO {TABLE_NAME} (
            bn_uuid, n, target_tw, achieved_tw, arity, alpha, determinism,
            variant_index, num_edges, num_nodes, base_sample_counter, seed,
            edge_density, avg_markov_blanket_size, naming_strategy,
            bn_pickle, dag_pickle
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params_list = []
    for row in bn_rows:
        params = (
            row["bn_uuid"],
            row["n"],
            row["target_tw"],
            row["achieved_tw"],
            row["arity"],
            row["alpha"],
            row["determinism"],
            row["variant_index"],
            row["num_edges"],
            row["num_nodes"],
            row["base_sample_counter"],
            row["seed"],
            row["edge_density"],
            row["avg_markov_blanket_size"],
            row["naming_strategy"],
            row["bn_pickle"],
            row["dag_pickle"],
        )
        params_list.append(params)

    execute_many(query, params_list)

    if debug:
        print(f"Batch inserted {len(bn_rows)} Bayesian network rows")

    return len(bn_rows)
