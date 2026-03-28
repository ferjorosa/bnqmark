"""
Database utilities for experiment data management.

This module provides SQLite database operations for local data management.
Lightweight, serverless, and requires no external dependencies.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Global state
_db_path: str | None = None
_connection: sqlite3.Connection | None = None


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------


def init_db(db_path: str | None = None) -> str:
    """
    Initialize the SQLite database.

    Args:
        db_path: Path to SQLite database file. If None, uses default location.

    Returns:
        str: The path to the database file
    """
    global _db_path

    if db_path is None:
        # Default to a 'data' directory in the project
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        db_path = str(data_dir / "probabilistic_inference_llms.db")

    _db_path = db_path

    # Ensure the directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Database initialized at: {db_path}")
    return db_path


def get_db_path() -> str:
    """
    Get the current database path, initializing if needed.

    Returns:
        str: The path to the database file
    """
    if _db_path is None:
        return init_db()
    return _db_path


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """
    Get a SQLite connection.

    Uses a single persistent connection with row factory for dict-like access.

    Returns:
        sqlite3.Connection: SQLite database connection
    """
    global _connection

    db_path = get_db_path()

    if _connection is None:
        _connection = sqlite3.connect(db_path)
        # Enable foreign keys
        _connection.execute("PRAGMA foreign_keys = ON")
        # Return rows as dictionary-like objects
        _connection.row_factory = sqlite3.Row

    return _connection


def close_connection() -> None:
    """Close the global database connection."""
    global _connection

    if _connection is not None:
        _connection.close()
        _connection = None
        logger.info("Database connection closed")


# ---------------------------------------------------------------------------
# Query Execution Functions
# ---------------------------------------------------------------------------


def query_db(query: str, params: list | None = None) -> pd.DataFrame:
    """
    Execute a SELECT query and return DataFrame.

    Args:
        query: SQL SELECT query to execute
        params: Query parameters as a list (optional)

    Returns:
        pd.DataFrame: Query results
    """
    conn = get_connection()

    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except sqlite3.Error as e:
        logger.error(f"SQLite query failed: {e}")
        raise RuntimeError(f"SQLite query failed: {e}") from e


def execute_query(query: str, params: tuple | None = None) -> None:
    """
    Execute a non-returning query.

    Use this for INSERT, UPDATE, DELETE, CREATE, etc.

    Args:
        query: SQL query to execute
        params: Query parameters (optional)
    """
    conn = get_connection()

    try:
        with conn:
            conn.execute(query, params or ())
    except sqlite3.Error as e:
        logger.error(f"SQLite execute failed: {e}")
        raise RuntimeError(f"SQLite execute failed: {e}") from e


def execute_many(query: str, params_list: list[tuple]) -> None:
    """
    Execute a query with multiple parameter sets.

    Efficient for batch inserts/updates.

    Args:
        query: SQL query to execute
        params_list: List of parameter tuples
    """
    conn = get_connection()

    try:
        with conn:
            conn.executemany(query, params_list)
    except sqlite3.Error as e:
        logger.error(f"SQLite batch execute failed: {e}")
        raise RuntimeError(f"SQLite batch execute failed: {e}") from e


# ---------------------------------------------------------------------------
# Schema Utilities
# ---------------------------------------------------------------------------


def create_table(table_name: str, schema: dict[str, str]) -> None:
    """
    Create a table with the given schema.

    Args:
        table_name: Name of the table to create
        schema: Dictionary mapping column names to SQL type definitions
    """
    columns = ", ".join(f"{col} {dtype}" for col, dtype in schema.items())
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
    execute_query(query)
    logger.info(f"Table '{table_name}' created/verified")


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        table_name: Name of the table to check

    Returns:
        bool: True if table exists, False otherwise
    """
    query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """
    df = query_db(query, [table_name])
    return not df.empty


# ---------------------------------------------------------------------------
# Connection Cleanup
# ---------------------------------------------------------------------------


def close_connections():
    """Close all database connections."""
    close_connection()


def vacuum() -> None:
    """Run VACUUM to optimize the database file size."""
    conn = get_connection()
    conn.execute("VACUUM")
    logger.info("Database vacuumed")
