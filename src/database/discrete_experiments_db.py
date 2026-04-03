"""
Database utilities specifically for the *discrete-experiments* experiment.

This module encapsulates all SQL helpers that touch the ``discrete_experiments`` table.
It is intentionally independent from the generic helpers that live in
:pyfile:`src.database.database` so that each experiment can evolve
without stepping on each other's toes.
"""

import json
import logging

from src.database.database import (
    execute_many,
    execute_query,
    query_db,
)

# Constants
TABLE_NAME = "discrete_experiments"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


def initialize_discrete_experiments_db() -> None:
    """
    Initialise the SQLite table for the *discrete-experiments* experiment.

    The function will create the ``discrete_experiments`` table if it
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
                query_uuid TEXT NOT NULL,
                naming_strategy TEXT NOT NULL,
                run INTEGER NOT NULL,
                experiment_type TEXT NOT NULL DEFAULT 'raw_reasoning',
                full_prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                response_reasoning_summary TEXT,
                response_metadata TEXT,
                model_name TEXT NOT NULL,
                reasoning_model BOOLEAN NOT NULL,
                openai_reasoning_effort TEXT,
                openai_reasoning_summary TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                usage_metadata TEXT,
                temperature REAL NOT NULL,
                llm_probability REAL,
                code_followed_formatting_instructions BOOLEAN,
                code_pgmpy_library_fix BOOLEAN,
                code_exception_type VARCHAR(100),
                code_exception_message TEXT,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (query_uuid, naming_strategy, run, model_name, experiment_type),
                FOREIGN KEY (query_uuid, naming_strategy) REFERENCES
                    discrete_queries(query_uuid, naming_strategy)
            );
        """
        execute_query(create_table_query)
        print(f"Table {TABLE_NAME} created successfully")

        # Create indexes for fast lookups
        index_queries = [
            f"CREATE INDEX idx_{TABLE_NAME}_query_uuid ON {TABLE_NAME} (query_uuid);",
            f"CREATE INDEX idx_{TABLE_NAME}_naming_strategy "
            f"ON {TABLE_NAME} (naming_strategy);",
            f"CREATE INDEX idx_{TABLE_NAME}_model_name ON {TABLE_NAME} (model_name);",
            f"CREATE INDEX idx_{TABLE_NAME}_run ON {TABLE_NAME} (run);",
            f"CREATE INDEX idx_{TABLE_NAME}_experiment_type "
            f"ON {TABLE_NAME} (experiment_type);",  # noqa: E501
        ]

        for index_query in index_queries:
            execute_query(index_query)

        print("Indexes created successfully")

    else:
        print(f"Table {TABLE_NAME} already exists")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_existing_experiment_identifiers() -> set[tuple[str, str, int, str, str]]:
    """
    Get all existing experiment identifiers from the database.

    Returns a set of (query_uuid, naming_strategy, run, model_name,
    experiment_type) tuples.

    Returns:
        Set of tuples (query_uuid, naming_strategy, run, model_name,
        experiment_type)
    """
    query = f"""
        SELECT query_uuid, naming_strategy, run, model_name, experiment_type
        FROM {TABLE_NAME}
    """

    result_df = query_db(query)
    if result_df.empty:
        return set()

    return set(
        zip(
            result_df["query_uuid"],
            result_df["naming_strategy"],
            result_df["run"],
            result_df["model_name"],
            result_df["experiment_type"],
            strict=False,
        )
    )


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def insert_experiment_row(
    query_uuid: str,
    naming_strategy: str,
    run: int,
    experiment_type: str,
    full_prompt: str,
    response: str,
    model_name: str,
    reasoning_model: bool,
    input_tokens: int | None,
    output_tokens: int | None,
    temperature: float,
    started_at: str,
    finished_at: str,
    usage_metadata: dict | None = None,
    llm_probability: float | None = None,
    code_followed_formatting_instructions: bool | None = None,
    code_pgmpy_library_fix: bool | None = None,
    code_exception_type: str | None = None,
    code_exception_message: str | None = None,
    openai_reasoning_effort: str | None = None,
    openai_reasoning_summary: str | None = None,
    response_reasoning_summary: str | None = None,
    response_metadata: dict | None = None,
    debug: bool = False,
) -> None:
    """
    Insert a row into the discrete_experiments table.

    Args:
        query_uuid: UUID of the associated query
        naming_strategy: Naming strategy used for the query
        run: Run number for this experiment
        experiment_type: Type of experiment ("raw_reasoning" or "code_generation")
        full_prompt: The complete prompt sent to the LLM
        response: The LLM's response
        model_name: Name of the model used (distinguishes same model with
            different reasoning modes)
        reasoning_model: Whether this is a reasoning model
        openai_reasoning_effort: Reasoning effort level ("low", "medium",
            "high", or None)
        openai_reasoning_summary: Reasoning summary level ("auto", "concise",
            "detailed", or None)
        response_reasoning_summary: Actual reasoning summary returned from the
            model (or None)
        response_metadata: Dictionary containing response metadata (or None)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        usage_metadata: Dictionary containing usage metadata (or None)
        temperature: Temperature parameter used
        started_at: Timestamp when the query execution started (ISO format string)
        finished_at: Timestamp when the query execution finished (ISO format string)
        llm_probability: Extracted probability from response (optional)
        code_followed_formatting_instructions: Whether code used <code></code>
            tags (optional)
        code_pgmpy_library_fix: Whether pgmpy API fix was applied (optional)
        code_exception_type: Type of exception if code execution failed (optional)
        code_exception_message: Error message if code execution failed (optional)
        debug: Flag to enable debug output
    """
    # Convert metadata dicts to JSON strings for storage
    response_metadata_json = (
        json.dumps(response_metadata) if response_metadata else None
    )
    usage_metadata_json = json.dumps(usage_metadata) if usage_metadata else None

    query = f"""
        INSERT INTO {TABLE_NAME} (
            query_uuid, naming_strategy, run, experiment_type, full_prompt,
            response, response_reasoning_summary, response_metadata,
            model_name, reasoning_model, openai_reasoning_effort,
            openai_reasoning_summary, input_tokens, output_tokens, usage_metadata,
            temperature, llm_probability, code_followed_formatting_instructions,
            code_pgmpy_library_fix, code_exception_type, code_exception_message,
            started_at, finished_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?)
    """

    params = (
        query_uuid,
        naming_strategy,
        run,
        experiment_type,
        full_prompt,
        response,
        response_reasoning_summary,
        response_metadata_json,
        model_name,
        reasoning_model,
        openai_reasoning_effort,
        openai_reasoning_summary,
        input_tokens,
        output_tokens,
        usage_metadata_json,
        temperature,
        llm_probability,
        code_followed_formatting_instructions,
        code_pgmpy_library_fix,
        code_exception_type,
        code_exception_message,
        started_at,
        finished_at,
    )

    execute_query(query, params)

    if debug:
        print(
            f"Inserted: {query_uuid}, naming_strategy {naming_strategy}, "
            f"run {run}, experiment_type {experiment_type}, model {model_name}"
        )


def insert_experiment_batch(experiment_rows: list[dict], debug: bool = False) -> int:
    """
    Insert multiple experiment rows in a single transaction.

    Args:
        experiment_rows: List of dictionaries containing experiment data
        debug: Flag to enable debug output

    Returns:
        Number of rows inserted
    """
    if not experiment_rows:
        return 0

    query = f"""
        INSERT INTO {TABLE_NAME} (
            query_uuid, naming_strategy, run, experiment_type, full_prompt,
            response, response_reasoning_summary, response_metadata,
            model_name, reasoning_model, openai_reasoning_effort,
            openai_reasoning_summary, input_tokens, output_tokens, usage_metadata,
            temperature, llm_probability, code_followed_formatting_instructions,
            code_pgmpy_library_fix, code_exception_type, code_exception_message,
            started_at, finished_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?)
    """

    params_list = []
    for row in experiment_rows:
        # Convert metadata dicts to JSON strings for storage
        response_metadata_json = (
            json.dumps(row.get("response_metadata"))
            if row.get("response_metadata")
            else None
        )
        usage_metadata_json = (
            json.dumps(row.get("usage_metadata")) if row.get("usage_metadata") else None
        )

        # Convert timestamps to ISO format strings for SQLite compatibility
        started_at = row["started_at"]
        finished_at = row["finished_at"]
        if hasattr(started_at, "isoformat"):
            started_at = started_at.isoformat()
        if hasattr(finished_at, "isoformat"):
            finished_at = finished_at.isoformat()

        params = (
            row["query_uuid"],
            row["naming_strategy"],
            row["run"],
            row["experiment_type"],
            row["full_prompt"],
            row["response"],
            row.get("response_reasoning_summary"),
            response_metadata_json,
            row["model_name"],
            row["reasoning_model"],
            row.get("openai_reasoning_effort"),
            row.get("openai_reasoning_summary"),
            row.get("input_tokens"),
            row.get("output_tokens"),
            usage_metadata_json,
            row["temperature"],
            row.get("llm_probability"),
            row.get("code_followed_formatting_instructions"),
            row.get("code_pgmpy_library_fix"),
            row.get("code_exception_type"),
            row.get("code_exception_message"),
            started_at,
            finished_at,
        )
        params_list.append(params)

    execute_many(query, params_list)

    if debug:
        print(f"Batch inserted {len(experiment_rows)} experiment rows")

    return len(experiment_rows)
