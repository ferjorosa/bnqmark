"""Database utilities for the code generation behaviour analysis table."""

import json
import logging
from datetime import datetime

from src.database.database import (
    execute_query,
    query_db,
)
from src.trace_analysis.pydantic_models.code_generation_behaviour_analysis import (
    CodeGenerationBehaviourAnalysis,
)

# Constants
TABLE_NAME = "discrete_code_generation_behaviour_analysis"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


def initialize_code_generation_behaviour_analysis_db() -> None:
    """
    Initialise the SQLite table for code generation behaviour analysis.

    Creates the table if it does not exist.
    """
    table_exists_query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """
    result_df = query_db(table_exists_query, [TABLE_NAME])

    if result_df.empty:
        create_query = f"""
            CREATE TABLE {TABLE_NAME} (
                query_uuid TEXT NOT NULL,
                naming_strategy TEXT NOT NULL,
                run INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                experiment_type TEXT NOT NULL,

                -- Analysis Fields (stored as JSON for structured objects)
                manual_computation_volume TEXT NOT NULL,
                uses_symbolic_math TEXT NOT NULL,

                analysis_duration_ms REAL,

                -- LLM Call Metadata (The analysis run itself)
                llm_call_full_prompt TEXT,
                llm_call_reasoning TEXT,
                llm_call_metadata TEXT,
                llm_call_model TEXT,
                llm_call_input_tokens INTEGER,
                llm_call_output_tokens INTEGER,
                llm_call_usage_metadata TEXT,
                llm_call_temperature REAL,
                llm_call_started_at TIMESTAMP,
                llm_call_finished_at TIMESTAMP,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (query_uuid, naming_strategy, run, model_name, experiment_type)
            );
        """
        execute_query(create_query)
        print(f"Table {TABLE_NAME} created successfully")

        # Indexes
        index_queries = [
            f"CREATE INDEX idx_{TABLE_NAME}_query_uuid ON {TABLE_NAME} (query_uuid);",
            f"CREATE INDEX idx_{TABLE_NAME}_model_run ON {TABLE_NAME} (model_name, run);",  # noqa: E501
        ]
        for q in index_queries:
            execute_query(q)
        print("Indexes created successfully")
    else:
        print(f"Table {TABLE_NAME} already exists")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_existing_code_generation_behaviour_analysis_identifiers() -> set[
    tuple[str, str, int, str, str]
]:
    """Get existing (query_uuid, naming_strategy, run, model, experiment) tuples."""
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


def insert_code_generation_behaviour_analysis(
    query_uuid: str,
    naming_strategy: str,
    run: int,
    model_name: str,
    experiment_type: str,
    analysis_object: CodeGenerationBehaviourAnalysis,
    analysis_duration_ms: float | None = None,
    # LLM Call Metadata
    llm_call_full_prompt: str | None = None,
    llm_call_reasoning: str | None = None,
    llm_call_metadata: dict | None = None,
    llm_call_model: str | None = None,
    llm_call_input_tokens: int | None = None,
    llm_call_output_tokens: int | None = None,
    llm_call_usage_metadata: dict | None = None,
    llm_call_temperature: float | None = None,
    llm_call_started_at: datetime | None = None,
    llm_call_finished_at: datetime | None = None,
) -> None:
    """Insert a row into the discrete_code_generation_behaviour_analysis table."""
    # Extract fields from Pydantic model
    manual_computation_volume = (
        analysis_object.manual_computation_volume.model_dump_json()
    )
    uses_symbolic_math = analysis_object.uses_symbolic_math.model_dump_json()

    # Convert metadata dicts to JSON strings
    llm_call_metadata_json = (
        json.dumps(llm_call_metadata) if llm_call_metadata else None
    )
    llm_call_usage_metadata_json = (
        json.dumps(llm_call_usage_metadata) if llm_call_usage_metadata else None
    )

    query = f"""
        INSERT INTO {TABLE_NAME} (
            query_uuid, naming_strategy, run, model_name, experiment_type,
            manual_computation_volume, uses_symbolic_math,
            analysis_duration_ms,
            llm_call_full_prompt, llm_call_reasoning, llm_call_metadata, llm_call_model,
            llm_call_input_tokens, llm_call_output_tokens, llm_call_usage_metadata,
            llm_call_temperature, llm_call_started_at, llm_call_finished_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        query_uuid,
        naming_strategy,
        run,
        model_name,
        experiment_type,
        manual_computation_volume,
        uses_symbolic_math,
        analysis_duration_ms,
        llm_call_full_prompt,
        llm_call_reasoning,
        llm_call_metadata_json,
        llm_call_model,
        llm_call_input_tokens,
        llm_call_output_tokens,
        llm_call_usage_metadata_json,
        llm_call_temperature,
        llm_call_started_at,
        llm_call_finished_at,
    )

    execute_query(query, params)
    logger.info(
        f"Inserted code generation behaviour analysis for query {query_uuid}, "
        f"model {model_name}, run {run}"
    )
