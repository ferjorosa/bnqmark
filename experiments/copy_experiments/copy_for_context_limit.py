#!/usr/bin/env python3
"""
Copy experiment rows from an anchor model to a target model.

Marks them as context limit exceeded. This utility script queries the
discrete_experiments table for rows from an anchor model where input_tokens
exceeds a threshold, and inserts corresponding rows for a target model with
llm_probability set to -1000 to indicate that the context length would be
exceeded.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.database import get_connection, query_db
from src.database.discrete_experiments_db import TABLE_NAME

# =============================================================================
# Configuration - Modify these values as needed
# =============================================================================

# Source model that has all experiments completed (anchor)
# Using grok-4.20 as it has 2M context and successfully handled all queries
ANCHOR_MODEL = "x-ai/grok-4.20"

# Target model to receive the copied rows
# This model will be marked as having context limit issues
TARGET_MODEL = "anthropic/claude-sonnet-4.6"

# Token threshold - rows with input_tokens > this value will be copied
# minimax-m2.7 has 200k context
TOKEN_THRESHOLD = 1000000

# Model configuration for the target model
# These values should match the model's configuration in experiments.yaml
TARGET_REASONING_MODEL = True
TARGET_TEMPERATURE = 0.0
TARGET_REASONING_EFFORT = "xhigh"
TARGET_REASONING_SUMMARY = "detailed"

# Value to indicate context limit exceeded
CONTEXT_LIMIT_MARKER = -1000

# =============================================================================


def fetch_anchor_rows(anchor_model: str, token_threshold: int) -> list[dict]:
    """
    Fetch rows from the anchor model where input_tokens exceeds threshold.

    Returns list of dictionaries with the fields needed for insertion.
    """
    query = f"""
        SELECT query_uuid, naming_strategy, run, experiment_type,
               full_prompt, input_tokens
        FROM {TABLE_NAME}
        WHERE model_name = ?
          AND input_tokens > ?
    """

    df = query_db(query, [anchor_model, token_threshold])

    if df.empty:
        return []

    return df.to_dict("records")


def build_target_rows(
    anchor_rows: list[dict],
    target_model: str,
    reasoning_model: bool,
    temperature: float,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
) -> list[tuple]:
    """
    Construct parameter tuples for inserting target model rows.

    Returns list of tuples matching the INSERT query parameters.
    """
    now = datetime.now().isoformat()
    params_list = []

    for row in anchor_rows:
        params = (
            row["query_uuid"],
            row["naming_strategy"],
            row["run"],
            row["experiment_type"],
            row["full_prompt"],
            "CONTEXT_LIMIT_EXCEEDED",  # response
            None,  # response_reasoning_summary
            None,  # response_metadata
            target_model,
            reasoning_model,
            reasoning_effort,
            reasoning_summary,
            row["input_tokens"],
            0,  # output_tokens
            None,  # usage_metadata
            temperature,
            CONTEXT_LIMIT_MARKER,  # llm_probability
            None,  # code_followed_formatting_instructions
            None,  # code_pgmpy_library_fix
            None,  # code_exception_type
            None,  # code_exception_message
            now,  # started_at
            now,  # finished_at
        )
        params_list.append(params)

    return params_list


def insert_target_rows(params_list: list[tuple]) -> int:
    """
    Insert target model rows in a single transaction.

    Uses INSERT OR IGNORE to skip rows that would violate the unique constraint.
    Returns the number of rows inserted.
    """
    if not params_list:
        return 0

    query = f"""
        INSERT OR IGNORE INTO {TABLE_NAME} (
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

    conn = get_connection()
    cursor = conn.cursor()

    inserted = 0
    try:
        with conn:
            cursor.executemany(query, params_list)
            inserted = cursor.rowcount
    except Exception as e:
        print(f"Error during insertion: {e}")
        raise

    return inserted


def main():
    """Main execution function."""
    print("=" * 60)
    print("Copy Experiments - Context Limit Marker")
    print("=" * 60)
    print()

    print(f"Anchor model: {ANCHOR_MODEL}")
    print(f"Target model: {TARGET_MODEL}")
    print(f"Token threshold: {TOKEN_THRESHOLD}")
    print()

    # Fetch rows from anchor model that exceed the threshold
    print("Fetching anchor model rows...")
    anchor_rows = fetch_anchor_rows(ANCHOR_MODEL, TOKEN_THRESHOLD)

    if not anchor_rows:
        print(f"No rows found for {ANCHOR_MODEL} with input_tokens > {TOKEN_THRESHOLD}")
        return

    print(f"Found {len(anchor_rows)} rows to copy")
    print()

    # Build target row parameters
    print("Building target row parameters...")
    target_params = build_target_rows(
        anchor_rows,
        TARGET_MODEL,
        TARGET_REASONING_MODEL,
        TARGET_TEMPERATURE,
        TARGET_REASONING_EFFORT,
        TARGET_REASONING_SUMMARY,
    )
    print(f"Built {len(target_params)} target row parameter sets")
    print()

    # Insert target rows
    print("Inserting target model rows...")
    inserted = insert_target_rows(target_params)
    print(
        f"Inserted {inserted} rows (skipped {len(target_params) - inserted} duplicates)"
    )
    print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
