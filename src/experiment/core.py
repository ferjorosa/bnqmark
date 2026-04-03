"""
Core experiment logic for probabilistic reasoning experiments.

This module contains the core LLM calling and result processing logic,
isolated from batching strategies and high-level orchestration.
"""

import logging
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

from src.database.discrete_experiments_db import (
    get_existing_experiment_identifiers,
    insert_experiment_row,
)
from src.queries.formatting.format_query_str import (
    format_discrete_cpds,
    format_probability_query,
)
from src.utils.error_utils import is_token_limit_error
from src.utils.llm import run_llm_call
from src.utils.tiktoken_utils import count_input_tokens

logger = logging.getLogger(__name__)


class ExperimentType(str, Enum):
    """Types of experiments available."""

    RAW_REASONING = "raw_reasoning"
    CODE_GENERATION = "code_generation"


def _prepare_prompt(
    system_prompt: str,
    task_prompt: str,
    cpts_str: str,
    query_str: str,
) -> str:
    """
    Prepare the full prompt string by substituting placeholders.

    Args:
        system_prompt: System prompt string (for all experiment types).
        task_prompt: Task prompt template with {cpts} and {query} placeholders.
            For raw_reasoning, this is prompt_base (direct answer).
            For code_generation, this is prompt_base_code (Python code).
        cpts_str: Formatted CPTs string.
        query_str: Formatted query string.

    Returns:
        Full formatted prompt string.
    """
    # Combine system and task prompts into a single prompt template
    combined_prompt = f"{system_prompt}\n\n{task_prompt}"

    # Substitute placeholders directly
    full_prompt = combined_prompt.replace("{cpts}", cpts_str).replace(
        "{query}", query_str
    )

    return full_prompt


def run_single_query(
    query_row: pd.Series,
    bn_map: dict[tuple[str, str], DiscreteBayesianNetwork],
    model_name: str,
    run: int,
    experiment_type: ExperimentType,
    task_prompt: str,
    system_prompt: str,
    reasoning_model: bool,
    temperature: float,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
    max_tokens: int | None,
    openrouter_api_key: str,
) -> dict[str, Any]:
    """
    Process a single query: run LLM call and insert to database.

    This function handles the complete flow for a single query:
    - Extracts query information from the row
    - Gets BN and formats query/CPTs
    - Calls the LLM
    - Inserts result to database

    Args:
        query_row: Single row from queries_df containing query information.
        bn_map: Mapping from (bn_uuid, naming_strategy) to BN objects.
        model_name: Name of the model (e.g., "openai/gpt-4o",
            "google/gemini-2.5-flash").
        run: Run number for this experiment.
        experiment_type: Type of experiment ("raw_reasoning" or "code_generation").
        task_prompt: Task prompt template string with {cpts} and {query} placeholders.
                    For raw_reasoning: prompt_base (asks for direct probability answer).
                    For code_generation: prompt_base_code (asks for Python code).
        system_prompt: System prompt string (same for all experiment types).
        reasoning_model: Whether this is a reasoning model.
        temperature: Temperature setting for the LLM.
        reasoning_effort: Reasoning effort level. Values: "xhigh", "high",
            "medium", "low", "minimal", "none".
        reasoning_summary: Reasoning summary level. Values: "auto",
            "concise", "detailed".
        max_tokens: Maximum tokens for the response.
        openrouter_api_key: OpenRouter API key. If not provided, will be read
                            from OPENROUTER_API_KEY environment variable.

    Returns:
        Dictionary with query_uuid and error (if any).
    """
    query_uuid = query_row["query_uuid"]
    bn_uuid = query_row["bn_uuid"]
    naming_strategy = query_row["naming_strategy"]
    target = query_row["target"]
    evidence = query_row["evidence"]

    try:
        # Get BN and format query/CPTs
        bn = bn_map[(bn_uuid, naming_strategy)]
        cpts = bn.get_cpds()

        query_str = format_probability_query(target, evidence=evidence)
        cpts_str = format_discrete_cpds(cpts)

        # Capture timestamp before running the query
        started_at = datetime.now(UTC).isoformat()

        # Prepare prompt for potential error handling
        full_prompt = _prepare_prompt(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            cpts_str=cpts_str,
            query_str=query_str,
        )

        # Initialize variables for error handling
        error_message = None
        response = None
        llm_probability = None
        usage_metadata = None
        response_reasoning_summary = None
        response_metadata = None
        input_tokens = None
        output_tokens = None

        # Run the LLM call
        try:
            (
                response,
                llm_probability,
                full_prompt,
                usage_metadata,
                response_reasoning_summary,
                response_metadata,
            ) = _run_single_discrete_experiment(
                full_prompt=full_prompt,
                query_str=query_str,
                model_name=model_name,
                run=run,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                max_tokens=max_tokens,
                openrouter_api_key=openrouter_api_key,
            )

            # Extract input_tokens and output_tokens from usage_metadata
            input_tokens = (
                usage_metadata.get("prompt_tokens") if usage_metadata else None
            )
            output_tokens = (
                usage_metadata.get("completion_tokens") if usage_metadata else None
            )

        except RuntimeError as e:
            # Check if this is specifically a token limit error
            if is_token_limit_error(e):
                # Estimate input tokens using tiktoken
                input_tokens = count_input_tokens(full_prompt)
                logger.warning(
                    f"Input exceeds context length for query {query_uuid[:8]}: "
                    f"estimated {input_tokens} input tokens"
                )
                error_message = f"Input exceeds context length (estimated {input_tokens} tokens using tiktoken o200k_base)"  # noqa: E501
                output_tokens = 0
                llm_probability = -1000.0
                response = f"LLM call failed: {error_message}"
                usage_metadata = None
                response_reasoning_summary = None
                response_metadata = None
            else:
                # For other errors, log and return error
                error_message = str(e)
                # Don't show full traceback for 400 errors
                # (often context length issues)
                if "Error code: 400" in error_message:
                    logger.error(
                        f"LLM call error (400) for query {query_uuid[:8]}",
                        exc_info=False,
                    )
                else:
                    logger.error(
                        f"LLM call unexpected error in query {query_uuid[:8]}: {e}",
                        exc_info=True,
                    )
                return {"query_uuid": query_uuid, "error": error_message}

        # Capture timestamp after running the query (or error)
        finished_at = datetime.now(UTC).isoformat()

        # Insert experiment row into database (
        # single call for both success and token limit error cases)
        insert_experiment_row(
            query_uuid=query_uuid,
            naming_strategy=naming_strategy,
            run=run,
            experiment_type=experiment_type.value,
            full_prompt=full_prompt,
            response=response,
            model_name=model_name,
            reasoning_model=reasoning_model,
            openai_reasoning_effort=reasoning_effort,
            openai_reasoning_summary=reasoning_summary,
            response_reasoning_summary=response_reasoning_summary,
            response_metadata=response_metadata,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_metadata=usage_metadata,
            temperature=temperature,
            started_at=started_at,
            finished_at=finished_at,
            llm_probability=llm_probability,
        )

        return {"query_uuid": query_uuid, "error": error_message}

    except Exception as e:
        logger.error(f"Error processing query {query_uuid[:8]}: {e}", exc_info=True)
        return {"query_uuid": query_uuid, "error": str(e)}


def _run_single_discrete_experiment(
    full_prompt: str,
    query_str: str,
    model_name: str,
    run: int,
    openrouter_api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, float | None, str, dict | None, str | None, dict | None]:
    """
    Run a single LLM call for a probabilistic reasoning query.

    Calls the LLM with the full prompt string.

    Args:
        full_prompt: Complete formatted prompt string.
        query_str: Formatted query string (for logging).
        model_name: Model name for the LLM (e.g., "openai/gpt-4o").
        run: Run number for this experiment.
        temperature: LLM temperature. Defaults to 0.0.
        openrouter_api_key: OpenRouter API key.
        reasoning_effort: Reasoning effort. Values: "xhigh", "high",
            "medium", "low", "minimal", "none". Defaults to None.
        reasoning_summary: Reasoning summary. Values: "auto", "concise",
            "detailed". Defaults to None.
        max_tokens: Max tokens for the response. Defaults to None.

    Returns:
        Tuple: (
            response_content: str,
            llm_probability: float or None,
            full_prompt: str,
            usage_metadata: dict or None,
            response_reasoning_summary: str or None,
            response_metadata: dict or None,
        )

    Raises:
        Exception: Any exception from run_llm_call will be propagated.
    """
    logger.debug(f"Running query: {query_str} with model: {model_name}, run: {run}")

    # Run the LLM call
    (
        response_content,
        usage_metadata,
        response_reasoning_summary,
        response_metadata,
    ) = run_llm_call(
        prompt=full_prompt,
        model_name=model_name,
        openrouter_api_key=openrouter_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )

    # Parse probability from response
    llm_probability = _parse_probability_from_response(response_content)

    in_tokens = usage_metadata.get("input_tokens") if usage_metadata else "N/A"
    out_tokens = usage_metadata.get("output_tokens") if usage_metadata else "N/A"
    logger.debug(
        f"Query completed: {query_str}, probability: {llm_probability}, "
        f"tokens: {in_tokens} in, {out_tokens} out"
    )

    return (
        response_content,
        llm_probability,
        full_prompt,
        usage_metadata,
        response_reasoning_summary,
        response_metadata,
    )


def filter_existing_queries(
    queries_df: pd.DataFrame,
    run: int,
    model_name: str,
    experiment_type: ExperimentType,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter out queries that already exist in the database.

    Args:
        queries_df: DataFrame containing queries.
        run: Run number for this experiment.
        model_name: Model name string.
        experiment_type: Type of experiment
        verbose: If True, print filtering summary.

    Returns:
        Filtered DataFrame containing only queries that don't exist yet.
    """
    total_queries = len(queries_df)

    # Get all existing experiment identifiers
    existing = get_existing_experiment_identifiers()

    # Filter out existing records
    queries_df["key"] = queries_df.apply(
        lambda row: (
            row["query_uuid"],
            row["naming_strategy"],
            run,
            model_name,
            experiment_type.value,
        ),
        axis=1,
    )
    filtered_df = queries_df[~queries_df["key"].isin(existing)].copy()
    filtered_df = filtered_df.drop(columns=["key"])

    if verbose:
        skipped_count = total_queries - len(filtered_df)
        print(
            f"  Processing {len(filtered_df)}/{total_queries} queries "
            f"(skipping {skipped_count} existing)"
        )

    return filtered_df


def _parse_probability_from_response(response: str) -> float | None:
    """
    Parse probability value from LLM response.

    Looks for pattern: "Final Answer: P(...) = 0.1234"

    Args:
        response: LLM response string

    Returns:
        Extracted probability as float, or None if not found
    """
    # Pattern to match "Final Answer: P(...) = 0.1234"
    pattern = r"Final Answer:\s*P\([^)]+\)\s*=\s*([0-9]+\.?[0-9]*)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
