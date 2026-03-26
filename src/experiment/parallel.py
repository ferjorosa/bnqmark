"""
Parallel execution logic for probabilistic reasoning experiments.

This module provides parallel batch processing capabilities for running
discrete queries in parallel using a master-worker pattern.
"""

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

from src.experiment.batching import create_query_batches
from src.experiment.core import (
    ExperimentType,
    filter_existing_queries,
    run_single_query,
)

logger = logging.getLogger(__name__)


def run_discrete_queries_parallel(
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
    model_name: str,
    run: int,
    experiment_type: ExperimentType,
    task_prompt: str,
    system_prompt: str,
    reasoning_model: bool,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
    openrouter_api_key: str | None = None,
    batch_size: int = 5,
    max_workers: int = 5,
    verbose: bool = False,
) -> None:
    """
    Run discrete queries in parallel batches.

    Each batch is processed sequentially, but queries within a batch
    are processed in parallel.

    Args:
        queries_df: DataFrame of queries.
        bns_df: DataFrame of Bayesian networks.
        model_name: LLM model name (e.g., "openai/gpt-4o", "google/gemini-2.5-flash").
        run: Experiment run number.
        experiment_type: ExperimentType.RAW_REASONING or .CODE_GENERATION.
        task_prompt: Prompt template with {cpts} and {query}.
        system_prompt: System prompt string for LLM.
        reasoning_model: If this is a reasoning model.
        temperature: LLM temperature.
        reasoning_effort: Effort ("low", "medium", "high", or None).
            Defaults to None.
        reasoning_summary: Summary ("auto", "concise", "detailed", or None).
            Defaults to None.
        max_tokens: Max tokens for the response. Defaults to None.
        openrouter_api_key: OpenRouter API key. If not provided, will be read
                            from OPENROUTER_API_KEY environment variable.
        batch_size: Queries per batch (default 5).
        max_workers: Max parallel workers per batch (default 5).
        verbose: Print progress if True.
    """
    # Build mapping from (bn_uuid, naming_strategy) to networks
    bn_map: dict[tuple[str, str], DiscreteBayesianNetwork] = {}
    for _, bn_row in bns_df.iterrows():
        bn_uuid = bn_row["bn_uuid"]
        naming_strategy = bn_row["naming_strategy"]
        bn = pickle.loads(bn_row["bn_pickle"])
        bn_map[(bn_uuid, naming_strategy)] = bn

    queries_df = filter_existing_queries(
        queries_df=queries_df,
        run=run,
        model_name=model_name,
        experiment_type=experiment_type,
        verbose=verbose,
    )

    if len(queries_df) == 0:
        if verbose:
            print("  No new queries to process")
        return

    # Create batches
    batches = create_query_batches(queries_df, batch_size=batch_size)
    total_batches = len(batches)
    total_queries = len(queries_df)

    if verbose:
        print(
            f"  Processing {total_queries} queries in {total_batches} batches "
            f"(batch_size={batch_size}, max_workers={max_workers})"
        )

    # Process each batch sequentially
    successful_queries = 0
    failed_queries = 0

    for batch_idx, batch_df in enumerate(batches, 1):
        if verbose:
            print(
                f"  Processing batch {batch_idx}/{total_batches} "
                f"({len(batch_df)} queries)"
            )

        # Process batch in parallel
        batch_results = run_batch_parallel(
            batch_df=batch_df,
            bn_map=bn_map,
            model_name=model_name,
            run=run,
            experiment_type=experiment_type,
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            reasoning_model=reasoning_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
            openrouter_api_key=openrouter_api_key,
            max_workers=max_workers,
        )

        # Count successes and failures
        for result in batch_results:
            if result.get("error") is None:
                successful_queries += 1
            else:
                failed_queries += 1

        if verbose:
            batch_successful = sum(1 for r in batch_results if r.get("error") is None)
            batch_failed = len(batch_results) - batch_successful
            print(
                f"    Batch {batch_idx} completed: {batch_successful} "
                f"successful, {batch_failed} failed"
            )

    if verbose:
        print(f"\n  Total: {successful_queries} successful, {failed_queries} failed")


def run_batch_parallel(
    batch_df: pd.DataFrame,
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
    openrouter_api_key: str | None,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """
    Process a single batch of queries in parallel using ThreadPoolExecutor.

    Args:
        batch_df: DataFrame containing queries for this batch.
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
        reasoning_effort: Reasoning effort level.
        reasoning_summary: Reasoning summary level.
        max_tokens: Maximum tokens for the response.
        openrouter_api_key: OpenRouter API key. If not provided, will be read
                            from OPENROUTER_API_KEY environment variable.
        max_workers: Maximum number of parallel workers.

    Returns:
        List of result dictionaries, one per query.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries in the batch
        future_to_query = {
            executor.submit(
                run_single_query,
                query_row=query_row,
                bn_map=bn_map,
                model_name=model_name,
                run=run,
                experiment_type=experiment_type,
                task_prompt=task_prompt,
                system_prompt=system_prompt,
                reasoning_model=reasoning_model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                max_tokens=max_tokens,
                openrouter_api_key=openrouter_api_key,
            ): query_row
            for _, query_row in batch_df.iterrows()
        }

        # Collect results as they complete
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                query_row = future_to_query[future]
                query_uuid = query_row["query_uuid"]
                logger.error(
                    f"Error processing query {query_uuid[:8]}: {e}", exc_info=True
                )
                # Create error result
                results.append(
                    {
                        "query_uuid": query_uuid,
                        "error": str(e),
                    }
                )

    return results
