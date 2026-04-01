"""
Parallel execution logic for trace analysis.

This module provides parallel batch processing capabilities for analyzing traces
using a master-worker pattern.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from src.trace_analysis.batching import create_experiment_batches
from src.trace_analysis.core import (
    AnalysisType,
    filter_existing_analyses,
    run_single_analysis,
)

logger = logging.getLogger(__name__)


def run_trace_analysis_parallel(
    experiments_df: pd.DataFrame,
    existing_analyses: set,
    model_name: str,
    analysis_type: AnalysisType,
    system_prompt: str,
    task_prompt: str,
    openrouter_api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
    batch_size: int = 5,
    max_workers: int = 5,
    verbose: bool = False,
) -> None:
    """
    Run trace analysis in parallel batches.

    Args:
        experiments_df: DataFrame containing experiments to analyze.
        existing_analyses: Set of existing analysis identifiers to skip.
        model_name: Model name for the analysis LLM (e.g., "openai/gpt-4o").
        analysis_type: Type of analysis to perform.
        system_prompt: System prompt string.
        task_prompt: Task prompt template string.
        openrouter_api_key: OpenRouter API key.
        temperature: Temperature setting for the analysis LLM.
        reasoning_effort: Reasoning effort level. Values: "xhigh", "high",
            "medium", "low", "minimal", "none".
        reasoning_summary: Reasoning summary level. Values: "auto",
            "concise", "detailed".
        max_tokens: Maximum tokens for the response.
        batch_size: Number of traces per batch.
        max_workers: Maximum number of parallel workers per batch.
        verbose: If True, print progress.
    """
    experiments_df = filter_existing_analyses(
        experiments_df=experiments_df,
        existing_analyses=existing_analyses,
        verbose=verbose,
    )

    if len(experiments_df) == 0:
        if verbose:
            print("  No new traces to analyze")
        return

    batches = create_experiment_batches(experiments_df, batch_size=batch_size)
    total_batches = len(batches)
    total_experiments = len(experiments_df)

    if verbose:
        print(
            f"  Processing {total_experiments} traces in {total_batches} batches "
            f"(batch_size={batch_size}, max_workers={max_workers})"
        )

    successful_analyses = 0
    failed_analyses = 0

    for batch_idx, batch_df in enumerate(batches, 1):
        if verbose:
            print(
                f"  Processing batch {batch_idx}/{total_batches} "
                f"({len(batch_df)} traces)"
            )

        batch_results = run_batch_parallel(
            batch_df=batch_df,
            model_name=model_name,
            analysis_type=analysis_type,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            temperature=temperature,
            openrouter_api_key=openrouter_api_key,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
            max_workers=max_workers,
        )

        for result in batch_results:
            if result.get("error") is None:
                successful_analyses += 1
            else:
                failed_analyses += 1

        if verbose:
            batch_successful = sum(1 for r in batch_results if r.get("error") is None)
            batch_failed = len(batch_results) - batch_successful
            print(
                f"    Batch {batch_idx} completed: {batch_successful} successful,",
                f"{batch_failed} failed",
            )

    if verbose:
        print(f"\n  Total: {successful_analyses} successful, {failed_analyses} failed")


def run_batch_parallel(
    batch_df: pd.DataFrame,
    model_name: str,
    analysis_type: AnalysisType,
    system_prompt: str,
    task_prompt: str,
    openrouter_api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """
    Process a single batch of traces in parallel using ThreadPoolExecutor.

    Args:
        batch_df: DataFrame containing traces for this batch.
        model_name: Model name for the analysis LLM (e.g., "openai/gpt-4o").
        analysis_type: Type of analysis.
        system_prompt: System prompt string.
        task_prompt: Task prompt template string.
        temperature: Temperature setting.
        openrouter_api_key: OpenRouter API key.
        reasoning_effort: Reasoning effort level.
        reasoning_summary: Reasoning summary level.
        max_tokens: Maximum tokens for the response.
        max_workers: Maximum number of parallel workers.

    Returns:
        List of result dictionaries.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trace = {
            executor.submit(
                run_single_analysis,
                row=row,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                model_name=model_name,
                analysis_type=analysis_type,
                temperature=temperature,
                openrouter_api_key=openrouter_api_key,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                max_tokens=max_tokens,
            ): row
            for _, row in batch_df.iterrows()
        }

        for future in as_completed(future_to_trace):
            try:
                result, _, _ = future.result()
                if result:
                    results.append({"status": "success"})
                else:
                    results.append({"error": "Analysis returned None"})
            except Exception as e:
                row = future_to_trace[future]
                query_uuid = row["query_uuid"]
                logger.error(
                    f"Error analyzing trace {query_uuid[:8]}: {e}", exc_info=True
                )
                results.append(
                    {
                        "query_uuid": query_uuid,
                        "error": str(e),
                    }
                )

    return results
