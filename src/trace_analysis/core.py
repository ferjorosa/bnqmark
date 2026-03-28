"""
Core trace analysis logic.

This module contains the core LLM calling and result
processing logic for trace analysis.
"""

import logging
import pickle
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.database.discrete_arithmetic_behavior_analysis_db import (
    insert_arithmetic_analysis,
)
from src.database.discrete_code_generation_behaviour_analysis_db import (
    insert_code_generation_behaviour_analysis,
)
from src.database.discrete_code_generation_strategy_analysis_db import (
    insert_code_generation_strategy_analysis,
)
from src.database.discrete_inference_algorithm_analysis_db import (
    insert_inference_analysis,
)
from src.queries.formatting.format_query_str import (
    format_discrete_cpds,
    format_probability_query,
)
from src.trace_analysis.pydantic_models.arithmetic_behaviour_analysis import (
    ArithmeticBehaviorAnalysis,
)
from src.trace_analysis.pydantic_models.code_generation_behaviour_analysis import (
    CodeGenerationBehaviourAnalysis,
)
from src.trace_analysis.pydantic_models.code_strategy_analysis import (
    CodeStrategyAnalysis,
)
from src.trace_analysis.pydantic_models.inference_algorithm_analysis import (
    InferenceAlgorithmAnalysis,
)
from src.utils.llm_utils import run_llm_call

logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis available."""

    INFERENCE_ALGORITHM = "inference_algorithm"
    ARITHMETIC_BEHAVIOR = "arithmetic_behavior"
    CODE_STRATEGY = "code_strategy"
    CODE_GENERATION_BEHAVIOUR = "code_generation_behaviour"


def _get_parser_for_analysis_type(analysis_type: AnalysisType) -> PydanticOutputParser:
    """
    Get the Pydantic parser for a given analysis type.

    Args:
        analysis_type: The type of analysis to perform.

    Returns:
        PydanticOutputParser configured for the appropriate model.

    Raises:
        ValueError: If analysis_type is unknown.
    """
    if analysis_type == AnalysisType.INFERENCE_ALGORITHM:
        return PydanticOutputParser(pydantic_object=InferenceAlgorithmAnalysis)
    elif analysis_type == AnalysisType.ARITHMETIC_BEHAVIOR:
        return PydanticOutputParser(pydantic_object=ArithmeticBehaviorAnalysis)
    elif analysis_type == AnalysisType.CODE_STRATEGY:
        return PydanticOutputParser(pydantic_object=CodeStrategyAnalysis)
    elif analysis_type == AnalysisType.CODE_GENERATION_BEHAVIOUR:
        return PydanticOutputParser(pydantic_object=CodeGenerationBehaviourAnalysis)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def insert_analysis_result(
    analysis_type: AnalysisType,
    query_uuid: str,
    naming_strategy: str,
    run: int,
    model_name: str,
    experiment_type: str,
    analysis_object: Any,
    analysis_duration_ms: float,
    llm_call_info: dict[str, Any],
) -> None:
    """
    Insert the analysis result into the appropriate database table.

    Args:
        analysis_type: The type of analysis performed.
        query_uuid: UUID of the query.
        naming_strategy: Naming strategy used.
        run: Run number.
        model_name: Model name.
        experiment_type: Experiment type.
        analysis_object: The parsed analysis result (Pydantic model).
        analysis_duration_ms: Duration of the analysis.
        llm_call_info: Metadata about the LLM call.
    """
    if analysis_type == AnalysisType.INFERENCE_ALGORITHM:
        insert_inference_analysis(
            query_uuid=query_uuid,
            naming_strategy=naming_strategy,
            run=run,
            model_name=model_name,
            experiment_type=experiment_type,
            analysis_object=analysis_object,
            analysis_duration_ms=analysis_duration_ms,
            **llm_call_info,
        )
    elif analysis_type == AnalysisType.ARITHMETIC_BEHAVIOR:
        insert_arithmetic_analysis(
            query_uuid=query_uuid,
            naming_strategy=naming_strategy,
            run=run,
            model_name=model_name,
            experiment_type=experiment_type,
            analysis_object=analysis_object,
            analysis_duration_ms=analysis_duration_ms,
            **llm_call_info,
        )
    elif analysis_type == AnalysisType.CODE_STRATEGY:
        insert_code_generation_strategy_analysis(
            query_uuid=query_uuid,
            naming_strategy=naming_strategy,
            run=run,
            model_name=model_name,
            experiment_type=experiment_type,
            analysis_object=analysis_object,
            analysis_duration_ms=analysis_duration_ms,
            **llm_call_info,
        )
    elif analysis_type == AnalysisType.CODE_GENERATION_BEHAVIOUR:
        insert_code_generation_behaviour_analysis(
            query_uuid=query_uuid,
            naming_strategy=naming_strategy,
            run=run,
            model_name=model_name,
            experiment_type=experiment_type,
            analysis_object=analysis_object,
            analysis_duration_ms=analysis_duration_ms,
            **llm_call_info,
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def _prepare_analysis_prompt_template(
    system_prompt: str,
    task_prompt: str,
    response_reasoning_summary: str,
    response: str,
    cpts: str,
    query: str,
    parser: PydanticOutputParser,
) -> tuple[PromptTemplate, dict[str, Any], str]:
    """
    Prepare the prompt template for analysis.

    Args:
        system_prompt: System prompt string.
        task_prompt: Task prompt template string.
        response_reasoning_summary: The reasoning summary to analyze.
        response: The final response to analyze.
        cpts: The CPTs string.
        query: The query string.
        parser: Output parser for the Pydantic model.

    Returns:
        Tuple of (full_prompt_template, parameters, full_prompt)
    """
    combined_prompt = f"{system_prompt}\n\n{task_prompt}"

    full_prompt_template = PromptTemplate(
        template=combined_prompt,
        input_variables=["response_reasoning_summary", "response", "cpts", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    parameters = {
        "response_reasoning_summary": response_reasoning_summary,
        "response": response,
        "cpts": cpts,
        "query": query,
    }

    full_prompt = full_prompt_template.format(**parameters)

    return full_prompt_template, parameters, full_prompt


def run_single_analysis(
    row: pd.Series,
    system_prompt: str,
    task_prompt: str,
    model_name: str,
    analysis_type: AnalysisType,
    temperature: float = 0.0,
    openrouter_api_key: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, float, dict[str, Any]]:
    """
    Run analysis on a single trace and return result, duration, and metadata.

    Args:
        row: DataFrame row containing experiment data.
        system_prompt: System prompt string.
        task_prompt: Task prompt template string.
        model_name: Model name (e.g., "openai/gpt-4o", "google/gemini-2.5-flash").
        analysis_type: Type of analysis to perform.
        temperature: Temperature for the analysis LLM.
        openrouter_api_key: OpenRouter API key. If not provided, will be read
            from OPENROUTER_API_KEY environment variable.
        reasoning_effort: Reasoning effort level. Values: "xhigh",
            "high", "medium", "low", "minimal", "none".
        reasoning_summary: Reasoning summary level. Values: "auto",
            "concise", "detailed".
        max_tokens: Maximum tokens for the response.

    Returns:
        Tuple containing:
        - content: The parsed analysis object (Pydantic model)
        - duration: Duration of the analysis in ms
        - llm_call_info: Metadata about the LLM call
    """
    parser = _get_parser_for_analysis_type(analysis_type)

    response_val = row.get("response", "")
    response_reasoning_summary_val = row.get("response_reasoning_summary", "")

    bn = pickle.loads(row["bn_pickle"])
    cpts = bn.get_cpds()
    cpts_val = format_discrete_cpds(cpts)

    target = row["target"]
    evidence = row["evidence"]
    query_val = format_probability_query(target, evidence=evidence)

    full_prompt_template, parameters, full_prompt_str = (
        _prepare_analysis_prompt_template(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            response_reasoning_summary=response_reasoning_summary_val,
            response=response_val,
            cpts=cpts_val,
            query=query_val,
            parser=parser,
        )
    )

    start_time = time.time()
    started_at = datetime.now(UTC).isoformat()

    try:
        (
            content,
            usage_metadata,
            response_reasoning,
            response_metadata,
        ) = run_llm_call(
            prompt_template=full_prompt_template,
            model_name=model_name,
            parameters=parameters,
            output_parser=parser,
            temperature=temperature,
            openrouter_api_key=openrouter_api_key,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

        duration = (time.time() - start_time) * 1000
        finished_at = datetime.now(UTC).isoformat()

        llm_call_info = {
            "llm_call_full_prompt": full_prompt_str,
            "llm_call_reasoning": response_reasoning,
            "llm_call_metadata": response_metadata,
            "llm_call_model": model_name,
            "llm_call_input_tokens": usage_metadata.get("input_tokens")
            if usage_metadata
            else None,
            "llm_call_output_tokens": usage_metadata.get("output_tokens")
            if usage_metadata
            else None,
            "llm_call_usage_metadata": usage_metadata,
            "llm_call_temperature": temperature,
            "llm_call_started_at": started_at,
            "llm_call_finished_at": finished_at,
        }

        if content:
            insert_analysis_result(
                analysis_type=analysis_type,
                query_uuid=row["query_uuid"],
                naming_strategy=row["naming_strategy"],
                run=row["run"],
                model_name=row["model_name"],
                experiment_type=row["experiment_type"],
                analysis_object=content,
                analysis_duration_ms=duration,
                llm_call_info=llm_call_info,
            )

        return content, duration, llm_call_info

    except Exception as e:
        logger.error(f"Analysis failed for {row['query_uuid']}: {e}")
        return None, 0.0, {}


def filter_existing_analyses(
    experiments_df: pd.DataFrame,
    existing_analyses: set,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter out experiments that have already been analyzed.

    Args:
        experiments_df: DataFrame containing experiments to analyze.
        existing_analyses: Set of identifiers for existing analyses.
                           Typically (query_uuid, naming_strategy,
                           run, model_name, experiment_type).
        verbose: If True, print filtering summary.

    Returns:
        Filtered DataFrame containing only experiments that need analysis.
    """
    total_experiments = len(experiments_df)

    experiments_df["key"] = experiments_df.apply(
        lambda row: (
            row["query_uuid"],
            row["naming_strategy"],
            row["run"],
            row["model_name"],
            row["experiment_type"],
        ),
        axis=1,
    )

    filtered_df = experiments_df[~experiments_df["key"].isin(existing_analyses)].copy()
    filtered_df = filtered_df.drop(columns=["key"])

    if verbose:
        skipped_count = total_experiments - len(filtered_df)
        print(
            f"  Processing {len(filtered_df)}/{total_experiments} traces "
            f"(skipping {skipped_count} existing analyses)",
        )

    return filtered_df
