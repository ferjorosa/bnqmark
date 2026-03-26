"""LLM utilities using only LangChain, tuned for OpenRouter."""

import os

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def langchain_get_llm(
    model_name: str,
    openrouter_api_key: str | None = None,
    inference_server_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI LLM instance configured for OpenRouter.

    This function creates a LangChain ChatOpenAI instance that connects to
    OpenRouter's OpenAI-compatible API.

    Args:
        model_name (str): The model name to use (e.g., "openai/gpt-4o",
            "google/gemini-2.5-flash").
        openrouter_api_key (Optional[str]): OpenRouter API key. If not provided,
            will be read from OPENROUTER_API_KEY environment variable.
        inference_server_url (str): The base URL of the OpenRouter server.
            Defaults to "https://openrouter.ai/api/v1".
        temperature (float): Controls randomness in model responses. Defaults to 0.0.
        max_tokens (Optional[int]): Maximum tokens for the response.
        top_p (Optional[float]): Adjusts the diversity of the model's responses.
        seed (Optional[int]): The seed for the random number generator.
        verbose (bool): Enables verbose logging if True. Defaults to False.
        reasoning_effort (str, optional): Controls the effort level for
            reasoning models. Values: "xhigh", "high", "medium", "low",
            "minimal", "none". Requires reasoning_summary to be set.
        reasoning_summary (str, optional): Controls the summary style for
            reasoning models. Values: "auto", "concise", "detailed".
            Requires reasoning_effort to be set.

    Returns:
        ChatOpenAI: A ChatOpenAI instance configured for the specified OpenRouter model.

    Raises:
        ValueError: If no API key is provided and OPENROUTER_API_KEY env var is not set.
    """
    # Get API key from environment if not provided
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not provided. Either pass openrouter_api_key parameter "
            "or set OPENROUTER_API_KEY environment variable."
        )

    # Build reasoning config if both effort and summary are provided
    reasoning = (
        {
            "effort": reasoning_effort,
            "summary": reasoning_summary,
        }
        if reasoning_effort is not None and reasoning_summary is not None
        else None
    )

    chat_llm = ChatOpenAI(
        model=model_name,  # ty: ignore
        openai_api_key=api_key,  # ty: ignore
        openai_api_base=inference_server_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=verbose,
        seed=seed,
        reasoning=reasoning,
    )

    return chat_llm


def run_llm_call(
    prompt_template: PromptTemplate,
    model_name: str,
    parameters: dict,
    output_parser: BaseOutputParser | None = None,
    temperature: float = 0.0,
    openrouter_api_key: str | None = None,
    inference_server_url: str = "https://openrouter.ai/api/v1",
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
) -> tuple[str, dict | None, str | None, dict | None]:
    """
    Run an LLM call with a given prompt and input using OpenRouter.

    Args:
        prompt_template (PromptTemplate): The prompt template to use.
        model_name (str): The model name to use (e.g., "openai/gpt-4o",
            "google/gemini-2.5-flash").
        parameters (dict): The parameters for the prompt.
        output_parser (Optional[BaseOutputParser]): The output parser to use.
        temperature (float): Temperature setting for the LLM. Defaults to 0.0.
        openrouter_api_key (Optional[str]): OpenRouter API key. If not provided,
            will be read from OPENROUTER_API_KEY environment variable.
        inference_server_url (str): OpenRouter API base URL. Defaults to
            "https://openrouter.ai/api/v1".
        max_tokens (Optional[int]): Maximum tokens for the response.
        reasoning_effort (Optional[str]): Reasoning effort level. Values:
            "xhigh", "high", "medium", "low", "minimal", "none".
            Requires reasoning_summary to be set.
        reasoning_summary (Optional[str]): Reasoning summary level. Values:
            "auto", "concise", "detailed". Requires reasoning_effort to be
            set.

    Returns:
        tuple[str, Optional[dict], Optional[str], Optional[dict]]: Tuple of
        (content, usage_metadata, response_reasoning_summary, response_metadata) where:
        - content: The parsed or raw output of the llm call
        - usage_metadata: Dictionary containing usage metadata including
          input_tokens and output_tokens (or None)
        - response_reasoning_summary: Reasoning content returned from the
          model (or None)
        - response_metadata: Dictionary containing response metadata (or None)
    """
    # Get LLM instance
    llm = langchain_get_llm(
        model_name=model_name,
        openrouter_api_key=openrouter_api_key,
        inference_server_url=inference_server_url,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )

    chain = prompt_template | llm

    response = chain.invoke(parameters)

    response_reasoning_summary = None
    response_metadata = None

    # Handle OpenRouter response format
    response_content = response.content[0]["text"] if len(response.content) > 0 else ""  # ty: ignore

    # Extract reasoning content if available (OpenRouter passes through
    # reasoning from providers)
    if (
        "reasoning" in response.additional_kwargs
        and "content" in response.additional_kwargs["reasoning"]
    ):
        reasoning_content = response.additional_kwargs["reasoning"]["content"]
        if reasoning_content and len(reasoning_content) > 0:
            response_reasoning_summary = reasoning_content[0]["text"]

    response_metadata = response.response_metadata
    usage_metadata = response.usage_metadata

    if output_parser:
        content = output_parser.parse(response_content)
    else:
        content = response_content

    return content, usage_metadata, response_reasoning_summary, response_metadata  # ty: ignore
