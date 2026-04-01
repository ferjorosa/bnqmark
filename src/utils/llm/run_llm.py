"""Main LLM runner using OpenAI library directly."""

from openai import OpenAI

from .parser_basic import parse_basic_response


def run_llm_call(
    prompt: str,
    model_name: str,
    openrouter_api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    provider_preferences: list[str] | None = None,
) -> tuple[str, dict | None, str | None, dict | None]:
    """
    Run an LLM call with a given prompt using OpenRouter directly.

    Args:
        prompt: The prompt text to send to the model.
        model_name: The model name to use (e.g., "google/gemini-3.1-pro",
            "z-ai/glm-5-20260211").
        openrouter_api_key: OpenRouter API key.
        base_url: OpenRouter API base URL. Defaults to
            "https://openrouter.ai/api/v1".
        temperature: Temperature setting for the LLM. Defaults to 0.0.
        max_tokens: Maximum tokens for the response.
        top_p: Top-p sampling parameter.
        seed: Seed for deterministic sampling.
        reasoning_effort: Reasoning effort level. Values:
            "xhigh", "high", "medium", "low", "minimal", "none".
            Requires reasoning_summary to be set.
        reasoning_summary: Reasoning summary level. Values:
            "auto", "concise", "detailed". Requires reasoning_effort to be
            set.
        provider_preferences: List of provider names in
            priority order for OpenRouter routing (e.g., ["openai", "together"]).
            If not provided or empty, no provider preference is sent.

    Returns:
        Tuple of (content, usage_metadata, response_reasoning_summary,
        response_metadata) where:
        - content: The output text from the model
        - usage_metadata: Dictionary containing usage metadata including
          input_tokens and output_tokens (or None)
        - response_reasoning_summary: Reasoning content returned from the
          model (or None)
        - response_metadata: Dictionary containing response metadata (or None)
    """
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url=base_url,
    )

    # Build the messages list
    messages = [{"role": "user", "content": prompt}]

    # Build extra_body with provider preferences and reasoning config
    extra_body: dict = {}

    if provider_preferences:
        extra_body["provider"] = {
            "order": provider_preferences,
            "allow_fallbacks": False,
            "require_parameters": True,
        }

    # Build reasoning config if both effort and summary are provided
    if reasoning_effort is not None and reasoning_summary is not None:
        extra_body["reasoning"] = {
            "effort": reasoning_effort,
            "summary": reasoning_summary,
            "exclude": False,  # Explicitly include reasoning in response
        }

    # Build request parameters
    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    if top_p is not None:
        request_params["top_p"] = top_p
    if seed is not None:
        request_params["seed"] = seed
    if extra_body:
        request_params["extra_body"] = extra_body

    # Make the API call
    response = client.chat.completions.create(**request_params)

    # Parse the response
    response_content, response_reasoning_summary, usage_metadata, response_metadata = (
        parse_basic_response(response)
    )

    return (
        response_content,
        usage_metadata,
        response_reasoning_summary,
        response_metadata,
    )
