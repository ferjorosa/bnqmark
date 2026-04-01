"""
Basic response parser for OpenRouter models using message.reasoning format.

Handles parsing of models that return reasoning in the simple message.reasoning format:
- OpenAI GPT-5
- Anthropic Claude (Sonnet)
- Z-AI GLM-5
"""


def parse_basic_response(response) -> tuple[str, str | None, dict | None, dict | None]:
    """
    Parse response for models using the basic message.reasoning format.

    These models return:
    - Final answer in message.content (plain string)
    - Reasoning in message.reasoning (plain string or list)

    Args:
        response: The raw response object from the OpenAI client

    Returns:
        Tuple of (response_content, response_reasoning_summary, usage_metadata, response_metadata)
    """
    response_content = ""
    response_reasoning_summary = None

    if not response.choices or len(response.choices) == 0:
        return "", None, None, None

    message = response.choices[0].message

    # Parse final answer from message.content (plain string)
    if hasattr(message, "content") and message.content is not None:
        if isinstance(message.content, str):
            response_content = message.content
        else:
            response_content = str(message.content)

    # Parse reasoning from message.reasoning (plain string or list)
    if hasattr(message, "reasoning") and message.reasoning:
        if isinstance(message.reasoning, str):
            response_reasoning_summary = message.reasoning
        elif isinstance(message.reasoning, list):
            # Handle list format if present
            reasoning_parts = []
            for item in message.reasoning:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        reasoning_parts.append(text)
                else:
                    reasoning_parts.append(str(item))
            response_reasoning_summary = (
                "\n\n".join(reasoning_parts) if reasoning_parts else None
            )

    # Parse usage metadata
    usage_metadata = _extract_usage_metadata(response)

    # Parse response metadata
    response_metadata = _extract_response_metadata(response)

    return (
        response_content,
        response_reasoning_summary,
        usage_metadata,
        response_metadata,
    )


def _extract_usage_metadata(response) -> dict | None:
    """Extract usage metadata including cost from cost_details."""
    if not hasattr(response, "usage") or not response.usage:
        return None

    usage_metadata = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
        "total_tokens": getattr(response.usage, "total_tokens", None),
    }

    # Add reasoning tokens from completion_tokens_details
    completion_details = getattr(response.usage, "completion_tokens_details", None)
    if completion_details:
        usage_metadata["reasoning_tokens"] = getattr(
            completion_details, "reasoning_tokens", None
        )

    # Add cached tokens from prompt_tokens_details
    prompt_details = getattr(response.usage, "prompt_tokens_details", None)
    if prompt_details:
        usage_metadata["cached_tokens"] = getattr(prompt_details, "cached_tokens", None)

    # Add cost info from cost_details
    cost_details = getattr(response.usage, "cost_details", None)
    if cost_details and isinstance(cost_details, dict):
        if "upstream_inference_cost" in cost_details:
            usage_metadata["upstream_inference_cost"] = cost_details[
                "upstream_inference_cost"
            ]
        if "upstream_inference_prompt_cost" in cost_details:
            usage_metadata["upstream_inference_prompt_cost"] = cost_details[
                "upstream_inference_prompt_cost"
            ]
        if "upstream_inference_completions_cost" in cost_details:
            usage_metadata["upstream_inference_completions_cost"] = cost_details[
                "upstream_inference_completions_cost"
            ]

    return usage_metadata


def _extract_response_metadata(response) -> dict | None:
    """Extract response metadata like model name, provider, etc."""
    response_metadata = {}

    if hasattr(response, "model"):
        response_metadata["model"] = response.model

    # Some responses include provider in the root
    if hasattr(response, "provider"):
        response_metadata["provider"] = response.provider

    return response_metadata if response_metadata else None
