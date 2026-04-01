"""
Detailed response parser for OpenRouter models.

Parses models that return reasoning in the structured
message.reasoning_details format:

- Google Gemini-3.1 (and other Gemini models)

These models return reasoning as a list of typed objects with
"reasoning.text" and "reasoning.encrypted" entries.
Only "reasoning.text" entries are extracted; encrypted entries are ignored.
"""


def parse_detailed_response(
    response,
) -> tuple[str, str | None, dict | None, dict | None]:
    """
    Parse response for models using the detailed message.reasoning_details format.

    These models return:
    - Final answer in message.content (string or list of dicts)
    - Reasoning in message.reasoning_details as a list of dicts with "type" field
      - "reasoning.text" entries contain the reasoning text
      - "reasoning.encrypted" entries are ignored (can't decrypt)

    Multiple reasoning.text entries are concatenated in order.

    Args:
        response: The raw response object from the OpenAI client

    Returns:
        Tuple of (response_content, response_reasoning_summary,
        usage_metadata, response_metadata)
    """
    response_content = ""
    reasoning_parts: list[str] = []

    if not response.choices or len(response.choices) == 0:
        return "", None, None, None

    message = response.choices[0].message

    # Parse final answer from message.content
    # Can be either a plain string or a list of dicts
    if hasattr(message, "content") and message.content is not None:
        if isinstance(message.content, str):
            response_content = message.content
        elif isinstance(message.content, list) and len(message.content) > 0:
            # If content is a list, look for text type entry
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        response_content = text
                        break

    # Parse reasoning from reasoning_details
    if hasattr(message, "reasoning_details") and message.reasoning_details:
        for detail in message.reasoning_details:
            if not isinstance(detail, dict):
                continue
            detail_type = detail.get("type", "")
            # Only extract reasoning.text entries, ignore encrypted ones
            if detail_type == "reasoning.text":
                text = detail.get("text", "")
                if text:
                    reasoning_parts.append(text)

    # Concatenate all reasoning parts in order with double newlines
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
