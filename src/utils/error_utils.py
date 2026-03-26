"""
Utilities for detecting errors from LLM calls.

This module provides functions to detect token limit errors across various providers.
"""


def is_token_limit_error(error: RuntimeError) -> bool:
    """
    Check if an LLM call error is specifically about token limit exceeded.

    This function checks multiple token limit error patterns in the lowercased
    error string:
    1. Gemini-style: "the input token count exceeds the maximum number of
       tokens allowed"
    2. Context length: "maximum context length" and "tokens" in the message
    3. Upstream error: "is longer than the model's context length"
    4. AWS Bedrock: "input is too long for requested model"
    5. OpenRouter/Chutes: "exceeds maximum input length"
    6. GPT-5: "your input exceeds the context window of this model"

    Args:
        error: The RuntimeError exception.

    Returns:
        True if the error is about token limit, False otherwise.
    """
    error_str_lower = str(error).lower()

    # Pattern 1: Gemini-style token limit error
    # Found in Gemini 3 Pro Preview
    gemini_pattern = (
        "the input token count exceeds the maximum number of tokens allowed"
    )
    if gemini_pattern in error_str_lower:
        return True

    # Pattern 2: Context length token limit error
    # Found in GLM-4.6
    if "maximum context length" in error_str_lower and "tokens" in error_str_lower:
        return True

    # Pattern 3: Upstream error from providers (e.g., Chutes)
    # Found in Qwen 3.235B A22B thinking 2507
    if "is longer than the model's context length" in error_str_lower:
        return True

    # Pattern 4: AWS Bedrock error
    # Found in Sonnet 4.5
    if "input is too long for requested model" in error_str_lower:
        return True

    # Pattern 5: OpenRouter/Chutes error with input length
    # Found in Qwen 3.235B A22B thinking 2507
    if "exceeds maximum input length" in error_str_lower:
        return True

    # Pattern 6: GPT-5 error
    # Found in GPT-5
    return "your input exceeds the context window of this model" in error_str_lower
