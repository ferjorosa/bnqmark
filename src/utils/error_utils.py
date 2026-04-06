"""
Utilities for detecting errors from LLM calls.

This module provides functions to detect token limit errors and extract
token counts from error messages when available.
"""

import re

from openai import BadRequestError


def is_token_limit_error(error: BadRequestError) -> tuple[bool, int | None]:
    """
    Check if an LLM call error is specifically about token limit exceeded.

    Extracts the token count from the error message when available.

    Error patterns handled:
    1. "maximum context length is X tokens. However, you requested about Y tokens"
    2. "prompt is too long: X tokens > Y maximum"

    Args:
        error: The RuntimeError exception.

    Returns:
        Tuple of (is_token_limit_error, token_count).
        - is_token_limit_error: True if the error is about token limit, False otherwise.
        - token_count: The extracted token count from the error message if available,
          otherwise None.
    """
    error_str = str(error)
    error_str_lower = error_str.lower()

    # Pattern 1: "maximum context length is X tokens.
    #             However, you requested about Y tokens"
    # Example: endpoint's max context is 204800 tokens.
    #          Requested 565880 tokens (565880 of text input).
    if (
        "maximum context length" in error_str_lower
        and "you requested about" in error_str_lower
    ):
        match = re.search(r"you requested about (\d+) tokens", error_str_lower)
        if match:
            return True, int(match.group(1))
        return True, None

    # Pattern 2: "prompt is too long: X tokens > Y maximum"
    # Example: "prompt is too long: 1452972 tokens > 1000000 max"
    # Note: Azure provider error for GPT models.
    if "prompt is too long" in error_str_lower:
        match = re.search(r"prompt is too long: (\d+) tokens", error_str_lower)
        if match:
            return True, int(match.group(1))
        return True, None

    return False, None
