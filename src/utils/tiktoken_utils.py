"""This module contains functions for counting tokens using tiktoken."""

import tiktoken

# Cache the encoding to avoid recreating it on every call
_encoding_cache = None


def _get_encoding():
    """Get or create the cached encoding."""
    global _encoding_cache
    if _encoding_cache is None:
        _encoding_cache = tiktoken.get_encoding("o200k_base")
    return _encoding_cache


def count_input_tokens(prompt: str) -> int:
    """
    Count the number of input tokens in a prompt string.

    Uses GPT-4o tokenizer (o200k_base encoding).

    Args:
        prompt (str): The prompt string to count tokens for.

    Returns:
        int: The number of input tokens.
    """
    # Use cached encoding for better performance
    encoding = _get_encoding()

    # Count tokens
    tokens = encoding.encode(prompt)
    return len(tokens)
