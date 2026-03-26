"""This module contains functions for counting tokens using tiktoken."""

import tiktoken
from langchain_core.prompts import PromptTemplate

# Cache the encoding to avoid recreating it on every call
_encoding_cache = None


def _get_encoding():
    """Get or create the cached encoding."""
    global _encoding_cache
    if _encoding_cache is None:
        _encoding_cache = tiktoken.get_encoding("o200k_base")
    return _encoding_cache


def count_input_tokens(
    prompt_template: PromptTemplate,
    parameters: dict,
) -> int:
    """
    Count the number of input tokens in a formatted prompt.

    Uses GPT-4o tokenizer (o200k_base encoding).

    Args:
        prompt_template (PromptTemplate): The prompt template to use.
        parameters (dict): The parameters for the prompt.

    Returns:
        int: The number of input tokens.
    """
    # Format the prompt template with parameters to get the actual prompt string
    formatted_prompt = prompt_template.format(**parameters)

    # Use cached encoding for better performance
    encoding = _get_encoding()

    # Count tokens
    tokens = encoding.encode(formatted_prompt)
    return len(tokens)
