"""
LLM utilities using OpenAI library directly (no LangChain).

Tuned for OpenRouter with direct API calls.
"""

from .run_llm import run_llm_call

__all__ = ["run_llm_call"]
