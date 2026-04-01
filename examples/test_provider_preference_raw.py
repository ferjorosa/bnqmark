"""Test script to verify OpenRouter provider preferences work correctly using raw OpenAI client."""

import sys
from pathlib import Path

from dotenv import get_key

# Get API key from .env file
openrouter_api_key = get_key(".env", "OPENROUTER_API_KEY")

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.utils.llm import run_llm_call

# Simple test prompt (direct string, no template)
TEST_QUESTION = "What color is the sky? red or blue"

# Model to test (a popular one available on multiple providers)
# MODEL_NAME = "z-ai/glm-5"
# PROVIDER_PREFERENCES = ["deepinfra/fp4"]

MODEL_NAME = "google/gemini-3.1-pro-preview"
PROVIDER_PREFERENCES = ["google-ai-studio"]

# MODEL_NAME = "openai/gpt-5.4"
# PROVIDER_PREFERENCES = None

# MODEL_NAME = "anthropic/claude-sonnet-4.6"
# PROVIDER_PREFERENCES = ["anthropic"]

REASONING_EFFORT = "xhigh"
REASONING_SUMMARY = "detailed"


def test_with_provider(provider_preferences: list[str]) -> None:
    """Test LLM call with a specific provider preference."""
    print(f"\n{'=' * 60}")
    print(f"Testing with provider: {provider_preferences}")
    print(f"{'=' * 60}")

    content, usage, reasoning, metadata = run_llm_call(
        prompt=TEST_QUESTION,
        model_name=MODEL_NAME,
        openrouter_api_key=openrouter_api_key,
        provider_preferences=provider_preferences,
        temperature=1.0,
        reasoning_effort=REASONING_EFFORT,
        reasoning_summary=REASONING_SUMMARY,

    )

    print(f"Response: {content}")
    print(f"Usage: {usage}")

    if reasoning:
        print(f"\nReasoning:\n{reasoning[:500]}..." if len(reasoning) > 500 else f"\nReasoning:\n{reasoning}")

    # Check response metadata for provider info
    if metadata:
        provider = metadata.get("provider", "unknown")
        model_id = metadata.get("model", "unknown")
        print(f"\nProvider used: {provider}")
        print(f"Model ID: {model_id}")


def main() -> None:
    """Run provider preference tests."""
    # Check API key was loaded
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return

    print("Testing OpenRouter provider preferences (raw OpenAI client)")
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {TEST_QUESTION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Reasoning summary: {REASONING_SUMMARY}")

    test_with_provider(PROVIDER_PREFERENCES)


if __name__ == "__main__":
    main()
