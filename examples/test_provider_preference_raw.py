"""
Test script to verify OpenRouter provider preferences work correctly.

Uses the raw OpenAI client.
"""

import sys
from pathlib import Path

from dotenv import get_key

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.utils.llm import run_llm_call

# Get API key from .env file
OPENROUTER_API_KEY = get_key(".env", "OPENROUTER_API_KEY")

# Model to test (a popular one available on multiple providers)
MODEL_NAME = "deepseek/deepseek-v3.2-speciale"
PROVIDER_PREFERENCES = ["atlas-cloud/fp8"]
# PROVIDER_PREFERENCES = None

REASONING_EFFORT = "xhigh"
REASONING_SUMMARY = "detailed"
TEMPERATURE = 0.0

# Simple test prompt (direct string, no template)
TEST_QUESTION = "What color is the sky? red or blue"


def main() -> None:
    """Run provider preference tests."""
    # Check API key was loaded
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return

    print("Testing OpenRouter provider preferences (raw OpenAI client)")
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {TEST_QUESTION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Reasoning summary: {REASONING_SUMMARY}")

    print(f"\n{'=' * 60}")
    print(f"Testing with provider: {PROVIDER_PREFERENCES}")
    print(f"{'=' * 60}")

    content, usage, reasoning, metadata = run_llm_call(
        prompt=TEST_QUESTION,
        model_name=MODEL_NAME,
        openrouter_api_key=OPENROUTER_API_KEY,
        provider_preferences=PROVIDER_PREFERENCES,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        reasoning_summary=REASONING_SUMMARY,
    )

    print(f"Response: {content}")

    if reasoning:
        print(
            f"\nReasoning:\n{reasoning[:500]}..."
            if len(reasoning) > 500
            else f"\nReasoning:\n{reasoning}"
        )

    # Check response metadata for provider info
    if metadata:
        provider = metadata.get("provider", "unknown")
        model_id = metadata.get("model", "unknown")
        print(f"\nProvider used: {provider}")
        print(f"Model ID: {model_id}")

    if usage:
        print(f"\nUsage: {usage}")

if __name__ == "__main__":
    main()
