"""
Simple Pydantic output parser - replacement for LangChain's PydanticOutputParser.

This module provides a lightweight alternative to LangChain's PydanticOutputParser
for parsing LLM responses into Pydantic models.
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser:
    """
    Parse LLM responses into Pydantic models.

    This is a lightweight replacement for LangChain's PydanticOutputParser
    that doesn't require the full LangChain dependency.
    """

    def __init__(self, pydantic_object: type[T]):
        """
        Initialize with a Pydantic model class.

        Args:
            pydantic_object: The Pydantic model class to parse into.
        """
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        """
        Generate format instructions from the Pydantic model schema.

        Returns:
            A string containing JSON format instructions for the LLM.
        """
        schema = self.pydantic_object.model_json_schema()

        # Remove internal Pydantic references for cleaner output
        schema.pop("title", None)
        schema.pop("$defs", None)

        return (
            "Respond with a JSON object that conforms to the following schema:\n\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Ensure your response is valid JSON and includes all required fields."
        )

    def parse(self, text: str) -> T:
        """
        Parse the LLM response text into the Pydantic model.

        Args:
            text: The raw text response from the LLM.

        Returns:
            An instance of the Pydantic model.

        Raises:
            ValueError: If the text is not valid JSON or doesn't match the schema.
        """
        text = text.strip()

        # Try to extract JSON from markdown code blocks
        # Handle ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            # Find the closing ```
            match = re.search(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        else:
            # Try to find JSON object in the text
            # Look for the first { and last }
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                text = text[start:end]
            except ValueError:
                pass  # No braces found, use text as-is

        # Parse JSON and validate against Pydantic model
        try:
            data = json.loads(text)
            return self.pydantic_object.model_validate(data)  # ty: ignore
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in response: {e}\nResponse preview: {text[:500]}..."
            ) from e
        except ValidationError as e:
            raise ValueError(
                f"Response doesn't match schema: {e}\nResponse preview: {text[:500]}..."
            ) from e

    @property
    def _type(self) -> str:
        """Return the parser type identifier."""
        return "pydantic"
