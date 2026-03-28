"""Pydantic models for code generation behavior analysis in LLM traces."""

from enum import Enum

from pydantic import BaseModel, Field

# --- Enums ---


class ManualComputationVolume(str, Enum):
    """How much of the problem the model solves manually versus delegating to code."""

    NONE = "none"  # No manual calculations; proceeds directly to code
    LIGHT = "light"  # 1-3 exploratory calculations; code handles the rest
    MODERATE = "moderate"  # Several intermediate values computed manually
    HEAVY = "heavy"  # Most/all of the problem solved manually before coding


# --- Helper Classes ---


class ManualComputationCheck(BaseModel):
    """Analysis of how much was solved manually versus delegated to code."""

    reasoning: str = Field(
        ...,
        description=(
            "Describe what calculations were done manually in the trace "
            "versus what the code computes. Quote specific examples and "
            "count distinct calculations."
        ),
    )
    volume: ManualComputationVolume = Field(
        ...,
        description=(
            "The level of manual computation observed (none, light, moderate, heavy)."
        ),
    )
    confidence_score: int = Field(
        ..., ge=1, le=5, description="Confidence in this assessment (1=Low, 5=High)."
    )


class SymbolicMathCheck(BaseModel):
    """Check if the code uses symbolic math libraries for exact arithmetic."""

    reasoning: str = Field(
        ...,
        description=(
            "Quote specific parts of the code showing symbolic math usage "
            "(fractions, sympy, decimal) or explain why floating point is used."
        ),
    )
    is_present: bool = Field(
        ...,
        description=(
            "True if symbolic libraries (sympy, fractions, decimal for exact "
            "arithmetic) are used instead of standard float."
        ),
    )
    confidence_score: int = Field(
        ..., ge=1, le=5, description="Confidence in this assessment (1=Low, 5=High)."
    )


# --- Main Analysis Object ---


class CodeGenerationBehaviourAnalysis(BaseModel):
    """
    Analysis of behavioral patterns during code generation.

    Focuses on how much manual calculation the model performs before
    delegating to code.
    """

    manual_computation_volume: ManualComputationCheck = Field(
        ...,
        description=(
            "How much of the problem was solved manually in the reasoning "
            "trace versus delegated to code?"
        ),
    )

    uses_symbolic_math: SymbolicMathCheck = Field(
        ...,
        description=(
            "Does the code use symbolic libraries (sympy, fractions, decimal) "
            "instead of floating point arithmetic?"
        ),
    )
