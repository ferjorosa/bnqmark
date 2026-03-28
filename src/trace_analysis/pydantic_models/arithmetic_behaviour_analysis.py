"""Pydantic models for arithmetic behavior analysis in LLM traces."""

from pydantic import BaseModel, Field

# --- Helper Classes ---


class ArithmeticCheck(BaseModel):
    """Detailed analysis of a specific arithmetic behavior or pattern."""

    reasoning: str = Field(
        ...,
        description=(
            "Quote specific parts of the trace or explain why this behavior "
            "is present or absent."
        ),
    )
    is_present: bool = Field(
        ..., description="True if the behavior is explicitly observed in the trace."
    )
    confidence_score: int = Field(
        ..., ge=1, le=5, description="Confidence in this assessment (1=Low, 5=High)."
    )


# --- Main Analysis Object ---


class ArithmeticBehaviorAnalysis(BaseModel):
    """
    Analysis of arithmetic behaviors exhibited by the LLM.

    Focuses on *how* the model performs calculations and validates its own work.
    """

    # --- Observable Calculation Signals ---
    obsessive_manual_calculation: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model engages in excessively long, step-by-step "
            "manual calculation (like long division) spanning multiple "
            "paragraphs, often chasing unreasonable precision."
        ),
    )

    fraction_conversion: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model explicitly converts decimals to fractions "
            "to perform calculations (e.g., '0.5 = 1/2')."
        ),
    )

    scientific_notation: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model explicitly uses scientific notation for "
            "intermediate steps (e.g., '1.5e-3')."
        ),
    )

    interpolation_or_estimation: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model explicitly estimates a value based on nearby "
            "knowns or bounds instead of calculating it directly."
        ),
    )

    arithmetic_hallucinations: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model states blatantly incorrect arithmetic facts "
            "(e.g., '0.2 * 0.2 = 0.4') or fails basic normalization (sum != 1)."
        ),
    )

    # --- Verification & Anxiety Patterns ---
    constant_verification_loops: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model *frequently* (more than 2-3 times) re-calculates "
            "the same value immediately to verify it."
        ),
    )

    multiple_method_verification: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model verifies a result using a *different* method "
            "(e.g., checking a decimal multiplication by converting to fractions)."
        ),
    )

    excessive_precision: ArithmeticCheck = Field(
        ...,
        description=(
            "Check if the model carries an unusually high number of decimal "
            "places (>8) throughout the entire trace, beyond what is reasonable."
        ),
    )
