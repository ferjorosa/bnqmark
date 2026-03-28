"""Pydantic models for code strategy analysis in LLM traces."""

from enum import Enum

from pydantic import BaseModel, Field

# --- Enums ---


class CodeGenerationStrategy(str, Enum):
    """The implementation approach used in the generated code."""

    LIBRARY_PGMPY = "library_pgmpy"  # Uses pgmpy for inference
    LIBRARY_PYAGRUM = "library_pyagrum"  # Uses pyAgrum
    GENERIC_ALGORITHM_IMPL = "generic_algorithm_impl"  # Implements a reusable solver
    MANUAL_INSTANCE_SCRIPT = "manual_instance_script"  # Hardcoded for this instance
    HARDCODED_PRINT = "hardcoded_print"  # Manual calc, code just prints
    OTHER = "other"  # Any other approach


class CodeFirstThinking(str, Enum):
    """Whether model considers using libraries vs defaulting to manual code."""

    LIBRARY_AWARE = "library_aware"  # Considers pgmpy/pyAgrum
    ALGORITHM_FOCUSED = "algorithm_focused"  # Implements generic algorithms
    MANUAL_DEFAULT = "manual_default"  # Goes straight to manual implementation


# --- Helper Classes ---


class StrategyClassification(BaseModel):
    """Detailed classification of the implementation strategy."""

    reasoning: str = Field(
        ...,
        description=(
            "Explain why this strategy matches the generated code. "
            "Quote specific parts of the code."
        ),
    )
    strategy: CodeGenerationStrategy = Field(
        ..., description="The primary implementation approach used in the code."
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Confidence in this classification (1=Low, 5=High).",
    )


class ThinkingStyleClassification(BaseModel):
    """Classification of whether the model thinks in terms of existing tools."""

    reasoning: str = Field(
        ...,
        description=(
            "Explain whether the model considered libraries/tools or went "
            "straight to manual implementation. Quote from the reasoning trace."
        ),
    )
    thinking_style: CodeFirstThinking = Field(
        ...,
        description="Whether the model is library-aware or defaults to manual code.",
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Confidence in this classification (1=Low, 5=High).",
    )


# --- Main Analysis Object ---


class CodeStrategyAnalysis(BaseModel):
    """
    Analysis of the strategic decisions made during code generation.

    Focuses on WHAT approach was chosen and WHY.
    """

    # 1. Implementation Strategy
    strategy_classification: StrategyClassification = Field(
        ...,
        description="Identification of the implementation approach used in the code.",
    )

    # 2. Thinking Style
    thinking_style_classification: ThinkingStyleClassification = Field(
        ...,
        description="Assessment of whether the model considered using libraries/tools.",
    )

    # 3. Metadata
    imported_libraries: list[str] = Field(
        default_factory=list,
        description=(
            "List of all Python libraries imported in the code "
            "(e.g., ['pgmpy', 'numpy', 'fractions']). Empty list if no imports."
        ),
    )

    code_summary: str = Field(
        ...,
        description=(
            "A 1-sentence summary of the code's approach (e.g., "
            "'Implemented Variable Elimination from scratch using dicts')."
        ),
    )
