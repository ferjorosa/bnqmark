"""Pydantic models for inference algorithm analysis in LLM traces."""

from enum import Enum

from pydantic import BaseModel, Field

# --- Enums ---


class InferenceAlgorithmType(str, Enum):
    """Type of inference algorithm used."""

    CHAIN_RULE_BRUTE_FORCE = "chain_rule_brute_force"
    VARIABLE_ELIMINATION = "variable_elimination"
    CUTSET_CONDITIONING = "cutset_conditioning"
    JUNCTION_TREE = "junction_tree"
    BELIEF_PROPAGATION = "belief_propagation"
    OTHER = "other"


# --- Helper Classes ---


class InferenceAlgorithmClassification(BaseModel):
    """Detailed classification of the primary inference algorithm used."""

    reasoning: str = Field(
        ...,
        description=(
            "Explain which algorithm definition best matches the trace steps and why."
        ),
    )
    algorithm_type: InferenceAlgorithmType = Field(
        ..., description="The identified algorithm category."
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Confidence in this classification (1=Low, 5=High).",
    )
    deviation_details: str | None = Field(
        None,
        description=(
            "If the approach deviates from the textbook definition "
            "(e.g., mixed strategies), describe how."
        ),
    )


class InferenceHeuristicCheck(BaseModel):
    """Analysis of a specific optimization technique or heuristic."""

    reasoning: str = Field(
        ...,
        description=(
            "Quote specific parts of the trace or explain why this "
            "heuristic is present or absent."
        ),
    )
    is_present: bool = Field(
        ..., description="True if the heuristic is explicitly used."
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Confidence in this heuristic check (1=Low, 5=High).",
    )


# --- Main Analysis Object ---


class InferenceAlgorithmAnalysis(BaseModel):
    """Full structured analysis of the probabilistic reasoning trace."""

    # 1. Main Strategy
    classification: InferenceAlgorithmClassification = Field(
        ..., description="Identification of the overall inference strategy."
    )

    # 2. Specific Optimizations
    barren_nodes: InferenceHeuristicCheck = Field(
        ...,
        description=(
            "Check if the model explicitly identified and removed/ignored "
            "irrelevant variables (barren nodes) not part of query or evidence."
        ),
    )

    conditional_independence: InferenceHeuristicCheck = Field(
        ..., description="Check if conditional independence was used to simplify terms."
    )

    algebraic_simplification: InferenceHeuristicCheck = Field(
        ...,
        description=(
            "Check if symbolic simplification was performed before calculation."
        ),
    )
