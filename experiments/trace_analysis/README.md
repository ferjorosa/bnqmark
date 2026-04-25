# Trace Analysis

> ⚠️ **Not part of the current paper.** ⚠️

Scripts for analyzing LLM reasoning traces in detail, developed for potential future work with models that expose reasoning tokens. These modules are **not used in the current paper** because closed-source models (GPT-5.4, Gemini 3.1 Pro, Claude Sonnet 4.6, Grok 4.20) do not provide the original reasoning traces. Grok does not even provide a reasoning summary. S

The analysis types below require access to the model's internal reasoning process (step-by-step thinking, tool calls, or explicit reasoning tokens), which are only available in some open-weight models. And, we decided not to do it for only half of the models.

## Scripts

### `run_ta_raw_reasoning_arithmetic_behaviour.py`

Analyze arithmetic correctness in raw reasoning traces.

**What it analyzes:**
- Detects arithmetic operations performed by the model (addition, multiplication, division)
- Checks for numerical precision errors
- Identifies when models approximate vs. compute exact probabilities
- Classifies arithmetic strategies (brute force, variable elimination, etc.)

**Main function:**
- `main()` — Run arithmetic behavior analysis on experiment batches

### `run_ta_raw_reasoning_inference_algorithm.py`

Identify inference algorithms used by models in raw reasoning.

**What it analyzes:**
- Detects mentions of specific algorithms (variable elimination, junction tree, sampling)
- Classifies heuristic usage (min-fill ordering, etc.)
- Identifies whether models explicitly construct factor graphs
- Maps reasoning structure to known inference techniques

**Main function:**
- `main()` — Run inference algorithm classification

### `run_ta_code_generation_strategy.py`

Analyze code generation approaches and strategies.

**What it analyzes:**
- Library selection patterns (pgmpy vs. pyAgrum vs. custom implementations)
- Code structure (functional vs. object-oriented)
- API usage correctness
- Handling of edge cases (division by zero, invalid probabilities)

**Main function:**
- `main()` — Run code generation strategy analysis

### `run_ta_code_generation_behaviour.py`

Analyze code execution behavior and failure patterns.

**What it analyzes:**
- Runtime error types (syntax, import, logic, timeout)
- Manual computation fallback (when code fails, does model compute manually?)
- Symbolic vs. numeric computation choices
- Output formatting adherence

**Main function:**
- `main()` — Run code generation behavior analysis

## Configuration

Analysis types are configured via `config/trace_analysis/*.yaml` with:
- Analysis-specific prompts
- Pydantic output schemas for structured extraction
- Model selection for the analysis LLM (typically a strong reasoning model)

## Future Use

These modules are designed for:
- Open-weight models with exposed reasoning tokens (e.g., DeepSeek-R1, Qwen with reasoning)
- Fine-tuned models where reasoning patterns are of interest
- Extended versions of the benchmark with reasoning-aware evaluation
