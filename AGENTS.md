# Repository Guidelines

## Project Structure & Module Organization

BNqMark is a Python 3.12 project. Core importable code lives in `src/`, with packages for Bayesian networks (`bn/`), DAG generation (`dag/`), query generation and analysis (`queries/`), experiment orchestration (`experiment/`), database helpers (`database/`), and trace analysis (`trace_analysis/`). Experiment entry points are grouped under `experiments/`: `generate_data/`, `main/`, `export_data/`, `result_analysis/`, and `trace_analysis/`. Configuration YAMLs live in `config/`, prompt templates in `prompts/`, demos in `examples/`, and Hugging Face upload tooling in `hf_dataset/`. Generated outputs normally belong in `data/` or `plots/`; avoid committing large generated artifacts unless they are an intentional release asset.

## Build, Test, and Development Commands

- `uv sync --all-groups`: install runtime and development dependencies from `uv.lock`.
- `uv run ruff check .`: lint Python code using the repository Ruff rules.
- `uv run ruff format .`: format Python files.
- `uv run ty check`: run static type checks for `src/`.
- `uv run pre-commit run --all-files`: run formatting, linting, YAML/TOML/JSON checks, secret detection, and type checks.
- `uv build`: build package distributions with Hatchling.
- `uv run python experiments/generate_data/generate_bn_dataset.py`: run a representative data-generation script. Full LLM evaluation scripts may require provider credentials.

## Coding Style & Naming Conventions

Use 4-space indentation and keep lines at or below 88 characters. Ruff enforces import sorting, modern Python idioms, bugbear/comprehension/simplification checks, Google-style docstrings, and `pathlib` over `os.path`. Prefer type annotations on public functions and data structures. Use `snake_case` for modules, functions, variables, and script names; `PascalCase` for classes and Pydantic models; and `UPPER_SNAKE_CASE` for constants. Keep experiment scripts explicit and reproducible, with parameters sourced from `config/` where practical.

## Testing Guidelines

There is no dedicated pytest suite or coverage threshold configured yet. For changes in reusable logic, add focused tests under a future `tests/test_*.py` structure and wire the test runner into `pyproject.toml`. Until then, validate with `ruff`, `ty`, `pre-commit`, and the smallest relevant example or experiment script. Avoid using full LLM runs as routine tests because they depend on external model availability and API cost.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `refactor:`, and `chore:`; keep subjects imperative and scoped to one change. Pull requests should follow `.github/pr_template.md`: explain what changed and why. Include linked issues when available, note any generated data or plots, and mention the exact validation commands run. Never commit real secrets; copy `env.template` to a local `.env` and set `OPENROUTER_API_KEY` there.
