# BNqMark

BNqMark is a benchmark for evaluating Large Language Models (LLMs) on exact probabilistic inference in discrete Bayesian Networks. It isolates probabilistic computation from linguistic interpretation by presenting complete conditional probability table (CPT) specifications and asking models to answer conditional probability queries.

Two evaluation protocols are used:
- **Raw reasoning**: The model computes the probability directly from the CPTs
- **Code generation**: The model writes Python code to solve the same inference problem

## Repository Structure

- `src/` — Core source code for Bayesian network generation, query generation, and experiment orchestration
- `experiments/` — Experiment scripts for running evaluations and analyzing results
- `hf_dataset/` — HuggingFace dataset upload script and documentation
- `data/` — Generated datasets (parquet files)
- `prompts/` — LLM prompt templates
- `config/` — Configuration files for experiments

> **Note on Trace Analysis**: The `src/trace_analysis/` and `experiments/trace_analysis/` modules are **not part of the current paper**. Closed-source models (GPT-5.4, Gemini 3.1 Pro, Claude Sonnet 4.6, Grok 4.20) do not provide the original reasoning traces required for this analysis. In the case of Grok, no reasoning summary is provided at all. These modules were developed for potential future work with open-weight models that expose reasoning tokens.

## Dataset

The BNqMark-20 dataset is published on HuggingFace Hub:

```python
from datasets import load_dataset

bns = load_dataset("ferjorosa/bnqmark-20", "bns")["train"]
queries = load_dataset("ferjorosa/bnqmark-20", "queries")["train"]
experiments = load_dataset("ferjorosa/bnqmark-20", "experiments")["train"]
```

The dataset includes:
- 78 Bayesian networks (4-20 binary variables)
- 434 conditional probability queries
- 7,812 LLM evaluation results from 9 frontier models

## Reproducing the Paper Experiments

The main experiment pipeline follows this order:

1. Generate Bayesian networks:
   `python experiments/generate_data/generate_bn_dataset.py`
2. Generate inference queries:
   `python experiments/generate_data/generate_query_dataset.py`
3. Run LLM evaluations:
   `python experiments/main/run_experiments.py`
4. Export experiment results:
   `python experiments/export_data/export_experiments_to_parquet.py`
5. Generate result plots:
   `python experiments/result_analysis/plot_accuracy_heatmap_grid.py`
   `python experiments/result_analysis/plot_answerability_heatmap_grid.py`
   `python experiments/result_analysis/plot_mae_heatmap_grid.py`

Configuration files live in `config/`, prompt templates in `prompts/`, and generated outputs are written to `data/` and `plots/`. Exact reruns may differ slightly depending on OpenRouter model availability and provider-side model updates.

## Citation

```bibtex
@dataset{bnqmark_20,
  title={BNqMark-20 Dataset},
  author={Fernando Rodriguez and Bojan Mihaljevic},
  year={2026},
  url={https://huggingface.co/datasets/ferjorosa/bnqmark-20}
}
```

## License

MIT License
