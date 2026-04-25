# BNqMark

BNqMark is a benchmark for evaluating Large Language Models (LLMs) on exact probabilistic inference in discrete Bayesian Networks.

## Overview

The benchmark isolates probabilistic computation from linguistic interpretation by presenting LLMs with complete conditional probability table (CPT) specifications and asking them to answer conditional probability queries.

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

## Dataset

The BNqMark-20 dataset is published on HuggingFace Hub:

```python
from datasets import load_dataset

# Load each configuration
bns = load_dataset("ferjorosa/bnqmark-20", "bns")["train"]
queries = load_dataset("ferjorosa/bnqmark-20", "queries")["train"]
experiments = load_dataset("ferjorosa/bnqmark-20", "experiments")["train"]
```

The dataset includes:
- 78 Bayesian networks (4-20 binary variables)
- 434 conditional probability queries
- 7,812 LLM evaluation results from 9 frontier models

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
