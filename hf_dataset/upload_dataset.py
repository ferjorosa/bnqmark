#!/usr/bin/env python3
"""
Script to upload the BNqMark-20 dataset to Hugging Face Hub.

BNqMark-20 is a benchmark for exact probabilistic inference in discrete
Bayesian Networks, evaluating Large Language Models on conditional probability
queries.

The dataset consists of three parts:
- bns: 78 Bayesian network configurations using the simple naming strategy
- queries: 434 probabilistic inference queries using the simple naming strategy
- experiments: 7,812 LLM evaluation results (9 models * 2 protocols)
"""

import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Hugging Face repository ID
REPO_ID = "ferjorosa/bnqmark-20"


def main():
    """Main function to upload the BNqMark-20 dataset to Hugging Face Hub."""
    # Load environment variables from .env file
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
        print("Attempting to use environment variables from system...")

    # Get HF token from environment
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    if not hf_token:
        print("Error: HF_TOKEN not found in .env file or environment variables.")
        print("Please add HF_TOKEN=your_token_here to your .env file.")
        return

    # Define paths
    data_dir = project_root / "data"
    bns_path = data_dir / "bns.parquet"
    queries_path = data_dir / "queries.parquet"
    experiments_path = data_dir / "experiments.parquet"

    # Verify all files exist
    for path in [bns_path, queries_path, experiments_path]:
        if not path.exists():
            print(f"Error: Dataset file not found at {path}")
            print("Please run the data generation scripts first.")
            return

    print(f"Found all dataset files in {data_dir}")
    print(f"Repository ID: {REPO_ID}")

    # Login to Hugging Face
    print("\nAuthenticating with Hugging Face...")
    try:
        login(token=hf_token)
        print("Authentication successful!")
    except Exception as e:
        print(f"Login failed: {e}")
        print("Please ensure you have a valid Hugging Face token.")
        return

    # Load datasets from parquet files
    print("\nLoading datasets from parquet files...")

    print(f"  Loading bns from {bns_path.name}...")
    bns_dataset = Dataset.from_parquet(str(bns_path))
    bns_dataset = bns_dataset.filter(lambda row: row["naming_strategy"] == "simple")
    print(f"    -> {len(bns_dataset)} rows after filtering to naming_strategy='simple'")

    print(f"  Loading queries from {queries_path.name}...")
    queries_dataset = Dataset.from_parquet(str(queries_path))
    queries_dataset = queries_dataset.filter(
        lambda row: row["naming_strategy"] == "simple"
    )
    print(
        f"    -> {len(queries_dataset)} rows after filtering to "
        "naming_strategy='simple'"
    )

    print(f"  Loading experiments from {experiments_path.name}...")
    experiments_dataset = Dataset.from_parquet(str(experiments_path))
    experiments_dataset = experiments_dataset.filter(
        lambda row: row["naming_strategy"] == "simple"
    )
    print(
        f"    -> {len(experiments_dataset)} rows after filtering to "
        "naming_strategy='simple'"
    )

    # Push each dataset separately as different configs
    print(f"\nUploading to {REPO_ID}...")
    print("  (Uploading as separate configs: bns, queries, experiments)")

    try:
        # Upload bns config
        print("\n  Uploading 'bns' config...")
        bns_dataset.push_to_hub(REPO_ID, config_name="bns", private=False)
        print("    -> bns uploaded successfully!")

        # Upload queries config
        print("\n  Uploading 'queries' config...")
        queries_dataset.push_to_hub(REPO_ID, config_name="queries", private=False)
        print("    -> queries uploaded successfully!")

        # Upload experiments config
        print("\n  Uploading 'experiments' config...")
        experiments_dataset.push_to_hub(
            REPO_ID, config_name="experiments", private=False
        )
        print("    -> experiments uploaded successfully!")

        print("\nAll configs uploaded successfully!")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

    # Upload README.md if it exists
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        print("\nUploading README.md...")
        api = HfApi()
        try:
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=hf_token,
            )
            print("README.md uploaded successfully!")
        except Exception as e:
            print(f"Warning: Could not upload README.md: {e}")

    print(f"\n{'=' * 60}")
    print("Upload complete!")
    print(f"View the dataset at: https://huggingface.co/datasets/{REPO_ID}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
