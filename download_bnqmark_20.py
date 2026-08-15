#!/usr/bin/env python3
"""Download the BNqMark-20 dataset from the Hugging Face Hub.

Downloads the three parquet files (bns, queries, experiments) of the public
dataset `ferjorosa/bnqmark-20` and renames them to the filenames expected by
the analysis scripts in experiments/result_analysis.
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ferjorosa/bnqmark-20"
CONFIG_FILES = {
    "bns": "data/bns/train.parquet",
    "queries": "data/queries/train.parquet",
    "experiments": "data/experiments/train.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help=(
            "Directory where the parquet files will be saved with the names "
            "bns.parquet, queries.parquet, and experiments.parquet. Defaults to "
            "data/, the location expected by the result-analysis scripts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_name, repo_path in CONFIG_FILES.items():
        print(f"Downloading {config_name} from {repo_path}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
        )
        dest = output_dir / f"{config_name}.parquet"
        shutil.copy2(downloaded, dest)
        print(f"  -> {dest}")

    print("\nDone.")


if __name__ == "__main__":
    main()