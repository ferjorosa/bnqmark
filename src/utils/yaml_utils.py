"""YAML utilities for loading configuration and prompt files."""

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """
    Load a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    with path.open() as f:
        return yaml.safe_load(f)
