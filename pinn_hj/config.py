from __future__ import annotations

from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return a flat dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (key: value).")
    return data
