from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

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


_GROUP_KEYS = {
    "experiment",
    "setting",
    "training",
    "model",
    "evaluation",
    "io",
}


def _flatten_grouped_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten grouped config sections into a single dict.

    Grouped sections:
      - experiment
      - setting
      - training
      - model
      - evaluation
      - io

    Top-level keys override grouped values to preserve backward compatibility.
    """
    flat: Dict[str, Any] = {}
    if any(key in data for key in _GROUP_KEYS):
        for group in _GROUP_KEYS:
            section = data.get(group)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ValueError(f"Config section '{group}' must be a mapping.")
            flat.update(section)
    # Allow top-level keys to override grouped ones
    for key, value in data.items():
        if key in _GROUP_KEYS:
            continue
        flat[key] = value
    return flat


def parse_args_with_config(
    argv,
    build_parser: Callable[[Dict[str, Any]], argparse.ArgumentParser],
) -> argparse.Namespace:
    """
    Parse CLI args with optional YAML config defaults.

    build_parser(defaults) should return an argparse.ArgumentParser.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    cfg_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml_config(cfg_args.config) if cfg_args.config else {}
    cfg = _flatten_grouped_config(cfg)
    parser = build_parser(cfg)
    return parser.parse_args(argv)
