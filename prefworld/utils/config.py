from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


def parse_overrides(unknown: List[str]) -> List[str]:
    """Parse CLI overrides of the form key=value (OmegaConf style)."""
    overrides: List[str] = []
    for token in unknown:
        if "=" in token:
            overrides.append(token)
    return overrides


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> Any:
    """Load a YAML config and apply overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def make_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file (OmegaConf).",
    )
    return parser


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
