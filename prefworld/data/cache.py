from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


@dataclass
class CachedSampleIndex:
    path: str
    log_name: str
    scenario_token: str
    iteration: int


def write_index_jsonl(path: Path, items: List[CachedSampleIndex]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def read_index_jsonl(path: Path) -> List[CachedSampleIndex]:
    items: List[CachedSampleIndex] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            items.append(CachedSampleIndex(**d))
    return items


def save_npz(path: Path, sample: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
