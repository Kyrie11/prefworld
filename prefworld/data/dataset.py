from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from prefworld.data.cache import CachedSampleIndex, load_npz, read_index_jsonl


class CachedNuPlanDataset(Dataset):
    """Dataset that reads cached `.npz` samples produced by `prepare_dataset.py`."""

    def __init__(self, cache_dir: str, split: str, max_samples: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.split = split
        index_path = self.cache_dir / split / "index.jsonl"
        self.items: List[CachedSampleIndex] = read_index_jsonl(index_path)
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        sample_path = Path(item.path)
        sample = load_npz(sample_path)
        # Convert to torch tensors
        out: Dict[str, Any] = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                # use int64 for integer arrays, float32 otherwise
                if v.dtype.kind in ("i", "u"):
                    out[k] = torch.from_numpy(v.astype(np.int64))
                else:
                    out[k] = torch.from_numpy(v.astype(np.float32))
            else:
                out[k] = v
        # metadata (not tensor)
        out["_meta"] = {
            "log_name": item.log_name,
            "scenario_token": item.scenario_token,
            "iteration": item.iteration,
        }
        return out


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate dict-of-tensors batch."""
    keys = [k for k in batch[0].keys() if k != "_meta"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    out["_meta"] = [b["_meta"] for b in batch]
    return out
