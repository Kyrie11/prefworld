from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }

    # Store model constructor hyper-parameters when available.
    # This makes evaluation robust to config/checkpoint mismatches.
    if hasattr(model, "hparams"):
        try:
            ckpt["model_hparams"] = dict(getattr(model, "hparams"))
        except Exception:
            # Never fail saving due to hparams serialization.
            pass

    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def load_modules_from_checkpoint(
    path: str,
    model: torch.nn.Module,
    module_prefixes: Tuple[str, ...],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load a subset of modules from a checkpoint.

    This is useful for staged training (e.g., load stage-1 template+PC into a stage-2 run).

    Args:
      path: checkpoint path created by save_checkpoint
      model: target model
      module_prefixes: module name prefixes to load, e.g., ("template", "pc")
      strict: whether to require exact key match for the *subset*
    Returns:
      dict with keys: missing_keys, unexpected_keys
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    # filter keys
    keep = {}
    for k, v in sd.items():
        if any(k.startswith(p + ".") for p in module_prefixes):
            keep[k] = v

    missing, unexpected = model.load_state_dict(keep, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected, "loaded_prefixes": module_prefixes}
