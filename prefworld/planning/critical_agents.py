from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CriticalAgentSelection:
    indices: torch.Tensor  # [B,K_select] agent indices (within EB agent set)
    scores: torch.Tensor   # [B,K_select]
    mask: torch.Tensor     # [B,K] 1 if selected, else 0


def select_topk(
    pci_scores: torch.Tensor,  # [B,K]
    agent_mask: torch.Tensor,  # [B,K]
    k: int = 6,
    include_ego: bool = False,
) -> CriticalAgentSelection:
    B, K = pci_scores.shape
    scores = pci_scores.clone()
    # set invalid to -inf
    scores = scores.masked_fill(agent_mask <= 0.5, float("-inf"))
    if not include_ego:
        scores[:, 0] = float("-inf")
    topk_scores, topk_idx = torch.topk(scores, k=min(k, K), dim=1)
    sel_mask = torch.zeros_like(scores)
    sel_mask.scatter_(1, topk_idx, 1.0)
    # clean NaNs for cases with -inf
    topk_scores = torch.where(torch.isfinite(topk_scores), topk_scores, torch.zeros_like(topk_scores))
    return CriticalAgentSelection(indices=topk_idx, scores=topk_scores, mask=sel_mask)
