from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from prefworld.models.eb_stm import EBSTM, EBSTMOutput


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def jensen_shannon(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between categorical distributions p and q.
    Args:
      p,q: [B,C] distributions (sum to 1)
    Returns:
      js: [B]
    """
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (_safe_log(p, eps) - _safe_log(m, eps)), dim=-1)
    kl_qm = torch.sum(q * (_safe_log(q, eps) - _safe_log(m, eps)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def compute_pci_scores(
    ebstm: EBSTM,
    A_t: torch.Tensor,         # [B,K,K]
    agent_feat: torch.Tensor,  # [B,K,Da]
    z_mean: torch.Tensor,      # [B,K,Dz]
    z_logvar: torch.Tensor,    # [B,K,Dz]
    agent_mask: torch.Tensor,  # [B,K]
    *,
    baseline_mean: float = 0.0,
    baseline_logvar: float = 0.0,
    include_ego: bool = False,
) -> Tuple[torch.Tensor, EBSTMOutput]:
    """Compute PCI scores per agent for a one-step EB-STM candidate distribution.

    PCI(j) = JS( p_full(structure) , p_cf_j(structure) )
    where cf_j replaces agent j preference belief with a baseline (standard normal).

    Returns:
      pci: [B,K] (ego index 0 set to 0 unless include_ego=True)
      out_full: EBSTMOutput for reuse (candidate set + full probs)
    """
    out_full = ebstm(A_t, agent_feat, z_mean, z_logvar, agent_mask, oracle_next=None)
    p_full = out_full.probs  # [B,C]
    B, C = p_full.shape
    K = A_t.shape[1]

    pci = torch.zeros((B, K), device=A_t.device, dtype=torch.float32)

    # Precompute full energies
    E_full = out_full.energies  # [B,C]

    for j in range(K):
        if j == 0 and not include_ego:
            continue
        # Skip invalid agents
        valid_j = agent_mask[:, j] > 0.5
        if valid_j.sum() == 0:
            continue

        z_mean_cf = z_mean.clone()
        z_logvar_cf = z_logvar.clone()
        z_mean_cf[:, j] = baseline_mean
        z_logvar_cf[:, j] = baseline_logvar

        pair_cf = ebstm.energy_net(agent_feat, z_mean_cf, z_logvar_cf, agent_mask)
        # energy for each candidate under cf
        E_cf = []
        for c in range(C):
            E = ebstm.energy_net.energy_of_structure(pair_cf, out_full.candidate_A[:, c])
            E_cf.append(E)
        E_cf = torch.stack(E_cf, dim=1)  # [B,C]
        q = torch.softmax(-E_cf / float(ebstm.temperature), dim=1)

        js = jensen_shannon(p_full, q)  # [B]
        pci[:, j] = js

    return pci, out_full
