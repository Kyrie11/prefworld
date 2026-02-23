from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairEnergies:
    """Energy tensors for pairwise factorization."""
    e_dir: torch.Tensor   # [B, K, K] directed edge energy for i->j (i yields to j); diagonal unused
    e_none: torch.Tensor  # [B, K, K] symmetric none-energy for unordered pair (i<j used)


class EditFactorizedEnergyNet(nn.Module):
    """Edit-factorized, preference-local energy network (pairwise factorization).

    We model the interaction structure as a set of precedence/yield relations between agents.
    For each unordered pair (i,j), the relation can be:
      - NONE
      - i -> j   (i yields to j)
      - j -> i

    Total energy sums over unordered pairs. This factorization enables fast counterfactual reweighting
    when replacing a single agent's preference belief.
    """

    def __init__(
        self,
        agent_feat_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # Directed energy network: uses ordered pair features
        dir_in = agent_feat_dim * 2 + 5 + 2 * z_dim * 2  # (ai,aj)+(relpos2,relvel2,dist)+(zi_mean/logvar)+(zj_mean/logvar)
        self.dir_mlp = nn.Sequential(
            nn.Linear(dir_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # None energy network: symmetric features
        none_in = 5 + agent_feat_dim * 2 + 2 * z_dim * 2
        self.none_mlp = nn.Sequential(
            nn.Linear(none_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pair_features(
        agent_feat: torch.Tensor,  # [B,K,Da]
        z_mean: torch.Tensor,      # [B,K,Dz]
        z_logvar: torch.Tensor,    # [B,K,Dz]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, Da = agent_feat.shape
        Dz = z_mean.shape[-1]
        # broadcast
        ai = agent_feat.unsqueeze(2).expand(B, K, K, Da)
        aj = agent_feat.unsqueeze(1).expand(B, K, K, Da)
        zi_m = z_mean.unsqueeze(2).expand(B, K, K, Dz)
        zj_m = z_mean.unsqueeze(1).expand(B, K, K, Dz)
        zi_lv = z_logvar.unsqueeze(2).expand(B, K, K, Dz)
        zj_lv = z_logvar.unsqueeze(1).expand(B, K, K, Dz)

        # rel pos/vel (assume agent_feat includes x,y,vx,vy at indices 0,1,3,4)
        rel_pos = aj[..., 0:2] - ai[..., 0:2]  # [B,K,K,2]
        rel_vel = aj[..., 3:5] - ai[..., 3:5]  # [B,K,K,2]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B,K,K,1]
        geom = torch.cat([rel_pos, rel_vel, dist], dim=-1)  # [B,K,K,5]

        dir_feat = torch.cat([ai, aj, geom, zi_m, zi_lv, zj_m, zj_lv], dim=-1)
        none_feat = torch.cat([geom, ai, aj, zi_m, zi_lv, zj_m, zj_lv], dim=-1)
        return dir_feat, none_feat

    def forward(
        self,
        agent_feat: torch.Tensor,  # [B,K,Da]
        z_mean: torch.Tensor,      # [B,K,Dz]
        z_logvar: torch.Tensor,    # [B,K,Dz]
        agent_mask: torch.Tensor,  # [B,K] 1 valid, 0 pad
    ) -> PairEnergies:
        dir_feat, none_feat = self._pair_features(agent_feat, z_mean, z_logvar)
        B, K, _, _ = dir_feat.shape

        # compute energies
        e_dir = self.dir_mlp(dir_feat).squeeze(-1)    # [B,K,K]
        e_none = self.none_mlp(none_feat).squeeze(-1) # [B,K,K]

        # mask invalid pairs (set huge energy so they are unlikely)
        m_i = agent_mask.unsqueeze(2)  # [B,K,1]
        m_j = agent_mask.unsqueeze(1)  # [B,1,K]
        pair_mask = m_i * m_j
        # no self edges
        eye = torch.eye(K, device=agent_feat.device, dtype=agent_feat.dtype).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)

        huge = torch.tensor(1e6, device=agent_feat.device, dtype=agent_feat.dtype)
        e_dir = torch.where(pair_mask > 0.5, e_dir, huge)
        e_none = torch.where(pair_mask > 0.5, e_none, huge)
        # enforce symmetry for none-energy (average)
        e_none = 0.5 * (e_none + e_none.transpose(1, 2))
        return PairEnergies(e_dir=e_dir, e_none=e_none)

    @staticmethod
    def energy_of_structure(pair: PairEnergies, A: torch.Tensor) -> torch.Tensor:
        """Compute total energy for adjacency A.
        Args:
          pair: PairEnergies
          A: [B,K,K] adjacency with A[i,j]=1 means i yields to j
        Returns:
          E: [B]
        """
        B, K, _ = A.shape
        # Determine relation state per unordered pair (i<j)
        E = torch.zeros((B,), device=A.device, dtype=pair.e_dir.dtype)
        for i in range(K):
            for j in range(i + 1, K):
                a_ij = A[:, i, j]
                a_ji = A[:, j, i]
                # none if both 0
                e_none = pair.e_none[:, i, j]
                e_ij = pair.e_dir[:, i, j]
                e_ji = pair.e_dir[:, j, i]
                e = torch.where(a_ij > 0.5, e_ij, torch.where(a_ji > 0.5, e_ji, e_none))
                E = E + e
        return E

    @staticmethod
    def pair_energy_matrix(pair: PairEnergies, A: torch.Tensor) -> torch.Tensor:
        """Return a symmetric matrix P where P[i,j]=energy contribution for unordered pair (i,j)."""
        B, K, _ = A.shape
        P = torch.zeros((B, K, K), device=A.device, dtype=pair.e_dir.dtype)
        for i in range(K):
            for j in range(i + 1, K):
                a_ij = A[:, i, j]
                a_ji = A[:, j, i]
                e_none = pair.e_none[:, i, j]
                e_ij = pair.e_dir[:, i, j]
                e_ji = pair.e_dir[:, j, i]
                e = torch.where(a_ij > 0.5, e_ij, torch.where(a_ji > 0.5, e_ji, e_none))
                P[:, i, j] = e
                P[:, j, i] = e
        return P
