from __future__ import annotations

"""Edit-token EFEN (paper-faithful EB-STM transition energy).

The original repo implemented EFEN energies for *static relation tokens* (directed / none).
The paper factorizes EB-STM transitions over **edit tokens** δ_{ij,t} that encode the
pairwise relation change from A_t to A_{t+1}.

This module implements an EFEN variant that produces energies for the 9 edit types:

  prev ∈ {NONE(0), I_TO_J(1), J_TO_I(2)}
  next ∈ {NONE(0), I_TO_J(1), J_TO_I(2)}
  token_id = 3 * prev + next  ∈ [0..8]

Energy for a token involving agents i and j:

  ε = ε0 + Σ_{u∈{i,j}} [ -μ_u^T φ_u + (λ_u/2) φ_u^T Σ_u φ_u ]

We assume diagonal Σ_u (from log-variance) for stability.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


N_REL_STATES = 3
N_EDIT_TOKENS = 9


@dataclass
class EditTokenEnergies:
    """Per-pair edit token energies.

    Attributes:
      e_edit:    [B,K,K,9] energies for each (prev,next) edit token.
      pair_mask: [B,K,K] bool mask for valid (i!=j, both present) pairs.

    Optional (for PCI factorized reweighting):
      base_edit:  [B,K,K,9]
      phi_edit_i: [B,K,K,9,Dz]
      phi_edit_j: [B,K,K,9,Dz]
    """

    e_edit: torch.Tensor
    pair_mask: torch.Tensor

    base_edit: Optional[torch.Tensor] = None
    phi_edit_i: Optional[torch.Tensor] = None
    phi_edit_j: Optional[torch.Tensor] = None


class EditFactorizedEnergyNetEdit(nn.Module):
    """Edit-Factorized Energy Network (EFEN) for EB-STM transitions."""

    def __init__(
        self,
        agent_feat_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        lambda_u: float = 1.0,
        return_token_params: bool = False,
    ) -> None:
        super().__init__()
        self.agent_feat_dim = int(agent_feat_dim)
        self.z_dim = int(z_dim)
        self.lambda_u = float(lambda_u)
        self.return_token_params = bool(return_token_params)

        # Geometry features: rel_pos(2), dist(1), rel_vel(2), yaw_diff(1)
        geom_dim = 6
        pair_dim = 2 * self.agent_feat_dim + geom_dim

        self.trunk = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Produce base energies and preference-demand vectors for all 9 edit token types.
        self.base_head = nn.Linear(hidden_dim, N_EDIT_TOKENS)
        self.phi_head = nn.Linear(hidden_dim, N_EDIT_TOKENS * 2 * self.z_dim)

        # Initialize base energies near 0.
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)

    def _pref_term(self, mu: torch.Tensor, logvar: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Compute -μ^T φ + (λ_u/2) φ^T Σ φ for diagonal Σ."""
        var = torch.exp(logvar)
        mean_align = -(mu * phi).sum(dim=-1)
        unc_pen = 0.5 * self.lambda_u * ((phi**2) * var).sum(dim=-1)
        return mean_align + unc_pen

    def forward(
        self,
        agent_feat: torch.Tensor,  # [B,K,D]
        z_mean: torch.Tensor,      # [B,K,Dz]
        z_logvar: torch.Tensor,    # [B,K,Dz]
        agent_mask: torch.Tensor,  # [B,K] float/bool
    ) -> EditTokenEnergies:
        B, K, D = agent_feat.shape
        assert D == self.agent_feat_dim, f"agent_feat_dim mismatch: expected {self.agent_feat_dim}, got {D}"

        m = agent_mask > 0.5
        pair_mask = (m.unsqueeze(2) & m.unsqueeze(1))
        eye = torch.eye(K, device=agent_feat.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

        feat_i = agent_feat.unsqueeze(2).expand(B, K, K, D)
        feat_j = agent_feat.unsqueeze(1).expand(B, K, K, D)

        # geometry
        pos_i = feat_i[..., 0:2]
        pos_j = feat_j[..., 0:2]
        rel_pos = pos_j - pos_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        vel_i = feat_i[..., 3:5]
        vel_j = feat_j[..., 3:5]
        rel_vel = vel_j - vel_i

        yaw_i = feat_i[..., 2]
        yaw_j = feat_j[..., 2]
        dyaw = torch.atan2(torch.sin(yaw_j - yaw_i), torch.cos(yaw_j - yaw_i)).unsqueeze(-1)

        geom = torch.cat([rel_pos, dist, rel_vel, dyaw], dim=-1)  # [B,K,K,6]
        pair_feat = torch.cat([feat_i, feat_j, geom], dim=-1)     # [B,K,K,pair_dim]

        h = self.trunk(pair_feat)  # [B,K,K,H]
        base = self.base_head(h)   # [B,K,K,9]
        phi = self.phi_head(h).view(B, K, K, N_EDIT_TOKENS, 2, self.z_dim)
        phi_i = phi[..., 0, :]
        phi_j = phi[..., 1, :]

        # Broadcast beliefs
        mu_i = z_mean.unsqueeze(2).expand(B, K, K, self.z_dim)
        mu_j = z_mean.unsqueeze(1).expand(B, K, K, self.z_dim)
        lv_i = z_logvar.unsqueeze(2).expand(B, K, K, self.z_dim)
        lv_j = z_logvar.unsqueeze(1).expand(B, K, K, self.z_dim)

        pref_i = self._pref_term(mu_i.unsqueeze(3), lv_i.unsqueeze(3), phi_i)  # [B,K,K,9]
        pref_j = self._pref_term(mu_j.unsqueeze(3), lv_j.unsqueeze(3), phi_j)  # [B,K,K,9]
        e_edit = base + pref_i + pref_j

        # apply pair mask
        e_edit = e_edit * pair_mask.unsqueeze(-1).to(dtype=e_edit.dtype)

        if self.return_token_params:
            return EditTokenEnergies(
                e_edit=e_edit,
                pair_mask=pair_mask,
                base_edit=base,
                phi_edit_i=phi_i,
                phi_edit_j=phi_j,
            )

        return EditTokenEnergies(e_edit=e_edit, pair_mask=pair_mask)
