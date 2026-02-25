from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairEnergies:
    """Pairwise edit token energies.

    - e_dir[i,j] corresponds to the directed relation i -> j.
    - e_none[i,j] is symmetric and corresponds to the undirected NONE relation.

    For interpretability / calibration we optionally also expose the preference-demand vectors φ
    and preference-independent base energies ε0.
    """

    e_dir: torch.Tensor                 # [B,K,K]
    e_none: torch.Tensor                # [B,K,K]
    pair_mask: torch.Tensor             # [B,K,K] bool

    # Optional debugging / interpretability outputs
    base_dir: Optional[torch.Tensor] = None          # [B,K,K]
    base_none: Optional[torch.Tensor] = None         # [B,K,K]
    phi_dir_i: Optional[torch.Tensor] = None         # [B,K,K,Dz]
    phi_dir_j: Optional[torch.Tensor] = None         # [B,K,K,Dz]
    phi_none_i: Optional[torch.Tensor] = None        # [B,K,K,Dz]
    phi_none_j: Optional[torch.Tensor] = None        # [B,K,K,Dz]


class EditFactorizedEnergyNet(nn.Module):
    """Edit-Factorized Energy Network (EFEN) with explicit structured token energy.

    This implementation matches the paper's key design goal:
      - preference-independent base token energy ε0(τ, a)
      - preference-dependent term via demand vectors φ in latent preference space
      - uncertainty calibration through an analytic quadratic penalty (marginalization)

    For a token involving agents i and j:
      ε = ε0 + Σ_{u∈{i,j}} [ -μ_u^T φ_u + (λ_u/2) φ_u^T Σ_u φ_u ]

    We use a diagonal Σ_u (from log-variance) for stability.
    """

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

        # Directed token parameters for relation i -> j
        self.dir_trunk = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dir_base = nn.Linear(hidden_dim, 1)
        self.dir_phi = nn.Linear(hidden_dim, 2 * self.z_dim)  # (φ_i, φ_j)

        # NONE token parameters (computed per ordered pair then symmetrized)
        self.none_trunk = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.none_base = nn.Linear(hidden_dim, 1)
        self.none_phi = nn.Linear(hidden_dim, 2 * self.z_dim)

        # initialize base energies near 0
        nn.init.zeros_(self.dir_base.weight)
        nn.init.zeros_(self.dir_base.bias)
        nn.init.zeros_(self.none_base.weight)
        nn.init.zeros_(self.none_base.bias)

    def _pref_term(self, mu: torch.Tensor, logvar: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Compute -μ^T φ + (λ_u/2) φ^T Σ φ for diagonal Σ."""
        var = torch.exp(logvar)
        mean_align = -(mu * phi).sum(dim=-1)
        unc_pen = 0.5 * self.lambda_u * ((phi ** 2) * var).sum(dim=-1)
        return mean_align + unc_pen

    def forward(
        self,
        agent_feat: torch.Tensor,   # [B,K,D]
        z_mean: torch.Tensor,       # [B,K,Dz]
        z_logvar: torch.Tensor,     # [B,K,Dz]
        agent_mask: torch.Tensor,   # [B,K] float/bool
    ) -> PairEnergies:
        B, K, D = agent_feat.shape
        assert D == self.agent_feat_dim, f"agent_feat_dim mismatch: expected {self.agent_feat_dim}, got {D}"

        m = agent_mask > 0.5
        # pair mask excludes self-pairs
        pair_mask = (m.unsqueeze(2) & m.unsqueeze(1))
        eye = torch.eye(K, device=agent_feat.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

        # Build pair features
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
        pair_feat = torch.cat([feat_i, feat_j, geom], dim=-1)

        # Directed token params
        h_dir = self.dir_trunk(pair_feat)
        base_dir = self.dir_base(h_dir).squeeze(-1)  # [B,K,K]
        phi_dir = self.dir_phi(h_dir).view(B, K, K, 2, self.z_dim)
        phi_dir_i = phi_dir[..., 0, :]
        phi_dir_j = phi_dir[..., 1, :]

        # Preference terms
        mu_i = z_mean.unsqueeze(2).expand(B, K, K, self.z_dim)
        mu_j = z_mean.unsqueeze(1).expand(B, K, K, self.z_dim)
        lv_i = z_logvar.unsqueeze(2).expand(B, K, K, self.z_dim)
        lv_j = z_logvar.unsqueeze(1).expand(B, K, K, self.z_dim)

        pref_i = self._pref_term(mu_i, lv_i, phi_dir_i)
        pref_j = self._pref_term(mu_j, lv_j, phi_dir_j)
        e_dir = base_dir + pref_i + pref_j

        # NONE token params (ordered, then symmetrize energy)
        h_none = self.none_trunk(pair_feat)
        base_none_raw = self.none_base(h_none).squeeze(-1)
        phi_none = self.none_phi(h_none).view(B, K, K, 2, self.z_dim)
        phi_none_i = phi_none[..., 0, :]
        phi_none_j = phi_none[..., 1, :]
        pref_none_i = self._pref_term(mu_i, lv_i, phi_none_i)
        pref_none_j = self._pref_term(mu_j, lv_j, phi_none_j)
        e_none_raw = base_none_raw + pref_none_i + pref_none_j

        # symmetrize NONE energy
        e_none = 0.5 * (e_none_raw + e_none_raw.transpose(1, 2))
        base_none = 0.5 * (base_none_raw + base_none_raw.transpose(1, 2))

        # apply mask by zeroing out invalid pairs (energy_of_structure will ignore via pair_mask)
        e_dir = e_dir * pair_mask.to(dtype=e_dir.dtype)
        e_none = e_none * pair_mask.to(dtype=e_none.dtype)

        if self.return_token_params:
            # Note: φ are not symmetrized; for NONE tokens we still expose the ordered φ.
            return PairEnergies(
                e_dir=e_dir,
                e_none=e_none,
                pair_mask=pair_mask,
                base_dir=base_dir,
                base_none=base_none,
                phi_dir_i=phi_dir_i,
                phi_dir_j=phi_dir_j,
                phi_none_i=phi_none_i,
                phi_none_j=phi_none_j,
            )

        return PairEnergies(e_dir=e_dir, e_none=e_none, pair_mask=pair_mask)
