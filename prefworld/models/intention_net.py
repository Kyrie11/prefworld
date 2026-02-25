from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import NUM_MANEUVERS


class PolylineEncoder(nn.Module):
    """Encode a set of polylines (map or ego future) into a fixed-size embedding."""

    def __init__(self, point_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, polylines: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
        polylines: [B, M, L, point_dim]
        mask: [B, M] (1 valid, 0 pad)
        Returns:
          emb: [B, out_dim]
        """
        B, M, L, D = polylines.shape
        x = self.mlp(polylines)  # [B,M,L,H]
        x = x.max(dim=2).values  # [B,M,H]
        x = self.out(x)          # [B,M,out]
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            x = x.sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        return x


class AgentHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hist: [B*N, T, Din], mask: [B*N,T]."""
        # nuPlan agent tracks may appear mid-history, so masks are often *suffix*-valid
        # (e.g., 0 0 1 1 1). pack_padded_sequence expects *prefix*-valid masks.
        # We reverse time so valid steps become a prefix (1 1 1 0 0).
        hist = torch.flip(hist, dims=[1])
        mask = torch.flip(mask, dims=[1])
        lengths = mask.sum(dim=1).clamp_min(1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(hist, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed)
        # h: [1, B*N, H]
        h = h[-1]
        return self.proj(h)


class IntentionNet(nn.Module):
    """Predict per-agent maneuver distribution conditioned on preference and template context."""

    def __init__(
        self,
        agent_input_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        ctx_dim: int = 128,
        tau_dim: int = 0,
        num_maneuvers: int = NUM_MANEUVERS,
    ):
        super().__init__()
        self.agent_enc = AgentHistoryEncoder(agent_input_dim, hidden_dim, ctx_dim)
        self.map_enc = PolylineEncoder(point_dim=2, hidden_dim=hidden_dim, out_dim=ctx_dim)
        self.ego_plan_enc = PolylineEncoder(point_dim=3, hidden_dim=hidden_dim, out_dim=ctx_dim)

        self.tau_dim = int(tau_dim)

        in_dim = ctx_dim + z_dim + ctx_dim + ctx_dim + self.tau_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_maneuvers),
        )

    def forward(
        self,
        agents_hist: torch.Tensor,        # [B,N,T,Din]
        agents_hist_mask: torch.Tensor,   # [B,N,T]
        z: torch.Tensor,                  # [B,N,Dz]
        map_polylines: torch.Tensor,      # [B,M,L,2]
        map_poly_mask: torch.Tensor,      # [B,M]
        ego_future: torch.Tensor,         # [B,Tf,3] ego candidate plan in ego-local coords
        tau_curr: Optional[torch.Tensor] = None,  # [B,N,Dt] agent-centric template embedding at current step
    ) -> torch.Tensor:
        B, N, T, Din = agents_hist.shape
        # agent embedding
        a = agents_hist.reshape(B * N, T, Din)
        m = agents_hist_mask.reshape(B * N, T)
        agent_emb = self.agent_enc(a, m).reshape(B, N, -1)

        # context embeddings
        map_emb = self.map_enc(map_polylines, mask=map_poly_mask)  # [B,ctx]
        ego_poly = ego_future.unsqueeze(1)  # [B,1,Tf,3]
        ego_mask = torch.ones((B, 1), device=ego_future.device, dtype=ego_future.dtype)
        ego_emb = self.ego_plan_enc(ego_poly, mask=ego_mask)  # [B,ctx]

        map_emb_exp = map_emb.unsqueeze(1).expand(B, N, map_emb.shape[-1])
        ego_emb_exp = ego_emb.unsqueeze(1).expand(B, N, ego_emb.shape[-1])

        if self.tau_dim > 0:
            assert tau_curr is not None, "tau_curr must be provided when tau_dim>0"
            x = torch.cat([agent_emb, z, map_emb_exp, ego_emb_exp, tau_curr], dim=-1)
        else:
            x = torch.cat([agent_emb, z, map_emb_exp, ego_emb_exp], dim=-1)
        logits = self.head(x)  # [B,N,K]
        return logits
