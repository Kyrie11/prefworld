from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.gaussian import DiagGaussian, NaturalDiagGaussian


@dataclass
class PreferencePosterior:
    """Preference posterior for a batch of agents."""

    q: DiagGaussian          # [B, N, Dz]
    nat: NaturalDiagGaussian # natural params for debugging
    alpha: torch.Tensor      # [B, N, T] gate per evidence step


class EvidenceEncoder(nn.Module):
    """Encode per-agent history into per-step evidence for Gaussian natural-parameter updates."""

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_delta_eta = nn.Linear(hidden_dim, z_dim)
        self.to_delta_logLambda = nn.Linear(hidden_dim, z_dim)
        self.to_alpha = nn.Linear(hidden_dim, 1)

        # init small so updates start near prior
        nn.init.zeros_(self.to_delta_eta.weight)
        nn.init.zeros_(self.to_delta_eta.bias)
        nn.init.zeros_(self.to_delta_logLambda.weight)
        nn.init.zeros_(self.to_delta_logLambda.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward.
        Args:
            x: [B*N, T, Din]
            mask: [B*N, T] float (1 valid, 0 pad)
        Returns:
            delta_eta: [B*N, T, Dz]
            delta_Lambda: [B*N, T, Dz] (positive)
            alpha: [B*N, T] in [0,1]
        """
        # Pack sequence for speed and to ignore padding
        lengths = mask.sum(dim=1).clamp_min(1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])  # [B*N,T,H]

        delta_eta = self.to_delta_eta(out)
        delta_Lambda = F.softplus(self.to_delta_logLambda(out))  # ensure non-negative
        alpha = torch.sigmoid(self.to_alpha(out)).squeeze(-1)
        # apply mask
        alpha = alpha * mask
        return delta_eta, delta_Lambda, alpha


class PreferenceCompletion(nn.Module):
    """Streaming preference completion with diagonal Gaussian belief update."""

    def __init__(
        self,
        agent_input_dim: int,
        z_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 1,
        prior_logvar: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = EvidenceEncoder(agent_input_dim, hidden_dim, z_dim, num_layers=num_layers)

        # learned (or fixed) prior
        self.prior_mean = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior_logvar = nn.Parameter(torch.full((z_dim,), float(prior_logvar)), requires_grad=False)

    def prior(self, batch_shape: Tuple[int, ...], device=None, dtype=None) -> DiagGaussian:
        mean = self.prior_mean.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        logvar = self.prior_logvar.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        return DiagGaussian(mean=mean, logvar=logvar)

    def forward(
        self,
        agents_hist: torch.Tensor,       # [B, N, T, Din]
        agents_hist_mask: torch.Tensor,  # [B, N, T]
        drop_prob: float = 0.0,
    ) -> PreferencePosterior:
        B, N, T, Din = agents_hist.shape
        x = agents_hist.reshape(B * N, T, Din)
        m = agents_hist_mask.reshape(B * N, T)

        if drop_prob > 0.0 and self.training:
            # evidence dropout on valid steps
            keep = (torch.rand_like(m) > drop_prob).float()
            m = m * keep

        delta_eta, delta_Lambda, alpha = self.encoder(x, m)  # [B*N,T,Dz], [B*N,T,Dz], [B*N,T]

        # Start from standard normal prior in natural form
        nat = NaturalDiagGaussian.standard_normal((B * N, self.z_dim), device=x.device, dtype=x.dtype)
        # Streaming accumulate across time steps
        for t in range(T):
            nat = nat.update(delta_eta[:, t], delta_Lambda[:, t], alpha[:, t])

        q = nat.to_moment()
        q = DiagGaussian(mean=q.mean.reshape(B, N, self.z_dim), logvar=q.logvar.reshape(B, N, self.z_dim))
        nat = NaturalDiagGaussian(eta=nat.eta.reshape(B, N, self.z_dim), Lambda=nat.Lambda.reshape(B, N, self.z_dim))
        alpha = alpha.reshape(B, N, T)
        return PreferencePosterior(q=q, nat=nat, alpha=alpha)
