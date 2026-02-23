from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class DiagGaussian:
    """Diagonal-covariance Gaussian in moment form."""

    mean: torch.Tensor  # [..., D]
    logvar: torch.Tensor  # [..., D]

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.logvar)

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        return self.mean + eps * self.std

    def kl_to_standard_normal(self) -> torch.Tensor:
        """KL(q || N(0,I)) for diagonal Gaussians."""
        return 0.5 * torch.sum(torch.exp(self.logvar) + self.mean**2 - 1.0 - self.logvar, dim=-1)

    def kl_to(self, other: "DiagGaussian") -> torch.Tensor:
        """KL(self || other)."""
        v0 = torch.exp(self.logvar)
        v1 = torch.exp(other.logvar)
        term = (v0 + (self.mean - other.mean) ** 2) / (v1 + 1e-8)
        kl = 0.5 * torch.sum(other.logvar - self.logvar + term - 1.0, dim=-1)
        return kl


@dataclass
class NaturalDiagGaussian:
    """Diagonal Gaussian in natural-parameter form: precision diag Lambda and eta=Lambda*mu."""

    eta: torch.Tensor  # [..., D]
    Lambda: torch.Tensor  # [..., D] precision diag (positive)

    @staticmethod
    def standard_normal(shape: Tuple[int, ...], device=None, dtype=None) -> "NaturalDiagGaussian":
        D = shape[-1]
        eta = torch.zeros(shape, device=device, dtype=dtype)
        Lambda = torch.ones(shape, device=device, dtype=dtype)
        return NaturalDiagGaussian(eta=eta, Lambda=Lambda)

    def to_moment(self) -> DiagGaussian:
        var = 1.0 / (self.Lambda + 1e-8)
        mean = var * self.eta
        logvar = torch.log(var + 1e-8)
        return DiagGaussian(mean=mean, logvar=logvar)

    @staticmethod
    def from_moment(q: DiagGaussian) -> "NaturalDiagGaussian":
        var = torch.exp(q.logvar)
        Lambda = 1.0 / (var + 1e-8)
        eta = Lambda * q.mean
        return NaturalDiagGaussian(eta=eta, Lambda=Lambda)

    def update(self, delta_eta: torch.Tensor, delta_Lambda: torch.Tensor, alpha: torch.Tensor) -> "NaturalDiagGaussian":
        """Streaming update: eta += alpha * delta_eta; Lambda += alpha * delta_Lambda (delta must be >=0)."""
        # alpha broadcast to [..., 1] if needed
        while alpha.dim() < self.eta.dim():
            alpha = alpha.unsqueeze(-1)
        new_eta = self.eta + alpha * delta_eta
        new_Lambda = self.Lambda + alpha * delta_Lambda
        return NaturalDiagGaussian(eta=new_eta, Lambda=new_Lambda.clamp_min(1e-6))
