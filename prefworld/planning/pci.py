from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from prefworld.models.eb_stm import EBSTM


def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def jensen_shannon(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen–Shannon divergence between categorical distributions.

    Args:
      p,q: [B,S] distributions (sum to 1)
    Returns:
      js: [B]
    """
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (_safe_log(p, eps) - _safe_log(m, eps)), dim=-1)
    kl_qm = torch.sum(q * (_safe_log(q, eps) - _safe_log(m, eps)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _normalize_log_weights(logw: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert log weights to a probability simplex along the last dim."""
    logw = logw - torch.max(logw, dim=-1, keepdim=True).values
    w = torch.exp(logw)
    return w / (w.sum(dim=-1, keepdim=True) + eps)


def compute_cvar(values: torch.Tensor, alpha: float = 0.9, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute weighted CVaR of 'loss-like' values in the upper tail.

    Args:
      values:  [B,S]
      alpha:   tail quantile (e.g., 0.9 keeps worst 10%)
      weights: [B,S] simplex (optional). If None, uniform weights.
    Returns:
      cvar:    [B]
    """
    B, S = values.shape
    tail = float(1.0 - alpha)
    if tail <= 0.0:
        return values.max(dim=-1).values

    if weights is None:
        weights = torch.full_like(values, 1.0 / float(S))
    else:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

    # sort descending by values (worst first)
    v_sorted, idx = torch.sort(values, dim=-1, descending=True)
    w_sorted = weights.gather(dim=-1, index=idx)

    cumsum = torch.cumsum(w_sorted, dim=-1)
    in_tail = cumsum <= tail
    boundary = torch.argmax((cumsum >= tail).to(torch.int64), dim=-1)  # [B]

    w_tail = w_sorted * in_tail.to(dtype=w_sorted.dtype)
    b = torch.arange(B, device=values.device)
    w_before = torch.where(boundary > 0, cumsum[b, boundary - 1], torch.zeros((B,), device=values.device, dtype=values.dtype))
    w_part = (tail - w_before).clamp(min=0.0)
    w_tail[b, boundary] = w_tail[b, boundary] + w_part

    denom = w_tail.sum(dim=-1).clamp_min(1e-12)
    return (v_sorted * w_tail).sum(dim=-1) / denom


@dataclass
class PCIResult:
    pci: torch.Tensor            # [B,K]
    tiebreak_risk: torch.Tensor  # [B,K]
    p_rollout: torch.Tensor      # [B,S] rollout distribution under full belief
    rollouts: torch.Tensor       # [B,S,H,K,K]
    logp_full: torch.Tensor      # [B,S]


def _pref_term(mu: torch.Tensor, logvar: torch.Tensor, phi: torch.Tensor, lambda_u: float) -> torch.Tensor:
    """-μ^T φ + (λ_u/2) φ^T Σ φ, assuming diagonal Σ."""
    var = torch.exp(logvar)
    mean_align = -(mu * phi).sum(dim=-1)
    unc_pen = 0.5 * float(lambda_u) * ((phi**2) * var).sum(dim=-1)
    return mean_align + unc_pen


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
    # Rollout support
    horizon_steps: int = 5,
    num_rollouts: int = 32,
    use_beam_rollout: bool = False,
    beam_size: int = 16,
    # Divergence estimation
    use_model_probs: bool = True,
    # Tie-break risk (CVaR of interaction count with ego)
    cvar_alpha: float = 0.9,
    # Paper Eq.(30): epsilon smoothing for beam/search rollouts
    smooth_eps: float = 1e-6,
    dt: float = 0.5,
) -> PCIResult:
    """Compute PCI using shared rollout support + *factorized* counterfactual reweighting.

    Compared to a naive implementation that re-runs EB-STM for each counterfactual agent,
    this version:
      1) samples/beam-searches rollouts once under the full belief;
      2) computes per-rollout preference-information deltas Δ^{(j)} using EFEN token φ;
      3) obtains counterfactual rollout distributions by importance reweighting:
           q_j(R_k) ∝ p(R_k) * exp(Δ^{(j)}_k / T_eb)

    This matches the paper's Proposition 1 when the EB-STM rollout score is an energy sum.
    (Here we approximate p(R_k) using the model's rollout log-probabilities.)
    """
    B, K, _ = A_t.shape
    H = int(horizon_steps)

    # --- Shared rollout support under full belief ---
    if use_beam_rollout:
        rollouts, logp_full = ebstm.beam_rollout(
            A_t,
            agent_feat,
            z_mean,
            z_logvar,
            agent_mask,
            max_horizon_steps=H,
            beam_size=int(beam_size),
            entropy_stop_threshold=None,
            min_horizon_steps=1,
        )
    else:
        rollouts, logp_full, _ = ebstm.rollout(
            A_t,
            agent_feat,
            z_mean,
            z_logvar,
            agent_mask,
            horizon_steps=H,
            num_samples=int(num_rollouts),
            return_indices=False,
        )

    S = rollouts.shape[1]

    # rollout distribution under full belief
    if use_model_probs:
        p_full = _normalize_log_weights(logp_full)  # [B,S]
        log_p_full = _safe_log(p_full)              # [B,S]
    else:
        p_full = torch.full((B, S), 1.0 / float(S), device=A_t.device, dtype=torch.float32)
        log_p_full = _safe_log(p_full)

    pci = torch.zeros((B, K), device=A_t.device, dtype=torch.float32)
    risk = torch.zeros((B, K), device=A_t.device, dtype=torch.float32)

    # --- EFEN token params for factorized Δ computation ---
    pair = ebstm.energy_net(agent_feat, z_mean, z_logvar, agent_mask)
    if pair.phi_edit_i is None or pair.phi_edit_j is None:
        raise RuntimeError("Edit EFEN must be created with return_token_params=True to use factorized PCI reweighting.")

    lambda_u = getattr(ebstm.energy_net, "lambda_u", 1.0)
    pair_mask = pair.pair_mask  # [B,K,K] bool

    # full belief expanded to [B,K,K,Dz] for i/j roles
    mu_i = z_mean.unsqueeze(2).expand(B, K, K, -1)
    lv_i = z_logvar.unsqueeze(2).expand(B, K, K, -1)
    mu_j = z_mean.unsqueeze(1).expand(B, K, K, -1)
    lv_j = z_logvar.unsqueeze(1).expand(B, K, K, -1)

    # baseline belief (same for all agents)
    mu0 = torch.full_like(z_mean, float(baseline_mean))
    lv0 = torch.full_like(z_logvar, float(baseline_logvar))
    mu0_i = mu0.unsqueeze(2).expand(B, K, K, -1)
    lv0_i = lv0.unsqueeze(2).expand(B, K, K, -1)
    mu0_j = mu0.unsqueeze(1).expand(B, K, K, -1)
    lv0_j = lv0.unsqueeze(1).expand(B, K, K, -1)

    # diff preference term per ordered pair **edit token** and participant role
    # phi_edit_*: [B,K,K,9,Dz]
    diff_edit_i = _pref_term(mu_i.unsqueeze(3), lv_i.unsqueeze(3), pair.phi_edit_i, lambda_u) - _pref_term(
        mu0_i.unsqueeze(3), lv0_i.unsqueeze(3), pair.phi_edit_i, lambda_u
    )  # [B,K,K,9]
    diff_edit_j = _pref_term(mu_j.unsqueeze(3), lv_j.unsqueeze(3), pair.phi_edit_j, lambda_u) - _pref_term(
        mu0_j.unsqueeze(3), lv0_j.unsqueeze(3), pair.phi_edit_j, lambda_u
    )  # [B,K,K,9]

    pm = pair_mask.to(dtype=torch.float32)
    diff_edit_i = diff_edit_i * pm.unsqueeze(-1)
    diff_edit_j = diff_edit_j * pm.unsqueeze(-1)

    triu = torch.triu(torch.ones((K, K), device=A_t.device, dtype=torch.bool), diagonal=1)
    valid_ut = (pair_mask & triu).unsqueeze(1)  # [B,1,K,K]

    # --- Compute Δ_rollout[b,s,k] for all agents k ---
    delta = torch.zeros((B, S, K), device=A_t.device, dtype=torch.float32)

    for h in range(H):
        A_next = rollouts[:, :, h]  # [B,S,K,K]
        if h == 0:
            A_prev = A_t.unsqueeze(1).expand(B, S, K, K)
        else:
            A_prev = rollouts[:, :, h - 1]

        prev_state = EBSTM._relation_state3(A_prev)  # [B,S,K,K] in {0,1,2,3}
        next_state = EBSTM._relation_state3(A_next)  # [B,S,K,K]
        # clamp invalid to keep gather safe; invalid transitions contribute ~0 in delta.
        prev_state = prev_state.clamp_max(2)
        next_state = next_state.clamp_max(2)
        tok = (prev_state * 3 + next_state).clamp(min=0, max=8)  # [B,S,K,K]

        # gather per-pair diff contributions for this token type
        # NOTE: torch.gather does **not** broadcast the rollout/sample dimension.
        di = diff_edit_i.unsqueeze(1).expand(-1, S, -1, -1, -1)  # [B,S,K,K,9]
        dj = diff_edit_j.unsqueeze(1).expand(-1, S, -1, -1, -1)
        tok_idx = tok.unsqueeze(-1)
        g_i = torch.gather(di, dim=-1, index=tok_idx).squeeze(-1)  # [B,S,K,K]
        g_j = torch.gather(dj, dim=-1, index=tok_idx).squeeze(-1)  # [B,S,K,K]

        # only count unordered pairs once
        g_i = g_i * valid_ut.to(dtype=g_i.dtype)
        g_j = g_j * valid_ut.to(dtype=g_j.dtype)

        contrib_i = g_i.sum(dim=-1)  # sum over j -> [B,S,K]
        contrib_j = g_j.sum(dim=-2)  # sum over i -> [B,S,K]
        delta = delta + contrib_i + contrib_j

    # --- For each agent j: importance reweighting ---
    T_eb = float(getattr(ebstm, "temperature", 1.0))
    ego = 0

    for j in range(K):
        if j == 0 and not include_ego:
            continue
        valid_j = agent_mask[:, j] > 0.5
        if valid_j.sum() == 0:
            continue

        logw = log_p_full + (delta[:, :, j] / max(1e-6, T_eb))
        q = _normalize_log_weights(logw)

        # Paper Eq.(30): compute divergences on epsilon-smoothed discrete distributions
        if float(smooth_eps) > 0.0:
            p_js = (p_full + float(smooth_eps)) / (p_full.sum(dim=-1, keepdim=True) + float(S) * float(smooth_eps))
            q_js = (q + float(smooth_eps)) / (q.sum(dim=-1, keepdim=True) + float(S) * float(smooth_eps))
        else:
            p_js, q_js = p_full, q

        pci[:, j] = jensen_shannon(p_js, q_js)

        # tie-break risk: paper Eq.(29) uses |CVaR(indiv) - CVaR(baseline-j)|.
        # We approximate trajectories with constant velocity and yield-conditioned slowdown.
        vel_ego = agent_feat[:, ego, 3:5]  # [B,2]
        tvec = (torch.arange(1, H + 1, device=A_t.device, dtype=torch.float32) * float(dt)).view(1, H, 1)
        ego_xy = tvec * vel_ego.unsqueeze(1)  # [B,H,2] (ego starts at origin)

        pos_j0 = agent_feat[:, j, 0:2]  # [B,2]
        vel_j0 = agent_feat[:, j, 3:5]  # [B,2]
        # radius proxy from size if present
        if agent_feat.shape[-1] >= 7:
            radius_j = 0.5 * torch.maximum(agent_feat[:, j, 5].clamp_min(0.1), agent_feat[:, j, 6].clamp_min(0.1))
        else:
            radius_j = torch.full((B,), 1.5, device=A_t.device, dtype=torch.float32)

        slowdown = 0.4
        pos = pos_j0.unsqueeze(1).expand(B, S, 2).clone()
        scale = torch.ones((B, S), device=A_t.device, dtype=torch.float32)
        traj = torch.zeros((B, S, H, 2), device=A_t.device, dtype=torch.float32)
        for h in range(H):
            yield_mask = (rollouts[:, :, h, j, ego] > 0.5)  # [B,S]
            scale = torch.minimum(scale, torch.where(yield_mask, torch.full_like(scale, slowdown), torch.ones_like(scale)))
            pos = pos + vel_j0.unsqueeze(1) * scale.unsqueeze(-1) * float(dt)
            traj[:, :, h, :] = pos

        d = torch.norm(traj - ego_xy.unsqueeze(1), dim=-1)  # [B,S,H]
        dmin = d.min(dim=-1).values
        thr = 2.0 + radius_j.unsqueeze(1)
        coll = (dmin < thr).to(torch.float32)
        # If agent is invalid for a batch element, set collision loss to 0.
        coll = coll * valid_j.to(torch.float32).unsqueeze(1)
        cvar_indiv = compute_cvar(coll, alpha=float(cvar_alpha), weights=p_full)
        cvar_basej = compute_cvar(coll, alpha=float(cvar_alpha), weights=q)
        risk[:, j] = torch.abs(cvar_indiv - cvar_basej)

    return PCIResult(pci=pci, tiebreak_risk=risk, p_rollout=p_full, rollouts=rollouts, logp_full=logp_full)
