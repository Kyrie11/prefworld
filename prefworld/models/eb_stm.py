from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.efen import EditFactorizedEnergyNet, PairEnergies


@dataclass
class EBSTMOutput:
    """Output of a one-step EB-STM transition."""

    candidate_A: torch.Tensor   # [B,C,K,K]
    candidate_mask: torch.Tensor  # [B,C] bool
    energies: torch.Tensor      # [B,C]
    probs: torch.Tensor         # [B,C]
    pair_energies: PairEnergies # pair energies for the factual belief (for reweighting)


def _unique_adjacency(candidates: List[torch.Tensor]) -> List[torch.Tensor]:
    """Remove duplicates adjacency matrices (by exact equality)."""
    uniq: List[torch.Tensor] = []
    for A in candidates:
        found = False
        for U in uniq:
            if torch.equal(A, U):
                found = True
                break
        if not found:
            uniq.append(A)
    return uniq


class EBSTM(nn.Module):
    """Energy-Based Structured Transition Model (one-step + optional rollout)."""

    def __init__(
        self,
        energy_net: EditFactorizedEnergyNet,
        temperature: float = 1.0,
        max_candidates: int = 64,
        edit_dist_threshold: float = 30.0,
    ):
        super().__init__()
        self.energy_net = energy_net
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.edit_dist_threshold = edit_dist_threshold

    @staticmethod
    def _pair_distance(agent_feat: torch.Tensor) -> torch.Tensor:
        # agent_feat: [B,K,Da], assume x,y at 0,1
        pos = agent_feat[..., 0:2]
        rel = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.norm(rel, dim=-1)  # [B,K,K]
        return dist

    def generate_candidates(
        self,
        A_t: torch.Tensor,        # [B,K,K]
        agent_feat: torch.Tensor, # [B,K,Da]
        agent_mask: torch.Tensor, # [B,K]
        oracle_next: Optional[torch.Tensor] = None,  # [B,K,K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a candidate set of next adjacency matrices per batch item.

        For simplicity, we generate candidates independently per batch element; we then pad to max_candidates.
        Returns:
          candidate_A:   [B,C,K,K]
          candidate_mask:[B,C] (True for valid candidate)
        """
        B, K, _ = A_t.shape
        dist = self._pair_distance(agent_feat)  # [B,K,K]
        candidates_all: List[List[torch.Tensor]] = []

        for b in range(B):
            Ab = A_t[b]
            mb = agent_mask[b]
            db = dist[b]
            cand: List[torch.Tensor] = [Ab.clone()]

            # enumerate close valid pairs
            pairs: List[Tuple[int, int]] = []
            for i in range(K):
                if mb[i] < 0.5:
                    continue
                for j in range(i + 1, K):
                    if mb[j] < 0.5:
                        continue
                    if float(db[i, j]) <= self.edit_dist_threshold:
                        pairs.append((i, j))

            # generate edits
            for (i, j) in pairs:
                if len(cand) >= self.max_candidates:
                    break
                a_ij = int(Ab[i, j].item())
                a_ji = int(Ab[j, i].item())
                if a_ij == 0 and a_ji == 0:
                    # add either direction
                    A1 = Ab.clone()
                    A1[i, j] = 1
                    A1[j, i] = 0
                    cand.append(A1)
                    if len(cand) >= self.max_candidates:
                        break
                    A2 = Ab.clone()
                    A2[j, i] = 1
                    A2[i, j] = 0
                    cand.append(A2)
                else:
                    # delete
                    A1 = Ab.clone()
                    A1[i, j] = 0
                    A1[j, i] = 0
                    cand.append(A1)
                    if len(cand) >= self.max_candidates:
                        break
                    # flip if directed
                    if a_ij == 1 and a_ji == 0:
                        A2 = Ab.clone()
                        A2[i, j] = 0
                        A2[j, i] = 1
                        cand.append(A2)
                    elif a_ji == 1 and a_ij == 0:
                        A2 = Ab.clone()
                        A2[j, i] = 0
                        A2[i, j] = 1
                        cand.append(A2)

            cand = _unique_adjacency(cand)

            # Ensure oracle is included and never truncated away.
            if oracle_next is not None:
                A_or = oracle_next[b].clone()
                # Reorder: [A_t, A_or, ...]
                new: List[torch.Tensor] = []
                new.append(Ab.clone())
                if not torch.equal(A_or, new[0]):
                    new.append(A_or)
                for A in cand:
                    if torch.equal(A, new[0]) or (len(new) > 1 and torch.equal(A, new[1])):
                        continue
                    new.append(A)
                    if len(new) >= self.max_candidates:
                        break
                cand = new
            else:
                cand = cand[: self.max_candidates]

            candidates_all.append(cand)

        # pad to [B,C,K,K]
        C = max(len(c) for c in candidates_all)
        C = min(C, self.max_candidates)
        out = torch.zeros((B, C, K, K), device=A_t.device, dtype=A_t.dtype)
        out_mask = torch.zeros((B, C), device=A_t.device, dtype=torch.bool)
        for b, cand in enumerate(candidates_all):
            n = min(C, len(cand))
            for c_idx in range(n):
                out[b, c_idx] = cand[c_idx]
            out_mask[b, :n] = True
        return out, out_mask

    @staticmethod
    def energy_of_structure(pair: PairEnergies, A: torch.Tensor) -> torch.Tensor:
        """Compute total energy ε(A) by summing token energies over unordered pairs.

        pair: PairEnergies with e_dir/e_none and pair_mask
        A:    [B,K,K] adjacency (0/1)
        Returns:
          E: [B]
        """
        e_dir, e_none, pm = pair.e_dir, pair.e_none, pair.pair_mask
        B, K, _ = A.shape
        E = torch.zeros((B,), device=A.device, dtype=torch.float32)
        A_bin = (A > 0.5)
        for i in range(K):
            for j in range(i + 1, K):
                valid = pm[:, i, j]
                if not valid.any():
                    continue
                # choose directed vs none
                a_ij = A_bin[:, i, j]
                a_ji = A_bin[:, j, i]
                e = torch.where(a_ij, e_dir[:, i, j], torch.where(a_ji, e_dir[:, j, i], e_none[:, i, j]))
                E = E + e.to(dtype=E.dtype) * valid.to(dtype=E.dtype)
        return E

    def forward(
        self,
        A_t: torch.Tensor,        # [B,K,K]
        agent_feat: torch.Tensor, # [B,K,Da]
        z_mean: torch.Tensor,     # [B,K,Dz]
        z_logvar: torch.Tensor,   # [B,K,Dz]
        agent_mask: torch.Tensor, # [B,K]
        oracle_next: Optional[torch.Tensor] = None,  # [B,K,K]
    ) -> EBSTMOutput:
        candidate_A, cand_mask = self.generate_candidates(A_t, agent_feat, agent_mask, oracle_next=oracle_next)
        B, C, K, _ = candidate_A.shape

        pair = self.energy_net(agent_feat, z_mean, z_logvar, agent_mask)  # [B,K,K] pair energies for factual
        # compute energy for each candidate
        energies = []
        for c in range(C):
            E = self.energy_of_structure(pair, candidate_A[:, c])
            energies.append(E)
        energies = torch.stack(energies, dim=1)  # [B,C]

        # mask out padded candidates
        logits = -energies / float(self.temperature)
        logits = logits.masked_fill(~cand_mask, -1e9)
        probs = torch.softmax(logits, dim=1)
        return EBSTMOutput(candidate_A=candidate_A, candidate_mask=cand_mask, energies=energies, probs=probs, pair_energies=pair)

    @staticmethod
    def oracle_index(candidate_A: torch.Tensor, oracle_next: torch.Tensor) -> torch.Tensor:
        """Return index of oracle_next in candidate_A (assumes exists)."""
        B, C, K, _ = candidate_A.shape
        idx = torch.zeros((B,), device=candidate_A.device, dtype=torch.int64)
        for b in range(B):
            found = 0
            for c in range(C):
                if torch.equal(candidate_A[b, c], oracle_next[b]):
                    found = c
                    break
            idx[b] = found
        return idx

    def rollout(
        self,
        A0: torch.Tensor,
        agent_feat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        agent_mask: torch.Tensor,
        horizon_steps: int = 5,
        num_samples: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample rollouts of interaction structures.

        Returns:
          A_samples: [B, num_samples, horizon_steps, K, K]
          logp: [B, num_samples] (log prob under the EB-STM transitions)
        """
        B, K, _ = A0.shape
        A_samples = torch.zeros((B, num_samples, horizon_steps, K, K), device=A0.device, dtype=A0.dtype)
        logp = torch.zeros((B, num_samples), device=A0.device, dtype=torch.float32)

        for s in range(num_samples):
            A_t = A0
            lp = torch.zeros((B,), device=A0.device, dtype=torch.float32)
            for h in range(horizon_steps):
                out = self.forward(A_t, agent_feat, z_mean, z_logvar, agent_mask, oracle_next=None)
                # sample candidate
                cat = torch.distributions.Categorical(probs=out.probs)
                idx = cat.sample()  # [B]
                # gather
                A_next = out.candidate_A[torch.arange(B, device=A0.device), idx]
                A_samples[:, s, h] = A_next
                lp = lp + torch.log(out.probs[torch.arange(B, device=A0.device), idx] + 1e-8)
                A_t = A_next
            logp[:, s] = lp
        return A_samples, logp
