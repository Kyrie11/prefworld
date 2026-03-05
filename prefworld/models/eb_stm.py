from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.efen_edit import EditFactorizedEnergyNetEdit, EditTokenEnergies


@dataclass
class EBSTMOutput:
    """Output of a one-step EB-STM transition."""

    candidate_A: torch.Tensor        # [B,C,K,K]
    candidate_mask: torch.Tensor     # [B,C] bool
    energies: torch.Tensor           # [B,C]
    probs: torch.Tensor              # [B,C]
    edit_tokens: torch.Tensor         # [B,C,4] (i,j,prev_state,next_state)
    pair_energies: EditTokenEnergies  # factual pair edit-energies for reuse/debug


def _unique_adjacency(candidates: List[torch.Tensor]) -> List[torch.Tensor]:
    """Remove duplicates adjacency matrices (by exact equality)."""
    uniq: List[torch.Tensor] = []
    for A in candidates:
        if not any(torch.equal(A, U) for U in uniq):
            uniq.append(A)
    return uniq


class EBSTM(nn.Module):
    """Energy-Based Structured Transition Model (EB-STM).

    Implemented features:
      - Candidate generation via *templateized edit tokens* (single-edge add/delete/flip).
      - Energy evaluation via EFEN token energies (already includes preference uncertainty penalty).
      - Vectorized candidate energy computation (faster + less bug-prone).
      - Rollout sampling and **beam rollout**.
      - Predictability-adaptive horizon stopping (entropy threshold).
      - Optional extra structure-level preference-uncertainty penalty (can be enabled via scale>0).
    """

    def __init__(
        self,
        energy_net: EditFactorizedEnergyNetEdit,
        temperature: float = 1.0,
        max_candidates: int = 64,
        edit_dist_threshold: float = 30.0,
        enforce_acyclic: bool = True,
        # Candidate-pair expansion beyond a fixed distance threshold.
        # This approximates map/interaction-region semantics by using a constant-velocity
        # closest-approach check. It helps include pairs that are currently farther apart
        # but on a collision course.
        conflict_horizon_s: float = 3.0,
        conflict_dist_threshold: float = 6.0,
        use_closest_approach_candidates: bool = True,
        # ------------------------------------------------------------------
        # Candidate expansion (Req-5): allow *two-step* edit combinations.
        #
        # Motivation: the oracle next structure A_{t+Δ} may differ from A_t
        # by >1 unordered-pair edits. If the candidate set only allows a single
        # edit token, the oracle is often out-of-support.
        #
        # We implement a small beam search over edit tokens (depth=2) restricted
        # to the most relevant unordered pairs (top-k by pair relevance score).
        # The resulting candidates are merged with the standard one-step edits
        # and truncated to max_candidates.
        #
        # Set two_step_topk_pairs=0 to disable.
        # ------------------------------------------------------------------
        two_step_topk_pairs: int = 0,
        two_step_beam_size: int = 24,
        # Additional uncertainty penalty at the *structure* level (0 disables; EFEN already includes one).
        uncertainty_penalty_scale: float = 0.0,
    ):
        super().__init__()
        self.energy_net = energy_net
        self.temperature = float(temperature)
        self.max_candidates = int(max_candidates)
        self.edit_dist_threshold = float(edit_dist_threshold)
        self.enforce_acyclic = bool(enforce_acyclic)
        self.conflict_horizon_s = float(conflict_horizon_s)
        self.conflict_dist_threshold = float(conflict_dist_threshold)
        self.use_closest_approach_candidates = bool(use_closest_approach_candidates)
        self.two_step_topk_pairs = int(two_step_topk_pairs)
        self.two_step_beam_size = int(two_step_beam_size)
        self.uncertainty_penalty_scale = float(uncertainty_penalty_scale)


    @staticmethod
    def extract_edit_tokens(A_t: torch.Tensor, candidate_A: torch.Tensor) -> torch.Tensor:
        """Extract a compact edit token (i,j,prev_state,next_state) for each candidate.

        States are 3-way for each unordered pair (i<j):
          0: NONE
          1: i -> j
          2: j -> i

        This is primarily for debugging / interpretability.
        """
        B, C, K, _ = candidate_A.shape
        tokens = torch.zeros((B, C, 4), device=candidate_A.device, dtype=torch.int64)
        for b in range(B):
            base = A_t[b]
            for c in range(C):
                cand = candidate_A[b, c]
                if torch.equal(cand, base):
                    continue
                found = False
                for i in range(K):
                    for j in range(i + 1, K):
                        if cand[i, j] != base[i, j] or cand[j, i] != base[j, i]:
                            # prev / next states
                            prev = 0
                            if base[i, j] > 0.5 and base[j, i] < 0.5:
                                prev = 1
                            elif base[j, i] > 0.5 and base[i, j] < 0.5:
                                prev = 2

                            nxt = 0
                            if cand[i, j] > 0.5 and cand[j, i] < 0.5:
                                nxt = 1
                            elif cand[j, i] > 0.5 and cand[i, j] < 0.5:
                                nxt = 2

                            tokens[b, c, 0] = i
                            tokens[b, c, 1] = j
                            tokens[b, c, 2] = prev
                            tokens[b, c, 3] = nxt
                            found = True
                            break
                    if found:
                        break
        return tokens

    @staticmethod
    def _relation_state3(A: torch.Tensor) -> torch.Tensor:
        """3-state relation encoding for each ordered pair slot.

        For each (i,j):
          0: none, 1: i->j, 2: j->i, 3: both (invalid)
        Only the upper triangle is meaningful when used as an unordered-pair state.
        """
        a = (A > 0.5)
        at = a.transpose(-1, -2)
        typ = torch.zeros_like(A, dtype=torch.int64)
        typ = typ + (a & (~at)).to(torch.int64) * 1
        typ = typ + ((~a) & at).to(torch.int64) * 2
        typ = typ + (a & at).to(torch.int64) * 3
        return typ

    @staticmethod
    def _is_acyclic(A: torch.Tensor, mask: torch.Tensor) -> bool:
        """Check acyclicity of the directed graph induced by A on valid nodes.

        A is a [K,K] adjacency matrix with {0,1} entries where A[i,j]=1 indicates a directed edge i->j.
        """
        K = A.shape[0]
        valid = (mask > 0.5).to(torch.bool)
        if valid.sum().item() <= 1:
            return True
        # Kahn's algorithm
        A_bin = (A > 0.5).to(torch.int64)
        indeg = A_bin.sum(dim=0)  # [K]
        indeg = indeg * valid.to(torch.int64)
        # nodes not valid are ignored
        q = [int(i) for i in range(K) if valid[i] and indeg[i].item() == 0]
        visited = 0
        A_work = A_bin.clone()
        indeg_work = indeg.clone()
        while q:
            n = q.pop()
            visited += 1
            # remove outgoing edges
            out = A_work[n]
            if out.sum().item() == 0:
                continue
            for j in range(K):
                if not valid[j] or out[j].item() == 0:
                    continue
                indeg_work[j] -= 1
                A_work[n, j] = 0
                if indeg_work[j].item() == 0:
                    q.append(int(j))
        return visited == int(valid.sum().item())

    @staticmethod
    def _pair_distance(agent_feat: torch.Tensor) -> torch.Tensor:
        # agent_feat: [B,K,Da], assume x,y at 0,1
        pos = agent_feat[..., 0:2]
        rel = pos.unsqueeze(2) - pos.unsqueeze(1)
        return torch.norm(rel, dim=-1)  # [B,K,K]

    @staticmethod
    def _pair_closest_approach_distance(
        agent_feat: torch.Tensor,
        horizon_s: float = 3.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Approximate interaction likelihood via constant-velocity closest approach.

        agent_feat is assumed to contain (x,y, yaw, vx, vy, ...).

        Returns:
          d_min: [B,K,K] where d_min[b,i,j] is the minimal distance between i and j
          within [0, horizon_s] under constant relative velocity.
        """
        if agent_feat.shape[-1] < 5:
            # Not enough kinematics information; fall back to current distance.
            return EBSTM._pair_distance(agent_feat)

        pos = agent_feat[..., 0:2]   # [B,K,2]
        vel = agent_feat[..., 3:5]   # [B,K,2]

        # r_ij = p_j - p_i ; v_ij = v_j - v_i
        r = pos.unsqueeze(1) - pos.unsqueeze(2)  # [B,K,K,2]
        v = vel.unsqueeze(1) - vel.unsqueeze(2)  # [B,K,K,2]
        vv = (v * v).sum(dim=-1)                 # [B,K,K]
        rv = (r * v).sum(dim=-1)                 # [B,K,K]
        t_star = (-rv / (vv + eps)).clamp(min=0.0, max=float(horizon_s))  # [B,K,K]
        closest = r + v * t_star.unsqueeze(-1)
        d_min = torch.norm(closest, dim=-1)  # [B,K,K]

        # mask diagonal (self-pairs)
        K = d_min.shape[-1]
        eye = torch.eye(K, device=d_min.device, dtype=torch.bool).unsqueeze(0)
        d_min = d_min.masked_fill(eye, float("inf"))
        return d_min

    def generate_candidates(
        self,
        A_t: torch.Tensor,        # [B,K,K]
        agent_feat: torch.Tensor, # [B,K,Da]
        agent_mask: torch.Tensor, # [B,K]
        oracle_next: Optional[torch.Tensor] = None,  # [B,K,K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a candidate set of next adjacency matrices per batch item.

        This is a *templateized edit-token* generator:
          - Candidate 0: keep A_t (NOOP token)
          - Other candidates: apply exactly one local edit on a valid close pair (i,j):
              add i->j, add j->i, delete, flip direction.

        If two_step_topk_pairs>0, we additionally include *two-step* edit combinations
        via a small beam search over edit tokens restricted to top-k relevant pairs.
        """
        B, K, _ = A_t.shape
        dist = self._pair_distance(agent_feat)  # [B,K,K]
        dmin = None
        if self.use_closest_approach_candidates:
            dmin = self._pair_closest_approach_distance(agent_feat, horizon_s=float(self.conflict_horizon_s))  # [B,K,K]
        candidates_all: List[List[torch.Tensor]] = []

        for b in range(B):
            Ab = A_t[b]
            mb = agent_mask[b]
            db = dist[b]
            dmb = dmin[b] if dmin is not None else db

            # Candidate list (prioritized). Always include NOOP.
            cand: List[torch.Tensor] = [Ab.clone()]

            # Enumerate candidate unordered pairs (i<j).
            # We include pairs that are either currently close OR have a small
            # closest-approach distance within a short horizon.
            pairs: List[Tuple[int, int]] = []
            pair_scores: List[float] = []
            if K > 1:
                tri = torch.triu_indices(K, K, offset=1, device=Ab.device)
                ii = tri[0]
                jj = tri[1]
                valid_pair = (mb[ii] > 0.5) & (mb[jj] > 0.5)

                d_now = db[ii, jj]
                d_close = dmb[ii, jj]
                eligible = valid_pair & (
                    (d_now <= float(self.edit_dist_threshold))
                    | (d_close <= float(self.conflict_dist_threshold))
                )

                # Prioritize by minimum predicted distance (smaller = more relevant).
                score = torch.minimum(d_now, d_close)
                score = score.masked_fill(~eligible, float("inf"))
                order = torch.argsort(score)
                for idx in order.tolist():
                    if not torch.isfinite(score[idx]).item():
                        continue
                    pairs.append((int(ii[idx].item()), int(jj[idx].item())))
                    pair_scores.append(float(score[idx].item()))

            # ------------------------------------------------------------------
            # Helper: enumerate *possible* single-pair edits.
            # ------------------------------------------------------------------
            def _pair_state(A: torch.Tensor, i: int, j: int) -> int:
                a_ij = int(A[i, j].item() > 0.5)
                a_ji = int(A[j, i].item() > 0.5)
                if a_ij == 0 and a_ji == 0:
                    return 0
                if a_ij == 1 and a_ji == 0:
                    return 1
                if a_ij == 0 and a_ji == 1:
                    return 2
                return 3

            def _set_pair_state(A: torch.Tensor, i: int, j: int, state: int) -> torch.Tensor:
                A2 = A.clone()
                if state == 0:
                    A2[i, j] = 0
                    A2[j, i] = 0
                elif state == 1:
                    A2[i, j] = 1
                    A2[j, i] = 0
                elif state == 2:
                    A2[i, j] = 0
                    A2[j, i] = 1
                else:
                    # invalid "both" state is never produced
                    A2[i, j] = 0
                    A2[j, i] = 0
                return A2

            def _single_edits(A: torch.Tensor, i: int, j: int) -> List[torch.Tensor]:
                s = _pair_state(A, i, j)
                # Only allow transitions among the valid 3 states.
                # If the current state is invalid (both), treat it as NONE.
                if s == 3:
                    s = 0
                outA: List[torch.Tensor] = []
                for tgt in (0, 1, 2):
                    if tgt == s:
                        continue
                    A2 = _set_pair_state(A, i, j, tgt)
                    if (not self.enforce_acyclic) or self._is_acyclic(A2, mb):
                        outA.append(A2)
                return outA

            # ------------------------------------------------------------------
            # (Req-5) Two-step edit combos via beam over edits (depth=2)
            # ------------------------------------------------------------------
            if self.two_step_topk_pairs > 0 and len(pairs) > 0:
                topk = min(int(self.two_step_topk_pairs), len(pairs))
                beam_size = max(1, min(int(self.two_step_beam_size), int(self.max_candidates)))
                pairs_top = pairs[:topk]
                scores_top = pair_scores[:topk]

                # Depth-1 beam
                beam1: List[Tuple[float, torch.Tensor]] = []
                for (pi, pj), sc in zip(pairs_top, scores_top):
                    for A1 in _single_edits(Ab, pi, pj):
                        beam1.append((float(sc), A1))
                beam1.sort(key=lambda x: x[0])
                # Keep unique adjacency only
                uniq1: List[Tuple[float, torch.Tensor]] = []
                for sc, A1 in beam1:
                    if not any(torch.equal(A1, U) for _, U in uniq1):
                        uniq1.append((sc, A1))
                    if len(uniq1) >= beam_size:
                        break

                # Add depth-1 candidates early (higher priority)
                for _, A1 in uniq1:
                    cand.append(A1)
                    if len(cand) >= self.max_candidates:
                        break

                # Depth-2 beam (expand from best depth-1 states)
                if len(cand) < self.max_candidates:
                    beam2: List[Tuple[float, torch.Tensor]] = []
                    for sc1, A1 in uniq1:
                        for (pi, pj), sc2 in zip(pairs_top, scores_top):
                            for A2 in _single_edits(A1, pi, pj):
                                beam2.append((float(sc1 + sc2), A2))
                    beam2.sort(key=lambda x: x[0])
                    uniq2: List[Tuple[float, torch.Tensor]] = []
                    for sc, A2 in beam2:
                        if not any(torch.equal(A2, U) for _, U in uniq2):
                            uniq2.append((sc, A2))
                        if len(uniq2) >= beam_size:
                            break

                    for _, A2 in uniq2:
                        cand.append(A2)
                        if len(cand) >= self.max_candidates:
                            break

            # Standard single-edit candidates over all eligible pairs.
            # NOTE: if two-step beam is enabled, many of the most relevant single-edits
            # are already included. This loop adds coverage for remaining pairs.
            for (i, j) in pairs:
                if len(cand) >= self.max_candidates:
                    break
                for A1 in _single_edits(Ab, i, j):
                    cand.append(A1)
                    if len(cand) >= self.max_candidates:
                        break

            cand = _unique_adjacency(cand)

            # Ensure oracle is included and never truncated away (for supervised structure training).
            if oracle_next is not None:
                A_or = oracle_next[b].clone()
                new: List[torch.Tensor] = [Ab.clone()]
                if not torch.equal(A_or, new[0]):
                    new.append(A_or)
                for A in cand:
                    if any(torch.equal(A, U) for U in new):
                        continue
                    new.append(A)
                    if len(new) >= self.max_candidates:
                        break
                cand = new
            else:
                cand = cand[: self.max_candidates]

            candidates_all.append(cand)

        # pad to [B,C,K,K]
        C = min(max(len(c) for c in candidates_all), self.max_candidates)
        out = torch.zeros((B, C, K, K), device=A_t.device, dtype=A_t.dtype)
        out_mask = torch.zeros((B, C), device=A_t.device, dtype=torch.bool)
        for b, cand in enumerate(candidates_all):
            n = min(C, len(cand))
            for c_idx in range(n):
                out[b, c_idx] = cand[c_idx]
            out_mask[b, :n] = True
        return out, out_mask

    @staticmethod
    def energies_of_candidates(pair: EditTokenEnergies, A_t: torch.Tensor, candidate_A: torch.Tensor) -> torch.Tensor:
        """Vectorized **transition** energy over edit tokens δ(prev→next).

        Args:
          pair:        EditTokenEnergies with e_edit: [B,K,K,9]
          A_t:         [B,K,K]
          candidate_A: [B,C,K,K]
        Returns:
          energies:    [B,C]
        """
        e_edit = pair.e_edit  # [B,K,K,9]
        pm = pair.pair_mask   # [B,K,K] bool
        B, C, K, _ = candidate_A.shape

        prev = EBSTM._relation_state3(A_t)  # [B,K,K] in {0,1,2,3}
        nxt = EBSTM._relation_state3(candidate_A)  # [B,C,K,K]

        # Handle invalid "both" edges by clamping and adding a large penalty.
        invalid = (prev == 3).unsqueeze(1) | (nxt == 3)
        prev = prev.clamp_max(2)
        nxt = nxt.clamp_max(2)
        tok = (prev.unsqueeze(1) * 3 + nxt).clamp(min=0, max=8)  # [B,C,K,K]

        # NOTE: torch.gather does **not** broadcast the candidate dimension.
        # Expand explicitly so this works for any C>1.
        e = e_edit.unsqueeze(1).expand(-1, C, -1, -1, -1)  # [B,C,K,K,9]
        e_tok = torch.gather(e, dim=-1, index=tok.unsqueeze(-1)).squeeze(-1)  # [B,C,K,K]

        triu = torch.triu(torch.ones((K, K), device=candidate_A.device, dtype=torch.bool), diagonal=1)
        valid = (pm & triu).unsqueeze(1)
        E = (e_tok * valid.to(dtype=e_tok.dtype)).sum(dim=(-1, -2))  # [B,C]

        if invalid.any():
            # Penalize any invalid relation state on valid pairs.
            inv = (invalid & valid).to(torch.float32).sum(dim=(-1, -2))
            E = E + inv * 1e6

        return E.to(dtype=torch.float32)

    def _uncertainty_penalty(self, candidate_A: torch.Tensor, z_logvar: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Optional structure-level uncertainty penalty.

        Penalize interactions between agents with uncertain preferences (high variance).
        """
        if self.uncertainty_penalty_scale <= 0.0:
            return torch.zeros(candidate_A.shape[:2], device=candidate_A.device, dtype=torch.float32)

        B, C, K, _ = candidate_A.shape
        triu = torch.triu(torch.ones((K, K), device=candidate_A.device, dtype=torch.bool), diagonal=1)
        valid = (pair_mask & triu).unsqueeze(1)  # [B,1,K,K]
        a = (candidate_A > 0.5)
        aT = a.transpose(-1, -2)
        edge = (a | aT)  # [B,C,K,K] True if any directed edge exists between i and j

        var = torch.exp(z_logvar).mean(dim=-1)  # [B,K]
        sum_var = var.unsqueeze(-1) + var.unsqueeze(-2)  # [B,K,K]
        sum_var = sum_var.unsqueeze(1)  # [B,1,K,K]
        pen = edge.to(dtype=torch.float32) * sum_var.to(dtype=torch.float32)
        pen = (pen * valid.to(dtype=torch.float32)).sum(dim=(-1, -2))  # [B,C]
        return pen * float(self.uncertainty_penalty_scale)

    def forward(
        self,
        A_t: torch.Tensor,        # [B,K,K]
        agent_feat: torch.Tensor, # [B,K,Da]
        z_mean: torch.Tensor,     # [B,K,Dz]
        z_logvar: torch.Tensor,   # [B,K,Dz]
        agent_mask: torch.Tensor, # [B,K]
        oracle_next: Optional[torch.Tensor] = None,  # [B,K,K]
        # Optional structure-level penalties (paper E_smooth + E_phys).
        smooth_scale: float = 0.0,
        phys_dist_threshold_m: float = 1e9,
        phys_penalty_scale: float = 0.0,
    ) -> EBSTMOutput:
        candidate_A, cand_mask = self.generate_candidates(A_t, agent_feat, agent_mask, oracle_next=oracle_next)

        pair = self.energy_net(agent_feat, z_mean, z_logvar, agent_mask)          # per-pair edit energies
        energies = self.energies_of_candidates(pair, A_t, candidate_A)            # [B,C]

        # Smoothness penalty: discourage large edits between A_t and A_{t+Δ}.
        if float(smooth_scale) > 0.0:
            energies = energies + float(smooth_scale) * self._smooth_penalty(A_t, candidate_A)

        # Physical plausibility penalty: discourage edges between very distant agents.
        if float(phys_penalty_scale) > 0.0:
            energies = energies + float(phys_penalty_scale) * self._phys_penalty(agent_feat, candidate_A, pair.pair_mask, float(phys_dist_threshold_m))

        # Optional extra penalty at structure level
        energies = energies + self._uncertainty_penalty(candidate_A, z_logvar, pair.pair_mask)

        # mask out padded candidates
        logits = -energies / float(self.temperature)
        logits = logits.masked_fill(~cand_mask, -1e9)
        probs = torch.softmax(logits, dim=1)
        edit_tokens = self.extract_edit_tokens(A_t, candidate_A)
        return EBSTMOutput(candidate_A=candidate_A, candidate_mask=cand_mask, energies=energies, probs=probs, edit_tokens=edit_tokens, pair_energies=pair)

    @staticmethod
    def _relation_type(A: torch.Tensor) -> torch.Tensor:
        """Map a directed adjacency matrix to a 3-state per unordered pair encoding.

        For each (i,j) with i<j:
          0: none, 1: i->j, 2: j->i, 3: both (invalid)

        Returns a tensor of shape [...,K,K] where only upper-triangle entries are meaningful.
        """
        a = (A > 0.5)
        at = a.transpose(-1, -2)
        # upper triangle encoding
        typ = torch.zeros_like(A, dtype=torch.int64)
        typ = typ + (a & (~at)).to(torch.int64) * 1
        typ = typ + ((~a) & at).to(torch.int64) * 2
        typ = typ + (a & at).to(torch.int64) * 3
        return typ

    def _smooth_penalty(self, A_t: torch.Tensor, candidate_A: torch.Tensor) -> torch.Tensor:
        """Hamming distance in 3-state relation space over i<j pairs."""
        B, C, K, _ = candidate_A.shape
        triu = torch.triu(torch.ones((K, K), device=candidate_A.device, dtype=torch.bool), diagonal=1)
        t0 = self._relation_type(A_t).unsqueeze(1)          # [B,1,K,K]
        t1 = self._relation_type(candidate_A)               # [B,C,K,K]
        diff = (t1 != t0) & triu.unsqueeze(0).unsqueeze(0)
        return diff.to(torch.float32).sum(dim=(-1, -2))     # [B,C]

    @staticmethod
    def _phys_penalty(
        agent_feat: torch.Tensor,       # [B,K,Da]
        candidate_A: torch.Tensor,      # [B,C,K,K]
        pair_mask: torch.Tensor,        # [B,K,K]
        dist_threshold_m: float,
    ) -> torch.Tensor:
        """Penalize candidate edges that connect agents farther than dist_threshold_m."""
        if not torch.isfinite(torch.tensor(dist_threshold_m)):
            return torch.zeros(candidate_A.shape[:2], device=candidate_A.device, dtype=torch.float32)

        pos = agent_feat[..., 0:2]  # [B,K,2] (ego-local x,y)
        rel = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B,K,K,2]
        dist = torch.norm(rel, dim=-1)  # [B,K,K]

        B, C, K, _ = candidate_A.shape
        triu = torch.triu(torch.ones((K, K), device=candidate_A.device, dtype=torch.bool), diagonal=1)
        valid = (pair_mask & triu).unsqueeze(1)  # [B,1,K,K]

        a = (candidate_A > 0.5)
        edge_any = a | a.transpose(-1, -2)  # [B,C,K,K]
        far = F.relu(dist.unsqueeze(1) - float(dist_threshold_m))
        pen = (edge_any.to(torch.float32) * far.to(torch.float32))
        pen = (pen * valid.to(torch.float32)).sum(dim=(-1, -2))
        return pen

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

    @torch.no_grad()
    def expected_ego_interaction_risk_proxy(
        self,
        candidate_A: torch.Tensor,   # [B,C,K,K]
        probs: torch.Tensor,         # [B,C]
        agent_feat: torch.Tensor,    # [B,K,Da]
        agent_mask: torch.Tensor,    # [B,K]
        *,
        horizon_s: float = 3.0,
        sigma_dist: float = 6.0,
    ) -> torch.Tensor:
        """Cheap proxy for incremental collision-risk contribution.

        This is used for predictability-adaptive horizon truncation (paper: entropy + risk test).

        We approximate "risk contribution" as expected *ego-involved interaction mass*
        weighted by ego↔agent closest-approach distance under constant-velocity kinematics.

        The proxy is intentionally lightweight:
          1) compute ego↔agent closest-approach distance d_min within `horizon_s`
          2) convert to weight w = exp(-d_min / sigma_dist)
          3) for each candidate structure, count ego-involved edges (either direction)
          4) take expectation under `probs`

        Returns:
          risk_proxy: [B]
        """
        if candidate_A.numel() == 0:
            return torch.zeros((agent_feat.shape[0],), device=agent_feat.device, dtype=torch.float32)

        B, C, K, _ = candidate_A.shape
        if K <= 1:
            return torch.zeros((B,), device=agent_feat.device, dtype=torch.float32)

        # Closest approach distances under constant velocity.
        dmin = self._pair_closest_approach_distance(agent_feat, float(horizon_s))  # [B,K,K]
        d_ego = dmin[:, 0, :]  # [B,K]

        # Distance weights (mask invalid agents)
        w = torch.exp(-d_ego / max(1e-6, float(sigma_dist))).to(torch.float32)  # [B,K]
        w = w * agent_mask.to(torch.float32)
        w[:, 0] = 0.0

        # Ego-involved edges (either direction)
        a = (candidate_A > 0.5)
        edge_any = a | a.transpose(-1, -2)  # [B,C,K,K]
        edge_ego = edge_any[:, :, 0, :]     # [B,C,K]

        risk_c = (edge_ego.to(torch.float32) * w.unsqueeze(1)).sum(dim=-1)  # [B,C]
        risk = (risk_c * probs.to(torch.float32)).sum(dim=-1)               # [B]
        return risk

    def rollout(
        self,
        A0: torch.Tensor,
        agent_feat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        agent_mask: torch.Tensor,
        horizon_steps: int = 5,
        num_samples: int = 16,
        return_indices: bool = False,
        entropy_stop_threshold: Optional[float] = None,
        min_horizon_steps: int = 1,
        # (Req-6) Combined truncation: entropy + risk contribution proxy
        risk_stop_threshold: Optional[float] = None,
        risk_horizon_s: float = 3.0,
        risk_sigma_dist: float = 6.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample rollouts of interaction structures.

        Args:
          entropy_stop_threshold: if set, we stop evolving a batch item once the transition entropy exceeds it.
                                  Remaining steps repeat the last structure (predictability-adaptive horizon).
          risk_stop_threshold: if set, we *only* stop when BOTH
              (entropy > entropy_stop_threshold) AND (risk_proxy < risk_stop_threshold).
            This aligns with the paper's "entropy + risk contribution" truncation.
        Returns:
          A_samples: [B, S, H, K, K]
          logp:      [B, S] (log prob under the EB-STM transitions)
          idx:       [B, S, H] indices of chosen candidates (optional; padding indices are 0)
        """
        B, K, _ = A0.shape
        H = int(horizon_steps)
        S = int(num_samples)

        A_samples = torch.zeros((B, S, H, K, K), device=A0.device, dtype=A0.dtype)
        logp = torch.zeros((B, S), device=A0.device, dtype=torch.float32)
        idx_samples = torch.zeros((B, S, H), device=A0.device, dtype=torch.int64) if return_indices else None

        for s in range(S):
            A_t = A0
            lp = torch.zeros((B,), device=A0.device, dtype=torch.float32)
            active = torch.ones((B,), device=A0.device, dtype=torch.bool)

            for h in range(H):
                if not active.any():
                    # fill remaining
                    A_samples[:, s, h:] = A_t.unsqueeze(1).expand(B, H - h, K, K)
                    break

                out = self.forward(A_t, agent_feat, z_mean, z_logvar, agent_mask, oracle_next=None)

                # ------------------------------------------------------------------
                # Predictability-adaptive horizon: entropy + (optional) risk proxy.
                # ------------------------------------------------------------------
                stop = torch.zeros((B,), device=A0.device, dtype=torch.bool)
                if entropy_stop_threshold is not None and h + 1 >= int(min_horizon_steps):
                    p = out.probs.clamp_min(1e-12)
                    ent = -(p * torch.log(p)).sum(dim=-1)  # [B]
                    ent_high = ent > float(entropy_stop_threshold)
                    if risk_stop_threshold is None:
                        stop = ent_high
                    else:
                        risk = self.expected_ego_interaction_risk_proxy(
                            out.candidate_A,
                            out.probs,
                            agent_feat,
                            agent_mask,
                            horizon_s=float(risk_horizon_s),
                            sigma_dist=float(risk_sigma_dist),
                        )
                        stop = ent_high & (risk < float(risk_stop_threshold))

                # sample next for active & not stopped
                cat = torch.distributions.Categorical(probs=out.probs)
                idx = cat.sample()  # [B]
                A_next = out.candidate_A[torch.arange(B, device=A0.device), idx]

                # apply stopping: if stop, keep A_t and no additional log prob
                A_next = torch.where(stop.view(B, 1, 1), A_t, A_next)
                p_sel = out.probs[torch.arange(B, device=A0.device), idx].clamp_min(1e-12)
                lp_step = torch.where(stop, torch.zeros_like(lp), torch.log(p_sel))
                lp = lp + lp_step

                A_samples[:, s, h] = A_next
                if return_indices:
                    idx_samples[:, s, h] = idx

                A_t = A_next
                active = active & (~stop)

            logp[:, s] = lp

        return A_samples, logp, idx_samples


    @torch.no_grad()
    def rollout_log_prob(
        self,
        A0: torch.Tensor,
        rollouts: torch.Tensor,  # [B,S,H,K,K]
        agent_feat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        agent_mask: torch.Tensor,
        *,
        smooth_scale: float = 0.0,
        phys_dist_threshold_m: float = 1e9,
        phys_penalty_scale: float = 0.0,
        missing_logp: float = -30.0,
    ) -> torch.Tensor:
        """Compute log-probabilities of provided rollouts under the EB-STM transitions.

        This is useful for importance sampling / mixed-support PCI.
        If a step's next structure is not in the candidate set, we assign `missing_logp`.

        Args:
          A0:       [B,K,K] initial adjacency
          rollouts: [B,S,H,K,K] adjacency sequences where rollouts[:,:,h] is A_{t+h+1}
        Returns:
          logp:     [B,S]
        """
        if rollouts.numel() == 0:
            B = A0.shape[0]
            S = rollouts.shape[1]
            return torch.zeros((B, S), device=A0.device, dtype=torch.float32)

        B, S, H, K, _ = rollouts.shape
        logp = torch.zeros((B, S), device=A0.device, dtype=torch.float32)

        A_prev = A0.unsqueeze(1).expand(B, S, K, K)
        for h in range(H):
            A_next = rollouts[:, :, h]

            # Flatten BS for a single EB-STM forward.
            A_prev_f = A_prev.reshape(B * S, K, K)
            A_next_f = A_next.reshape(B * S, K, K)

            feat_f = agent_feat.unsqueeze(1).expand(B, S, K, agent_feat.shape[-1]).reshape(B * S, K, -1)
            zm_f = z_mean.unsqueeze(1).expand(B, S, K, z_mean.shape[-1]).reshape(B * S, K, -1)
            zv_f = z_logvar.unsqueeze(1).expand(B, S, K, z_logvar.shape[-1]).reshape(B * S, K, -1)
            mask_f = agent_mask.unsqueeze(1).expand(B, S, K).reshape(B * S, K)

            out = self.forward(
                A_prev_f,
                feat_f,
                zm_f,
                zv_f,
                mask_f,
                oracle_next=None,
                smooth_scale=float(smooth_scale),
                phys_dist_threshold_m=float(phys_dist_threshold_m),
                phys_penalty_scale=float(phys_penalty_scale),
            )

            cand = out.candidate_A  # [BS,C,K,K]
            eq = (cand == A_next_f.unsqueeze(1)).all(dim=-1).all(dim=-1) & out.candidate_mask
            has = eq.any(dim=-1)
            idx_match = torch.argmax(eq.to(torch.int64), dim=-1)

            p_sel = out.probs[torch.arange(B * S, device=A0.device), idx_match].clamp_min(1e-12)
            lp_step = torch.where(has, torch.log(p_sel), torch.full_like(p_sel, float(missing_logp)))
            logp = logp + lp_step.view(B, S)

            A_prev = A_next

        return logp

    @torch.no_grad()
    def beam_rollout(
        self,
        A0: torch.Tensor,
        agent_feat: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        agent_mask: torch.Tensor,
        *,
        max_horizon_steps: int = 5,
        beam_size: int = 8,
        entropy_stop_threshold: Optional[float] = None,
        min_horizon_steps: int = 1,
        # (Req-6) Combined truncation: entropy + risk contribution proxy
        risk_stop_threshold: Optional[float] = None,
        risk_horizon_s: float = 3.0,
        risk_sigma_dist: float = 6.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam rollout over structures (top-K sequences).

        Args:
          entropy_stop_threshold: if set, stop expanding a batch item once transition entropy exceeds it.
          risk_stop_threshold: if set, we *only* stop when BOTH
              (entropy > entropy_stop_threshold) AND (risk_proxy < risk_stop_threshold)
        Returns:
          A_beams:  [B, beam, H, K, K] (H == max_horizon_steps; if stopped early, last structure is repeated)
          logp:     [B, beam]
        """
        B, K, _ = A0.shape
        H = int(max_horizon_steps)
        beam = int(max(1, beam_size))

        # per batch, maintain beams
        A_beams = torch.zeros((B, beam, H, K, K), device=A0.device, dtype=A0.dtype)
        logp_beams = torch.full((B, beam), float("-inf"), device=A0.device, dtype=torch.float32)

        # Initialize beams with A0
        curr_A = A0.unsqueeze(1).expand(B, beam, K, K).clone()
        logp_beams[:, 0] = 0.0  # first beam active

        active = torch.ones((B,), device=A0.device, dtype=torch.bool)
        last_h = -1

        for h in range(H):
            last_h = h
            if not active.any():
                break

            # Flatten active beams
            A_flat = curr_A.reshape(B * beam, K, K)
            mask_flat = agent_mask.unsqueeze(1).expand(B, beam, -1).reshape(B * beam, K)
            feat_flat = agent_feat.unsqueeze(1).expand(B, beam, -1, -1).reshape(B * beam, K, -1)
            z_mean_flat = z_mean.unsqueeze(1).expand(B, beam, -1, -1).reshape(B * beam, K, -1)
            z_logvar_flat = z_logvar.unsqueeze(1).expand(B, beam, -1, -1).reshape(B * beam, K, -1)

            out = self.forward(A_flat, feat_flat, z_mean_flat, z_logvar_flat, mask_flat, oracle_next=None)
            # out.probs: [B*beam, C]
            C = out.probs.shape[1]
            # entropy for each flattened beam
            if entropy_stop_threshold is not None:
                ent = -(out.probs.clamp_min(1e-12) * torch.log(out.probs.clamp_min(1e-12))).sum(dim=-1)  # [B*beam]
                ent = ent.view(B, beam)
                # stop criterion based on best beam (beam 0) entropy
                stop_b = ent[:, 0] > float(entropy_stop_threshold)
                if h + 1 >= int(min_horizon_steps):
                    if risk_stop_threshold is None:
                        active = active & (~stop_b)
                    else:
                        risk_flat = self.expected_ego_interaction_risk_proxy(
                            out.candidate_A,
                            out.probs,
                            feat_flat,
                            mask_flat,
                            horizon_s=float(risk_horizon_s),
                            sigma_dist=float(risk_sigma_dist),
                        )  # [B*beam]
                        risk = risk_flat.view(B, beam)[:, 0]
                        active = active & (~(stop_b & (risk < float(risk_stop_threshold))))

            # Expand beams: for each existing beam, take top candidates
            topk = min(beam, C)
            top_p, top_idx = torch.topk(out.probs, k=topk, dim=1)  # [B*beam, topk]
            top_logp = torch.log(top_p.clamp_min(1e-12))            # [B*beam, topk]

            # Gather next A
            cand = out.candidate_A  # [B*beam, C, K, K]
            next_A = cand[torch.arange(B * beam, device=A0.device).unsqueeze(-1), top_idx]  # [B*beam, topk, K,K]

            # Combine scores
            base_lp = logp_beams.view(B * beam, 1)  # [B*beam,1]
            new_lp = base_lp + top_logp             # [B*beam,topk]
            new_lp = new_lp.view(B, beam * topk)    # [B, beam*topk]
            next_A = next_A.view(B, beam * topk, K, K)

            # Mask inactive batches: keep previous beams (no update)
            if active.any():
                # Select top beams per batch
                sel_lp, sel_idx = torch.topk(new_lp, k=beam, dim=1)
                sel_A = next_A[torch.arange(B, device=A0.device).unsqueeze(-1), sel_idx]  # [B,beam,K,K]

                # Update only active batches
                curr_A = torch.where(active.view(B, 1, 1, 1), sel_A, curr_A)
                logp_beams = torch.where(active.view(B, 1), sel_lp, logp_beams)

            # Write to output sequences
            A_beams[:, :, h] = curr_A

        # Fill any remaining horizon steps if we stopped early (repeat the last structure).
        if last_h < H - 1 and last_h >= 0:
            A_beams[:, :, last_h + 1 :] = curr_A.unsqueeze(2).expand(B, beam, H - last_h - 1, K, K)

        return A_beams, logp_beams
