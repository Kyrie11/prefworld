from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-9) -> torch.Tensor:
    """Softmax with a boolean mask (True=keep).

    Bug-fix: when *all* entries are masked out along `dim`, plain softmax would return a uniform
    distribution (or NaNs depending on the implementation). We instead return all-zeros for that slice.
    """
    m = mask.to(dtype=torch.bool)
    logits = logits.masked_fill(~m, -1e9)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * m.to(dtype=probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs / denom


@dataclass
class TemplateEncoding:
    """Per-agent template encoding τ."""

    tau: torch.Tensor          # [B,N,T,D]
    map_context: torch.Tensor  # [B,N,T,Dm]
    nbr_context: torch.Tensor  # [B,N,T,Da]

    # --- Structured template components (paper-aligned) ---
    # These are optional and may be ignored by downstream modules.
    neighbor_mask: Optional[torch.Tensor] = None        # [B,N,T,N] bool
    feasible_actions: Optional[torch.Tensor] = None     # [B,N,T,M] bool
    kinematic_limits: Optional[torch.Tensor] = None     # [B,N,T,3] (max_acc, max_speed, max_curv)
    conflict_mask: Optional[torch.Tensor] = None        # [B,N,T,N] bool
    conflict_region_id: Optional[torch.Tensor] = None   # [B,N,T,N] int
    map_attn: Optional[torch.Tensor] = None             # [B,N,T,M_map] float


class MapPolylineNodeEncoder(nn.Module):
    """Encode each map polyline into a node embedding."""

    def __init__(
        self,
        point_dim: int = 2,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_poly_types: int = 4,
        n_tl_status: int = 8,
        n_on_route: int = 2,
    ) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.type_emb = nn.Embedding(n_poly_types, hidden_dim)
        self.tl_emb = nn.Embedding(n_tl_status, hidden_dim)
        self.route_emb = nn.Embedding(n_on_route, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        map_polylines: torch.Tensor,  # [B,M,L,2]
        map_poly_mask: torch.Tensor,  # [B,M] (float or bool)
        map_poly_type: Optional[torch.Tensor] = None,  # [B,M]
        map_tl_status: Optional[torch.Tensor] = None,  # [B,M]
        map_on_route: Optional[torch.Tensor] = None,  # [B,M]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, L, _ = map_polylines.shape
        m = map_poly_mask > 0.5

        h = self.point_mlp(map_polylines)  # [B,M,L,H]
        h = h.max(dim=2).values            # [B,M,H]

        if map_poly_type is None:
            map_poly_type = torch.zeros((B, M), device=map_polylines.device, dtype=torch.long)
        if map_tl_status is None:
            map_tl_status = torch.zeros((B, M), device=map_polylines.device, dtype=torch.long)
        if map_on_route is None:
            map_on_route = torch.zeros((B, M), device=map_polylines.device, dtype=torch.long)

        # clamp to embedding ranges
        map_poly_type = map_poly_type.long().clamp(min=0, max=self.type_emb.num_embeddings - 1)
        map_tl_status = map_tl_status.long().clamp(min=0, max=self.tl_emb.num_embeddings - 1)
        map_on_route = map_on_route.long().clamp(min=0, max=self.route_emb.num_embeddings - 1)

        h = h + self.type_emb(map_poly_type) + self.tl_emb(map_tl_status) + self.route_emb(map_on_route)
        h = self.out(h)  # [B,M,Dm]
        h = h * m.unsqueeze(-1).to(dtype=h.dtype)
        return h, m


class AgentStateEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemplateEncoder(nn.Module):
    """Build a per-agent, per-timestep template encoding τ from (HD map polylines + local neighbor graph).

    This is a pragmatic implementation aligned with the paper's notion of
    τ_{i,t} = (N_{i,t}, G_{i,t}, A_{i,t}, K_{i,t}) by explicitly using:
      - N/G: a distance-based neighbor graph with typed relative-geometry features;
      - map context: attention from agent state into nearby lane/lane-connector polylines;
    and producing a learned embedding h_{τ_{i,t}}.
    """

    def __init__(
        self,
        tau_dim: int = 64,
        agent_state_dim: int = 5,  # x,y,yaw,vx,vy in ego frame
        hidden_dim: int = 128,
        map_node_dim: int = 128,
        nbr_dim: int = 128,
        neighbor_radius_m: float = 30.0,
    ) -> None:
        super().__init__()
        self.neighbor_radius_m = float(neighbor_radius_m)

        self.map_node_enc = MapPolylineNodeEncoder(hidden_dim=hidden_dim, out_dim=map_node_dim)
        self.agent_enc = AgentStateEncoder(in_dim=agent_state_dim, hidden_dim=hidden_dim, out_dim=nbr_dim)

        # relative geometry -> scalar compatibility
        self.map_rel_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.nbr_rel_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),  # rel_pos(2), rel_vel(2), dist(1)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.tau_out = nn.Sequential(
            nn.Linear(nbr_dim + map_node_dim + nbr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tau_dim),
        )

    def forward(
        self,
        *,
        agents_state: torch.Tensor,       # [B,N,T,5]
        agents_mask: torch.Tensor,        # [B,N,T] (float/bool)
        map_polylines: torch.Tensor,      # [B,M,L,2]
        map_poly_mask: torch.Tensor,      # [B,M]
        map_poly_type: Optional[torch.Tensor] = None,  # [B,M]
        map_tl_status: Optional[torch.Tensor] = None,  # [B,M]
        map_on_route: Optional[torch.Tensor] = None,   # [B,M]
    ) -> TemplateEncoding:
        B, N, T, Ds = agents_state.shape
        assert Ds >= 5, "agents_state expected at least (x,y,yaw,vx,vy)"

        a_mask = agents_mask > 0.5
        a_feat = agents_state[..., :5]

        # Encode map nodes
        map_node, map_mask = self.map_node_enc(map_polylines, map_poly_mask, map_poly_type, map_tl_status, map_on_route)
        # Polyline "anchor" for relative geometry (center of polyline points)
        map_center = map_polylines.mean(dim=2)  # [B,M,2]

        # Encode agent state
        a_emb = self.agent_enc(a_feat)  # [B,N,T,D]
        a_emb = a_emb * a_mask.unsqueeze(-1).to(dtype=a_emb.dtype)

        # --------------------------
        # Agent -> map attention
        # --------------------------
        agent_pos = a_feat[..., 0:2]  # [B,N,T,2]
        rel = map_center.unsqueeze(1).unsqueeze(1) - agent_pos.unsqueeze(3)  # [B,N,T,M,2]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [B,N,T,M,1]
        rel_feat = torch.cat([rel, dist], dim=-1)     # [B,N,T,M,3]
        rel_score = self.map_rel_mlp(rel_feat).squeeze(-1)  # [B,N,T,M]

        # dot-product compatibility
        q = a_emb  # [B,N,T,D]
        k = map_node  # [B,M,Dm]
        dot = torch.einsum("bntd,bmd->bntm", q, k) / math.sqrt(k.shape[-1])
        score = dot + rel_score
        score_mask = map_mask.unsqueeze(1).unsqueeze(1).expand(B, N, T, -1)  # [B,N,T,M]
        attn = _masked_softmax(score, score_mask, dim=-1)
        map_ctx = torch.einsum("bntm,bmd->bntd", attn, map_node)

        # --------------------------
        # Neighbor aggregation (agent graph)
        # --------------------------
        # Pairwise relative geometry
        pos = a_feat[..., 0:2]  # [B,N,T,2]
        vel = a_feat[..., 3:5]  # [B,N,T,2]
        # NOTE (bug-fix): keep the (i, j) convention consistent across the encoder.
        # We use rel_pos = p_j - p_i and rel_vel = v_j - v_i.
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # [B,N,N,T,2] = p_j - p_i
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)  # [B,N,N,T,2] = v_j - v_i
        rel_pos = rel_pos.permute(0, 1, 3, 2, 4)  # [B,N,T,N,2] where last N is neighbor j
        rel_vel = rel_vel.permute(0, 1, 3, 2, 4)  # [B,N,T,N,2]
        dist_ij = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B,N,T,N,1]
        rel_ij = torch.cat([rel_pos, rel_vel, dist_ij], dim=-1)  # [B,N,T,N,5]
        nbr_score = self.nbr_rel_mlp(rel_ij).squeeze(-1)  # [B,N,T,N]

        # Mask: valid i & valid j & within radius & j != i
        valid_i = a_mask
        valid_j = a_mask.unsqueeze(1).expand(B, N, N, T).permute(0, 1, 3, 2)  # [B,N,T,N]
        valid_i_exp = valid_i.unsqueeze(-1).expand(B, N, T, N)
        radius_ok = (dist_ij.squeeze(-1) <= self.neighbor_radius_m)
        eye = torch.eye(N, device=agents_state.device, dtype=torch.bool).unsqueeze(0).unsqueeze(2)  # [1,N,1,N]
        not_self = ~eye.expand(B, N, T, N)
        nbr_mask = valid_i_exp & valid_j & radius_ok & not_self

        # Attention weights over neighbors j
        # use dot-product between i-query and j-key (from embeddings)
        a_emb_j = a_emb.permute(0, 2, 1, 3)  # [B,T,N,D]
        a_emb_j = a_emb_j.unsqueeze(1).expand(B, N, T, N, -1)  # [B,N,T,N,D]
        a_emb_i = a_emb.unsqueeze(3)  # [B,N,T,1,D]
        dot_n = (a_emb_i * a_emb_j).sum(dim=-1) / math.sqrt(a_emb.shape[-1])  # [B,N,T,N]
        score_n = dot_n + nbr_score
        w_n = _masked_softmax(score_n, nbr_mask, dim=-1)
        nbr_ctx = (w_n.unsqueeze(-1) * a_emb_j).sum(dim=3)  # [B,N,T,D]

        # final τ embedding
        tau = self.tau_out(torch.cat([a_emb, map_ctx, nbr_ctx], dim=-1))
        tau = tau * a_mask.unsqueeze(-1).to(dtype=tau.dtype)

        # ------------------------------------------------------------------
        # Structured template outputs (neighbors, feasible actions, constraints)
        # ------------------------------------------------------------------
        # Conflict region proxy: constant-velocity closest-approach within horizon.
        horizon_s = 3.0
        eps = 1e-6
        pos = a_feat[..., 0:2]
        vel = a_feat[..., 3:5]
        # r = p_j - p_i, v = v_j - v_i
        r = (pos.unsqueeze(1) - pos.unsqueeze(2)).permute(0, 1, 3, 2, 4)  # [B,N,T,N,2]
        v = (vel.unsqueeze(1) - vel.unsqueeze(2)).permute(0, 1, 3, 2, 4)  # [B,N,T,N,2]
        rv = (r * v).sum(dim=-1)                  # [B,N,T,N]
        vv = (v * v).sum(dim=-1).clamp_min(eps)   # [B,N,T,N]
        t_star = (-rv / vv).clamp(min=0.0, max=horizon_s)
        closest = r + v * t_star.unsqueeze(-1)
        d2 = (closest * closest).sum(dim=-1)      # [B,N,T,N]
        # threshold: radius-based (rough proxy). Since we don't have per-agent radii here, use constant.
        d0 = 4.0
        conflict = d2 < (d0 * d0)

        # Mask invalid pairs
        valid_j = a_mask.unsqueeze(1).expand(B, N, N, T).permute(0, 1, 3, 2)
        valid_i = a_mask.unsqueeze(-1).expand(B, N, T, N)
        eye = torch.eye(N, device=agents_state.device, dtype=torch.bool).unsqueeze(0).unsqueeze(2)
        not_self = ~eye.expand(B, N, T, N)
        conflict_mask = conflict & valid_i & valid_j & not_self

        # ------------------------------------------------------------------
        # Conflict region id: map-aware proxy (improves interaction region representation)
        # ------------------------------------------------------------------
        # We use the most-attended lane connector polyline as a stable "interaction region" id.
        # This augments the purely geometric closest-approach test.
        #   0: no conflict
        #   1: geometric conflict (no shared region)
        #   >=2: shared lane-connector id (polyline index + 2)
        if map_poly_type is None:
            map_poly_type = torch.zeros((B, map_polylines.shape[1]), device=agents_state.device, dtype=torch.long)

        top_poly = attn.argmax(dim=-1).to(torch.long)  # [B,N,T]
        poly_type_top = map_poly_type.gather(1, top_poly.reshape(B, -1)).reshape(B, N, T)  # [B,N,T]
        region_poly = torch.where(poly_type_top == 1, top_poly + 2, torch.zeros_like(top_poly))  # [B,N,T]

        region_i = region_poly.unsqueeze(-1)                     # [B,N,T,1]
        region_j = region_poly.permute(0, 2, 1).unsqueeze(1)     # [B,1,T,N]
        share_region = (region_i == region_j) & (region_i > 0) & valid_i & valid_j & not_self

        # Treat shared connector as a (soft) conflict cue
        conflict_mask = conflict_mask | share_region

        conflict_region_id = torch.zeros((B, N, T, N), device=agents_state.device, dtype=torch.int64)
        conflict_region_id = torch.where(share_region, region_i.expand_as(conflict_region_id), conflict_region_id)
        conflict_region_id = torch.where(conflict_mask & (~share_region), torch.ones_like(conflict_region_id), conflict_region_id)

        # Feasible actions proxy using map polyline types and nearby lane offsets.
        M_actions = 6  # keep, LCL, LCR, TL, TR, stop (matches labels.NUM_MANEUVERS)
        feasible = torch.ones((B, N, T, M_actions), device=agents_state.device, dtype=torch.bool)
        if map_poly_type is None:
            map_poly_type = torch.zeros((B, map_polylines.shape[1]), device=agents_state.device, dtype=torch.long)
        poly_type = map_poly_type.long()
        is_lane = poly_type == 0
        is_conn = poly_type == 1
        # relative position of lane centers in agent heading frame
        yaw = a_feat[..., 2]
        relc = map_center.unsqueeze(1).unsqueeze(1) - agent_pos.unsqueeze(3)  # [B,N,T,M,2]
        rel_s, rel_d = self._rotate_to_heading(relc[..., 0], relc[..., 1], yaw.unsqueeze(-1))
        # lanes roughly ahead and parallel proxy: |rel_d| indicates left/right adjacent lanes
        lane_ok = is_lane.unsqueeze(1).unsqueeze(1) & score_mask
        left_ok = lane_ok & (rel_s > 0.0) & (rel_d > 1.5) & (rel_d < 8.0)
        right_ok = lane_ok & (rel_s > 0.0) & (rel_d < -1.5) & (rel_d > -8.0)
        feasible[..., 1] = left_ok.any(dim=-1)
        feasible[..., 2] = right_ok.any(dim=-1)

        # turns: presence of lane connector ahead.
        conn_ok = is_conn.unsqueeze(1).unsqueeze(1) & score_mask & (rel_s > 0.0) & (rel_s < 25.0) & (rel_d.abs() < 15.0)
        feasible[..., 3] = conn_ok.any(dim=-1)  # left turn (coarse)
        feasible[..., 4] = conn_ok.any(dim=-1)  # right turn (coarse)

        # Kinematic limits (simple constants; could be made speed-dependent)
        max_acc = torch.full((B, N, T, 1), 3.0, device=agents_state.device, dtype=tau.dtype)
        speed = torch.norm(a_feat[..., 3:5], dim=-1, keepdim=True)
        max_speed = torch.clamp(speed + 10.0, min=5.0, max=30.0)
        max_curv = torch.full((B, N, T, 1), 0.2, device=agents_state.device, dtype=tau.dtype)
        kin_limits = torch.cat([max_acc, max_speed, max_curv], dim=-1) * a_mask.unsqueeze(-1).to(tau.dtype)

        return TemplateEncoding(
            tau=tau,
            map_context=map_ctx,
            nbr_context=nbr_ctx,
            neighbor_mask=nbr_mask,
            feasible_actions=feasible,
            kinematic_limits=kin_limits,
            conflict_mask=conflict_mask,
            conflict_region_id=conflict_region_id,
            map_attn=attn,
        )

    @staticmethod
    def _rotate_to_heading(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate (x,y) in ego frame into heading frame (forward,left).

        yaw can be broadcastable to x/y.
        """
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        s = cy * x + sy * y
        d = -sy * x + cy * y
        return s, d
