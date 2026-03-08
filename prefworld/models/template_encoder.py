from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import (
    LonConstraint,
    NUM_LON_CONSTRAINTS,
    NUM_PATH_TYPES,
    PathType,
    path_constraint_to_maneuver,
)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-9) -> torch.Tensor:
    """Softmax with a boolean mask (True=keep)."""
    m = mask.to(dtype=torch.bool)
    logits = logits.masked_fill(~m, -1e9)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * m.to(dtype=probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs / denom


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


@dataclass
class TemplateEncoding:
    """Per-agent deterministic template encoding τ^{det}."""

    tau: torch.Tensor                     # [B,N,T,D]
    map_context: torch.Tensor             # [B,N,T,Dm]
    nbr_context: torch.Tensor             # [B,N,T,Da]

    neighbor_mask: Optional[torch.Tensor] = None        # [B,N,T,N] bool
    feasible_actions: Optional[torch.Tensor] = None     # [B,N,T,A] bool
    action_features: Optional[torch.Tensor] = None      # [B,N,T,A,Da]
    action_family: Optional[torch.Tensor] = None        # [B,N,T,A] int64 (legacy maneuver family)
    action_path_type: Optional[torch.Tensor] = None     # [B,N,T,A] int64
    action_constraint_type: Optional[torch.Tensor] = None  # [B,N,T,A] int64
    action_source_index: Optional[torch.Tensor] = None  # [B,N,T,A] int64, -1 for none
    comparable_metrics: Optional[torch.Tensor] = None   # [B,N,T,A,C]
    dynamic_metrics: Optional[torch.Tensor] = None      # [B,N,T,A,D]
    path_valid: Optional[torch.Tensor] = None           # [B,N,T,P] bool
    path_features: Optional[torch.Tensor] = None        # [B,N,T,P,Dp]
    path_polyline_idx: Optional[torch.Tensor] = None    # [B,N,T,P] int64
    topological_edge_type: Optional[torch.Tensor] = None  # [B,N,T,N] int64 (0 none / 1 follow / 2 conflict / 3 merge)
    kinematic_limits: Optional[torch.Tensor] = None     # [B,N,T,3] (max_acc, max_speed, max_curv)
    conflict_mask: Optional[torch.Tensor] = None        # [B,N,T,N] bool
    conflict_region_id: Optional[torch.Tensor] = None   # [B,N,T,N] int64
    map_attn: Optional[torch.Tensor] = None             # [B,N,T,M] float


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
        map_poly_mask: torch.Tensor,  # [B,M]
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

        map_poly_type = map_poly_type.long().clamp(min=0, max=self.type_emb.num_embeddings - 1)
        map_tl_status = map_tl_status.long().clamp(min=0, max=self.tl_emb.num_embeddings - 1)
        map_on_route = map_on_route.long().clamp(min=0, max=self.route_emb.num_embeddings - 1)

        h = h + self.type_emb(map_poly_type) + self.tl_emb(map_tl_status) + self.route_emb(map_on_route)
        h = self.out(h)
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
    """Build a deterministic template aligned with the current paper method.

    The extractor exposes a fixed-size tensorized approximation of

      τ^{det}_{i,t} = (N_{i,t}, G^{topo}_{i,t}, A_{i,t}, K^{cmp}_{i,t}, K^{dyn}_{i,t})

    using the map polylines already cached in the repo.  The core change compared to the
    previous implementation is that the discrete action set is no longer a flat maneuver id.
    Instead we enumerate a small set of reference-path branches and combine each branch with
    longitudinal constraint sources (free-flow / follow(j) / stop-line / yield-to(j)).
    """

    def __init__(
        self,
        tau_dim: int = 64,
        agent_state_dim: int = 5,
        hidden_dim: int = 128,
        map_node_dim: int = 128,
        nbr_dim: int = 128,
        neighbor_radius_m: float = 60.0,
        topo_horizon_m: float = 80.0,
        conflict_time_s: float = 5.0,
        lane_width_m: float = 3.6,
        max_follow_targets: int = 2,
        max_yield_targets: int = 2,
    ) -> None:
        super().__init__()
        self.neighbor_radius_m = float(neighbor_radius_m)
        self.topo_horizon_m = float(topo_horizon_m)
        self.conflict_time_s = float(conflict_time_s)
        self.lane_width_m = float(lane_width_m)
        self.max_paths = int(NUM_PATH_TYPES)
        self.max_follow_targets = int(max_follow_targets)
        self.max_yield_targets = int(max_yield_targets)

        self.map_node_enc = MapPolylineNodeEncoder(hidden_dim=hidden_dim, out_dim=map_node_dim)
        self.agent_enc = AgentStateEncoder(in_dim=agent_state_dim, hidden_dim=hidden_dim, out_dim=nbr_dim)

        self.map_rel_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.nbr_rel_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),  # rel_pos(2), rel_vel(2), dist(1), heading_delta(1)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.tau_out = nn.Sequential(
            nn.Linear(nbr_dim + map_node_dim + nbr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tau_dim),
        )

        self.comparable_metric_dim = 6
        self.dynamic_metric_dim = 6
        self.path_feature_dim = int(map_node_dim + 8)
        self.action_feature_dim = int(map_node_dim + NUM_PATH_TYPES + NUM_LON_CONSTRAINTS + self.comparable_metric_dim + self.dynamic_metric_dim)

        slot_path_types = []
        slot_constraint_types = []
        slot_source_ranks = []
        slot_has_source = []
        for p in range(self.max_paths):
            slot_path_types.append(p)
            slot_constraint_types.append(int(LonConstraint.FREE_FLOW))
            slot_source_ranks.append(-1)
            slot_has_source.append(0)
            for k in range(self.max_follow_targets):
                slot_path_types.append(p)
                slot_constraint_types.append(int(LonConstraint.FOLLOW))
                slot_source_ranks.append(k)
                slot_has_source.append(1)
            slot_path_types.append(p)
            slot_constraint_types.append(int(LonConstraint.STOP_LINE))
            slot_source_ranks.append(-1)
            slot_has_source.append(0)
            for k in range(self.max_yield_targets):
                slot_path_types.append(p)
                slot_constraint_types.append(int(LonConstraint.YIELD_TO))
                slot_source_ranks.append(k)
                slot_has_source.append(1)
        self.slot_path_types = tuple(slot_path_types)
        self.slot_constraint_types = tuple(slot_constraint_types)
        self.slot_source_ranks = tuple(slot_source_ranks)
        self.slot_has_source = tuple(slot_has_source)
        self.num_action_slots = len(self.slot_path_types)

    @staticmethod
    def _rotate_to_heading(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        s = cy * x + sy * y
        d = -sy * x + cy * y
        return s, d

    @staticmethod
    def _batch_gather_map(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather a map tensor ``x[B,M,...]`` with indices ``idx[B,*]``."""
        batch_idx = torch.arange(x.shape[0], device=idx.device).view(x.shape[0], *([1] * (idx.dim() - 1))).expand_as(idx)
        return x[batch_idx, idx]


    @staticmethod
    def _gather_last_dim(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather along the last dimension with an index tensor of matching prefix shape."""
        return torch.gather(x, dim=-1, index=idx)

    @staticmethod
    def _best_index(mask: torch.Tensor, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return argmin over valid entries; invalid rows get index 0 and valid=False."""
        huge = torch.finfo(score.dtype).max / 8.0
        score_m = score.masked_fill(~mask, huge)
        idx = score_m.argmin(dim=-1)
        valid = mask.any(dim=-1)
        idx = torch.where(valid, idx, torch.zeros_like(idx))
        return idx, valid

    @staticmethod
    def _masked_topk_smallest(score: torch.Tensor, mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k smallest valid scores over the last dimension."""
        if k <= 0:
            shape = score.shape[:-1] + (0,)
            return (
                torch.zeros(shape, device=score.device, dtype=torch.long),
                torch.zeros(shape, device=score.device, dtype=torch.bool),
            )
        N = score.shape[-1]
        k_eff = min(int(k), int(N))
        huge = torch.finfo(score.dtype).max / 8.0
        s = score.masked_fill(~mask, huge)
        vals, idx = torch.topk(-s, k=k_eff, dim=-1)
        vals = -vals
        valid = vals < huge / 2.0
        if k_eff < k:
            pad_shape = idx.shape[:-1] + (k - k_eff,)
            idx = torch.cat([idx, torch.zeros(pad_shape, device=idx.device, dtype=idx.dtype)], dim=-1)
            valid = torch.cat([valid, torch.zeros(pad_shape, device=valid.device, dtype=valid.dtype)], dim=-1)
        return idx, valid

    def forward(
        self,
        *,
        agents_state: torch.Tensor,       # [B,N,T,5]
        agents_mask: torch.Tensor,        # [B,N,T]
        map_polylines: torch.Tensor,      # [B,M,L,2]
        map_poly_mask: torch.Tensor,      # [B,M]
        map_poly_type: Optional[torch.Tensor] = None,
        map_tl_status: Optional[torch.Tensor] = None,
        map_on_route: Optional[torch.Tensor] = None,
    ) -> TemplateEncoding:
        B, N, T, Ds = agents_state.shape
        if Ds < 5:
            raise ValueError("agents_state expected at least (x,y,yaw,vx,vy)")

        device = agents_state.device
        dtype = agents_state.dtype
        a_mask = agents_mask > 0.5
        a_feat = agents_state[..., :5]
        agent_pos = a_feat[..., 0:2]
        yaw = a_feat[..., 2]
        vel = a_feat[..., 3:5]

        # ------------------------------------------------------------------
        # Map polyline encoding + geometry
        # ------------------------------------------------------------------
        map_node, map_mask = self.map_node_enc(map_polylines, map_poly_mask, map_poly_type, map_tl_status, map_on_route)
        map_center = map_polylines.mean(dim=2)
        map_start = map_polylines[:, :, 0, :]
        map_end = map_polylines[:, :, -1, :]
        map_heading = torch.atan2(map_end[..., 1] - map_start[..., 1], map_end[..., 0] - map_start[..., 0])
        seg = map_polylines[:, :, 1:, :] - map_polylines[:, :, :-1, :]
        seg_len = torch.norm(seg, dim=-1)
        map_length = seg_len.sum(dim=-1).clamp_min(1e-3)
        seg_yaw = torch.atan2(seg[..., 1], seg[..., 0])
        if seg_yaw.shape[2] > 1:
            dseg = _wrap_angle(seg_yaw[:, :, 1:] - seg_yaw[:, :, :-1])
            map_curv = dseg.abs().sum(dim=-1) / map_length
        else:
            map_curv = torch.zeros_like(map_length)

        if map_poly_type is None:
            map_poly_type = torch.zeros((B, map_polylines.shape[1]), device=device, dtype=torch.long)
        if map_tl_status is None:
            map_tl_status = torch.zeros((B, map_polylines.shape[1]), device=device, dtype=torch.long)
        if map_on_route is None:
            map_on_route = torch.zeros((B, map_polylines.shape[1]), device=device, dtype=torch.long)
        map_poly_type = map_poly_type.long()
        map_tl_status = map_tl_status.long()
        map_on_route = map_on_route.long()
        is_lane = map_poly_type == 0
        is_conn = map_poly_type == 1
        has_stop_control = map_tl_status >= 2  # yellow/red treated as a stop-controlled connector proxy

        # ------------------------------------------------------------------
        # Agent embedding and agent->map attention
        # ------------------------------------------------------------------
        a_emb = self.agent_enc(a_feat)
        a_emb = a_emb * a_mask.unsqueeze(-1).to(dtype=a_emb.dtype)

        rel = map_center.unsqueeze(1).unsqueeze(1) - agent_pos.unsqueeze(3)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        rel_feat = torch.cat([rel, dist], dim=-1)
        rel_score = self.map_rel_mlp(rel_feat).squeeze(-1)

        q = a_emb
        k = map_node
        dot = torch.einsum("bntd,bmd->bntm", q, k) / math.sqrt(k.shape[-1])
        score = dot + rel_score
        score_mask = map_mask.unsqueeze(1).unsqueeze(1).expand(B, N, T, -1)
        attn = _masked_softmax(score, score_mask, dim=-1)
        map_ctx = torch.einsum("bntm,bmd->bntd", attn, map_node)
        top_poly = attn.argmax(dim=-1).to(torch.long)

        # ------------------------------------------------------------------
        # Pairwise geometry in each agent's local heading frame
        # ------------------------------------------------------------------
        pos_bt = agent_pos.permute(0, 2, 1, 3)  # [B,T,N,2]
        vel_bt = vel.permute(0, 2, 1, 3)        # [B,T,N,2]
        yaw_bt = yaw.permute(0, 2, 1)           # [B,T,N]
        valid_bt = a_mask.permute(0, 2, 1)      # [B,T,N]

        rel_pos_bt = pos_bt.unsqueeze(2) - pos_bt.unsqueeze(3)  # p_j - p_i, shape [B,T,N,N,2]
        rel_vel_bt = vel_bt.unsqueeze(2) - vel_bt.unsqueeze(3)  # v_j - v_i
        yaw_i_bt = yaw_bt.unsqueeze(3)
        rel_s_bt, rel_d_bt = self._rotate_to_heading(rel_pos_bt[..., 0], rel_pos_bt[..., 1], yaw_i_bt)
        rel_vs_bt, rel_vd_bt = self._rotate_to_heading(rel_vel_bt[..., 0], rel_vel_bt[..., 1], yaw_i_bt)
        pair_dist_bt = torch.norm(rel_pos_bt, dim=-1)
        pair_heading_bt = _wrap_angle(yaw_bt.unsqueeze(2) - yaw_bt.unsqueeze(3)).abs()

        valid_i_bt = valid_bt.unsqueeze(3)
        valid_j_bt = valid_bt.unsqueeze(2)
        eye_bt = torch.eye(N, device=device, dtype=torch.bool).view(1, 1, N, N)
        not_self_bt = ~eye_bt

        geo_ok_bt = pair_dist_bt <= self.neighbor_radius_m
        topo_forward_bt = (rel_s_bt > -10.0) & (rel_s_bt < self.topo_horizon_m) & (rel_d_bt.abs() < 8.0)
        raw_neighbor_bt = valid_i_bt & valid_j_bt & not_self_bt & (geo_ok_bt | topo_forward_bt)

        nbr_rel = torch.stack(
            [rel_s_bt, rel_d_bt, rel_vs_bt, rel_vd_bt, pair_dist_bt, pair_heading_bt],
            dim=-1,
        ).permute(0, 2, 1, 3, 4)  # [B,N,T,N,6]
        nbr_score = self.nbr_rel_mlp(nbr_rel).squeeze(-1)

        a_emb_bt = a_emb.permute(0, 2, 1, 3)
        a_emb_i_bt = a_emb_bt.unsqueeze(3)
        a_emb_j_bt = a_emb_bt.unsqueeze(2)
        dot_n_bt = (a_emb_i_bt * a_emb_j_bt).sum(dim=-1) / math.sqrt(a_emb.shape[-1])
        score_n_bt = dot_n_bt + nbr_score.permute(0, 2, 1, 3)
        w_n_bt = _masked_softmax(score_n_bt, raw_neighbor_bt, dim=-1)
        nbr_ctx = (w_n_bt.unsqueeze(-1) * a_emb_j_bt).sum(dim=3).permute(0, 2, 1, 3)

        tau = self.tau_out(torch.cat([a_emb, map_ctx, nbr_ctx], dim=-1))
        tau = tau * a_mask.unsqueeze(-1).to(dtype=tau.dtype)

        # ------------------------------------------------------------------
        # Conflict proxy + topo edge labels
        # ------------------------------------------------------------------
        eps = 1e-6
        rv = (rel_pos_bt * rel_vel_bt).sum(dim=-1)
        vv = (rel_vel_bt * rel_vel_bt).sum(dim=-1).clamp_min(eps)
        t_star_bt = (-rv / vv).clamp(min=0.0, max=self.conflict_time_s)
        closest_bt = rel_pos_bt + rel_vel_bt * t_star_bt.unsqueeze(-1)
        d2_bt = (closest_bt * closest_bt).sum(dim=-1)
        conflict_geom_bt = d2_bt < (4.0 ** 2)

        poly_type_top = self._batch_gather_map(map_poly_type, top_poly)
        region_poly = torch.where(poly_type_top == 1, top_poly + 2, torch.zeros_like(top_poly))
        region_bt = region_poly.permute(0, 2, 1)
        share_region_bt = (region_bt.unsqueeze(3) == region_bt.unsqueeze(2)) & (region_bt.unsqueeze(3) > 0)
        share_region_bt = share_region_bt & valid_i_bt & valid_j_bt & not_self_bt

        conflict_bt = (conflict_geom_bt | share_region_bt) & valid_i_bt & valid_j_bt & not_self_bt
        follow_bt = (rel_s_bt > -5.0) & (rel_s_bt < self.topo_horizon_m) & (rel_d_bt.abs() < 4.5) & (pair_heading_bt < 0.45)
        follow_bt = follow_bt & valid_i_bt & valid_j_bt & not_self_bt
        merge_bt = conflict_bt & (pair_heading_bt < 0.35)
        conflict_only_bt = conflict_bt & (~merge_bt)

        topo_edge_bt = torch.zeros((B, T, N, N), device=device, dtype=torch.long)
        topo_edge_bt = torch.where(follow_bt, torch.ones_like(topo_edge_bt), topo_edge_bt)
        topo_edge_bt = torch.where(conflict_only_bt, torch.full_like(topo_edge_bt, 2), topo_edge_bt)
        topo_edge_bt = torch.where(merge_bt, torch.full_like(topo_edge_bt, 3), topo_edge_bt)

        conflict_mask = conflict_bt.permute(0, 2, 1, 3)
        topological_edge_type = topo_edge_bt.permute(0, 2, 1, 3)
        neighbor_mask = (raw_neighbor_bt | conflict_bt).permute(0, 2, 1, 3)

        conflict_region_id_bt = torch.zeros((B, T, N, N), device=device, dtype=torch.long)
        shared_region_id = torch.where(share_region_bt, region_bt.unsqueeze(3).expand_as(conflict_region_id_bt), torch.zeros_like(conflict_region_id_bt))
        conflict_region_id_bt = torch.where(share_region_bt, shared_region_id, conflict_region_id_bt)
        conflict_region_id_bt = torch.where(conflict_bt & (~share_region_bt), torch.ones_like(conflict_region_id_bt), conflict_region_id_bt)
        conflict_region_id = conflict_region_id_bt.permute(0, 2, 1, 3)

        # ------------------------------------------------------------------
        # Candidate reference paths Π(τ^{det})
        # ------------------------------------------------------------------
        rel_center = map_center.unsqueeze(1).unsqueeze(1) - agent_pos.unsqueeze(3)
        rel_center_s, rel_center_d = self._rotate_to_heading(rel_center[..., 0], rel_center[..., 1], yaw.unsqueeze(-1))
        rel_end = map_end.unsqueeze(1).unsqueeze(1) - agent_pos.unsqueeze(3)
        rel_end_s, rel_end_d = self._rotate_to_heading(rel_end[..., 0], rel_end[..., 1], yaw.unsqueeze(-1))
        map_heading_delta = _wrap_angle(map_heading.unsqueeze(1).unsqueeze(1) - yaw.unsqueeze(-1))

        on_route_e = map_on_route.unsqueeze(1).unsqueeze(1) > 0
        stop_e = has_stop_control.unsqueeze(1).unsqueeze(1)
        lane_e = is_lane.unsqueeze(1).unsqueeze(1)
        conn_e = is_conn.unsqueeze(1).unsqueeze(1)
        map_valid_e = score_mask

        base_forward = (rel_center_s > -10.0) & (rel_center_s < self.topo_horizon_m) & map_valid_e
        prefer_route = on_route_e.to(dtype=dtype)

        # Keep/current path.
        keep_mask = base_forward & (lane_e | conn_e) & (rel_center_d.abs() < 4.5)
        keep_score = 0.35 * rel_center_d.abs() + 0.05 * rel_center_s.abs() + 0.35 * map_heading_delta.abs() - 0.40 * prefer_route
        keep_idx, keep_valid = self._best_index(keep_mask, keep_score)
        keep_valid = a_mask.clone()
        keep_idx = torch.where(keep_valid, keep_idx, top_poly)

        # Lane-change paths.
        left_mask = base_forward & lane_e & (rel_center_d > 1.2) & (rel_center_d < 8.0) & (map_heading_delta.abs() < 0.6)
        left_score = (rel_center_d - self.lane_width_m).abs() + 0.05 * rel_center_s.abs() + 0.25 * map_heading_delta.abs()
        left_idx, left_valid = self._best_index(left_mask, left_score)

        right_mask = base_forward & lane_e & (rel_center_d < -1.2) & (rel_center_d > -8.0) & (map_heading_delta.abs() < 0.6)
        right_score = (rel_center_d + self.lane_width_m).abs() + 0.05 * rel_center_s.abs() + 0.25 * map_heading_delta.abs()
        right_idx, right_valid = self._best_index(right_mask, right_score)

        # Branch connectors.
        conn_ahead = base_forward & conn_e & (rel_center_s > -2.0) & (rel_center_s < 50.0) & (rel_center_d.abs() < 20.0)
        conn_left = conn_ahead & ((map_heading_delta > 0.25) | (rel_end_d > 4.0))
        conn_right = conn_ahead & ((map_heading_delta < -0.25) | (rel_end_d < -4.0))
        conn_straight = conn_ahead & (~conn_left) & (~conn_right)

        straight_score = 0.05 * rel_center_s.abs() + 0.25 * rel_center_d.abs() + 0.25 * map_heading_delta.abs() - 0.20 * prefer_route
        left_turn_score = 0.05 * rel_center_s.abs() + 0.10 * (rel_end_d - 6.0).abs() + 0.20 * (map_heading_delta - 0.6).abs() - 0.20 * prefer_route
        right_turn_score = 0.05 * rel_center_s.abs() + 0.10 * (rel_end_d + 6.0).abs() + 0.20 * (map_heading_delta + 0.6).abs() - 0.20 * prefer_route

        straight_idx, straight_valid = self._best_index(conn_straight, straight_score)
        branch_left_idx, branch_left_valid = self._best_index(conn_left, left_turn_score)
        branch_right_idx, branch_right_valid = self._best_index(conn_right, right_turn_score)

        path_idx = torch.stack(
            [keep_idx, left_idx, right_idx, straight_idx, branch_left_idx, branch_right_idx],
            dim=-1,
        )
        path_valid = torch.stack(
            [keep_valid, left_valid, right_valid, straight_valid, branch_left_valid, branch_right_valid],
            dim=-1,
        )
        fallback_idx = keep_idx.unsqueeze(-1).expand_as(path_idx)
        path_idx = torch.where(path_valid, path_idx, fallback_idx)

        path_node = self._batch_gather_map(map_node, path_idx)
        path_rel_s = self._gather_last_dim(rel_center_s, path_idx)
        path_rel_d = self._gather_last_dim(rel_center_d, path_idx)
        path_heading = self._gather_last_dim(map_heading_delta, path_idx)
        path_curv = self._batch_gather_map(map_curv, path_idx)
        path_length = self._batch_gather_map(map_length, path_idx)
        path_stop_poly = self._batch_gather_map(has_stop_control, path_idx)
        path_on_route = self._batch_gather_map(map_on_route > 0, path_idx)
        path_is_conn = self._batch_gather_map(is_conn, path_idx)

        # Path-specific stop-control reachability (approximation of StopLine in the appendix).
        stop_keep = (conn_ahead & conn_straight & stop_e).any(dim=-1)
        stop_left = (conn_left & stop_e).any(dim=-1)
        stop_right = (conn_right & stop_e).any(dim=-1)
        path_stop = torch.stack(
            [stop_keep, stop_keep, stop_keep, stop_keep, stop_left, stop_right],
            dim=-1,
        ) | path_stop_poly

        path_type_ids = torch.arange(self.max_paths, device=device, dtype=torch.long).view(1, 1, 1, self.max_paths).expand(B, N, T, self.max_paths)
        path_one_hot = F.one_hot(path_type_ids, num_classes=NUM_PATH_TYPES).to(dtype)
        path_feature_extra = torch.stack(
            [
                path_rel_s / max(1.0, self.topo_horizon_m),
                path_rel_d / max(1.0, self.lane_width_m),
                path_heading / math.pi,
                path_curv,
                path_length / max(1.0, self.topo_horizon_m),
                path_stop.to(dtype),
                path_on_route.to(dtype),
                path_is_conn.to(dtype),
            ],
            dim=-1,
        )
        path_features = torch.cat([path_node, path_feature_extra], dim=-1) * path_valid.unsqueeze(-1).to(dtype)

        # ------------------------------------------------------------------
        # K^{cmp}, K^{dyn}, and feasible action set A_t = Y_path x Y_lon(filtered)
        # ------------------------------------------------------------------
        speed = torch.norm(vel, dim=-1)
        max_acc = torch.full((B, N, T), 3.0, device=device, dtype=dtype)
        max_speed = torch.clamp(speed + 10.0, min=5.0, max=30.0)
        max_curv = torch.full((B, N, T), 0.20, device=device, dtype=dtype)
        kin_limits = torch.stack([max_acc, max_speed, max_curv], dim=-1) * a_mask.unsqueeze(-1).to(dtype)

        # Path-conditioned lead candidates.
        target_d = torch.tensor(
            [0.0, self.lane_width_m, -self.lane_width_m, 0.0, 0.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, self.max_paths, 1)
        rel_s_pair = rel_s_bt.permute(0, 2, 1, 3)   # [B,N,T,N]
        rel_d_pair = rel_d_bt.permute(0, 2, 1, 3)
        rel_vs_pair = rel_vs_bt.permute(0, 2, 1, 3)
        pair_heading = pair_heading_bt.permute(0, 2, 1, 3)
        t_star = t_star_bt.permute(0, 2, 1, 3)
        topo_edge = topological_edge_type

        lead_base = neighbor_mask & (rel_s_pair > 0.0) & (rel_s_pair < min(45.0, self.topo_horizon_m)) & (pair_heading < 0.55)
        lead_score_per_path = []
        lead_mask_per_path = []
        yield_score_per_path = []
        yield_mask_per_path = []
        for p in range(self.max_paths):
            td = float(target_d.view(-1)[p].item())
            corridor = 2.8 if p in (int(PathType.KEEP), int(PathType.LANE_CHANGE_LEFT), int(PathType.LANE_CHANGE_RIGHT)) else 4.5
            lead_mask_p = lead_base & ((rel_d_pair - td).abs() < corridor)
            lead_score_p = rel_s_pair + 1.5 * (rel_d_pair - td).abs() + 3.0 * pair_heading
            lead_mask_per_path.append(lead_mask_p)
            lead_score_per_path.append(lead_score_p)

            yield_mask_p = ((topo_edge == 2) | (topo_edge == 3)) & (t_star < self.conflict_time_s)
            if p == int(PathType.BRANCH_LEFT):
                yield_mask_p = yield_mask_p & (rel_d_pair > -8.0)
            elif p == int(PathType.BRANCH_RIGHT):
                yield_mask_p = yield_mask_p & (rel_d_pair < 8.0)
            yield_score_p = t_star + 0.05 * rel_s_pair.abs() + 0.05 * rel_d_pair.abs()
            yield_mask_per_path.append(yield_mask_p)
            yield_score_per_path.append(yield_score_p)

        lead_score = torch.stack(lead_score_per_path, dim=3)    # [B,N,T,P,N]
        lead_mask = torch.stack(lead_mask_per_path, dim=3)
        yield_score = torch.stack(yield_score_per_path, dim=3)
        yield_mask = torch.stack(yield_mask_per_path, dim=3)

        follow_idx_all = []
        follow_valid_all = []
        yield_idx_all = []
        yield_valid_all = []
        for p in range(self.max_paths):
            idx_p, valid_p = self._masked_topk_smallest(lead_score[:, :, :, p, :], lead_mask[:, :, :, p, :], self.max_follow_targets)
            follow_idx_all.append(idx_p)
            follow_valid_all.append(valid_p)
            idx_y, valid_y = self._masked_topk_smallest(yield_score[:, :, :, p, :], yield_mask[:, :, :, p, :], self.max_yield_targets)
            yield_idx_all.append(idx_y)
            yield_valid_all.append(valid_y)
        follow_idx = torch.stack(follow_idx_all, dim=3)     # [B,N,T,P,Kf]
        follow_valid = torch.stack(follow_valid_all, dim=3)
        yield_idx = torch.stack(yield_idx_all, dim=3)       # [B,N,T,P,Ky]
        yield_valid = torch.stack(yield_valid_all, dim=3)

        # Gather source metrics.
        def gather_pair(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return torch.gather(x, dim=-1, index=idx)

        follow_rel_s = gather_pair(rel_s_pair.unsqueeze(3).expand(B, N, T, self.max_paths, N), follow_idx)
        follow_rel_d = gather_pair(rel_d_pair.unsqueeze(3).expand(B, N, T, self.max_paths, N), follow_idx)
        follow_ttc = follow_rel_s / (-gather_pair(rel_vs_pair.unsqueeze(3).expand(B, N, T, self.max_paths, N), follow_idx)).clamp_min(0.5)
        yield_rel_s = gather_pair(rel_s_pair.unsqueeze(3).expand(B, N, T, self.max_paths, N), yield_idx)
        yield_rel_d = gather_pair(rel_d_pair.unsqueeze(3).expand(B, N, T, self.max_paths, N), yield_idx)
        yield_ttc = gather_pair(t_star.unsqueeze(3).expand(B, N, T, self.max_paths, N), yield_idx)

        # Base path metrics reused across action slots.
        base_cmp = torch.stack(
            [
                path_rel_s / max(1.0, self.topo_horizon_m),
                torch.zeros_like(path_rel_s),
                torch.zeros_like(path_rel_s),
                torch.zeros_like(path_rel_s),
                path_stop.to(dtype),
                path_on_route.to(dtype),
            ],
            dim=-1,
        )
        base_dyn = torch.stack(
            [
                path_curv,
                path_heading / math.pi,
                target_d.squeeze(-1).expand(B, N, T, self.max_paths) / max(1.0, self.lane_width_m),
                max_speed.unsqueeze(-1).expand(B, N, T, self.max_paths) / 30.0,
                max_acc.unsqueeze(-1).expand(B, N, T, self.max_paths) / 4.0,
                max_curv.unsqueeze(-1).expand(B, N, T, self.max_paths) / 0.25,
            ],
            dim=-1,
        )

        A = self.num_action_slots
        action_features = torch.zeros((B, N, T, A, self.action_feature_dim), device=device, dtype=dtype)
        feasible = torch.zeros((B, N, T, A), device=device, dtype=torch.bool)
        action_family = torch.zeros((B, N, T, A), device=device, dtype=torch.long)
        action_path_type = torch.zeros((B, N, T, A), device=device, dtype=torch.long)
        action_constraint_type = torch.zeros((B, N, T, A), device=device, dtype=torch.long)
        action_source_index = torch.full((B, N, T, A), -1, device=device, dtype=torch.long)
        comparable_metrics = torch.zeros((B, N, T, A, self.comparable_metric_dim), device=device, dtype=dtype)
        dynamic_metrics = torch.zeros((B, N, T, A, self.dynamic_metric_dim), device=device, dtype=dtype)

        slot_path_types_t = torch.tensor(self.slot_path_types, device=device, dtype=torch.long)
        slot_constraint_types_t = torch.tensor(self.slot_constraint_types, device=device, dtype=torch.long)
        family_lookup = torch.tensor(
            [path_constraint_to_maneuver(int(p), int(c)) for p, c in zip(self.slot_path_types, self.slot_constraint_types)],
            device=device,
            dtype=torch.long,
        )

        for a in range(A):
            p = int(self.slot_path_types[a])
            c = int(self.slot_constraint_types[a])
            r = int(self.slot_source_ranks[a])

            valid = path_valid[..., p] & a_mask
            cmp_a = base_cmp[..., p, :].clone()
            dyn_a = base_dyn[..., p, :].clone()
            src_idx = torch.full((B, N, T), -1, device=device, dtype=torch.long)

            if c == int(LonConstraint.FOLLOW):
                valid = valid & follow_valid[..., p, r]
                src_idx = torch.where(valid, follow_idx[..., p, r], src_idx)
                cmp_a[..., 1] = torch.where(valid, follow_rel_s[..., p, r] / max(1.0, self.topo_horizon_m), torch.zeros_like(cmp_a[..., 1]))
                cmp_a[..., 2] = torch.where(valid, follow_ttc[..., p, r] / max(0.5, self.conflict_time_s), torch.zeros_like(cmp_a[..., 2]))
                cmp_a[..., 3] = torch.where(valid, follow_rel_d[..., p, r].abs() / max(1.0, self.lane_width_m), torch.zeros_like(cmp_a[..., 3]))
            elif c == int(LonConstraint.STOP_LINE):
                valid = valid & path_stop[..., p]
                cmp_a[..., 4] = path_stop[..., p].to(dtype)
            elif c == int(LonConstraint.YIELD_TO):
                valid = valid & yield_valid[..., p, r]
                src_idx = torch.where(valid, yield_idx[..., p, r], src_idx)
                cmp_a[..., 1] = torch.where(valid, yield_rel_s[..., p, r].abs() / max(1.0, self.topo_horizon_m), torch.zeros_like(cmp_a[..., 1]))
                cmp_a[..., 2] = torch.where(valid, yield_ttc[..., p, r] / max(0.5, self.conflict_time_s), torch.zeros_like(cmp_a[..., 2]))
                cmp_a[..., 3] = torch.where(valid, yield_rel_d[..., p, r].abs() / max(1.0, self.lane_width_m), torch.zeros_like(cmp_a[..., 3]))
            else:
                valid = valid

            path_oh = F.one_hot(slot_path_types_t[a].expand(B, N, T), num_classes=NUM_PATH_TYPES).to(dtype)
            lon_oh = F.one_hot(slot_constraint_types_t[a].expand(B, N, T), num_classes=NUM_LON_CONSTRAINTS).to(dtype)
            feat_a = torch.cat([path_node[..., p, :], path_oh, lon_oh, cmp_a, dyn_a], dim=-1)
            feat_a = feat_a * valid.unsqueeze(-1).to(dtype)

            action_features[..., a, :] = feat_a
            feasible[..., a] = valid
            action_family[..., a] = family_lookup[a]
            action_path_type[..., a] = slot_path_types_t[a]
            action_constraint_type[..., a] = slot_constraint_types_t[a]
            action_source_index[..., a] = src_idx
            comparable_metrics[..., a, :] = cmp_a * valid.unsqueeze(-1).to(dtype)
            dynamic_metrics[..., a, :] = dyn_a * valid.unsqueeze(-1).to(dtype)

        # Always keep at least one feasible action for valid tokens.
        keep_free_flow_slot = 0
        feasible[..., keep_free_flow_slot] = feasible[..., keep_free_flow_slot] | a_mask
        action_features[..., keep_free_flow_slot, :] = action_features[..., keep_free_flow_slot, :] + 0.0

        return TemplateEncoding(
            tau=tau,
            map_context=map_ctx,
            nbr_context=nbr_ctx,
            neighbor_mask=neighbor_mask,
            feasible_actions=feasible,
            action_features=action_features,
            action_family=action_family,
            action_path_type=action_path_type,
            action_constraint_type=action_constraint_type,
            action_source_index=action_source_index,
            comparable_metrics=comparable_metrics,
            dynamic_metrics=dynamic_metrics,
            path_valid=path_valid,
            path_features=path_features,
            path_polyline_idx=path_idx,
            topological_edge_type=topological_edge_type,
            kinematic_limits=kin_limits,
            conflict_mask=conflict_mask,
            conflict_region_id=conflict_region_id,
            map_attn=attn,
        )
