from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# Worker pools (nuPlan provides both parallel and sequential executors, names may differ across versions).
try:
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor  # type: ignore
except Exception:  # pragma: no cover
    SingleMachineParallelExecutor = None  # type: ignore


def _make_worker_pool(max_workers: int):
    if SingleMachineParallelExecutor is None:
        raise ImportError(
            "Unable to import nuPlan's SingleMachineParallelExecutor. "
            "Please check your nuplan-devkit installation."
        )

    # If nuPlan provides a sequential worker pool, prefer it when max_workers <= 1.
    if max_workers <= 1:
        try:
            from nuplan.planning.utils.multithreading.worker_sequential import SequentialWorkerPool  # type: ignore

            return SequentialWorkerPool()
        except Exception:
            return SingleMachineParallelExecutor(max_workers=1)
    return SingleMachineParallelExecutor(max_workers=max_workers)


@dataclass
class NuPlanDataConfig:
    """Configuration to build nuPlan scenarios from DB logs."""

    data_root: str
    map_root: str
    map_version: str = "nuplan-maps-v1.0"

    # Can be None / dir / list / file. If None, builder searches under data_root.
    db_files: Optional[str] = None

    # Not used in this repo (no sensors), but scenario builder requires a sensor_root path.
    sensor_root: Optional[str] = None

    include_cameras: bool = False
    max_workers: int = 8

    # Optional filters
    scenario_types: Optional[List[str]] = None
    scenario_tokens: Optional[List[str]] = None
    log_names: Optional[List[str]] = None
    shuffle: bool = True
    expand_scenarios: bool = False
    num_scenarios_per_type: Optional[int] = None
    limit_total_scenarios: Optional[int] = None


def build_scenarios(cfg: NuPlanDataConfig):
    """Build nuPlan scenarios from DB logs using the official scenario builder."""
    sensor_root = cfg.sensor_root or os.path.join(cfg.data_root, "sensor_blobs")

    builder = NuPlanScenarioBuilder(
        data_root=cfg.data_root,
        map_root=cfg.map_root,
        sensor_root=sensor_root,
        db_files=cfg.db_files,
        map_version=cfg.map_version,
        include_cameras=cfg.include_cameras,
        max_workers=cfg.max_workers,
        verbose=True,
    )

    
    # Build ScenarioFilter with best-effort compatibility across nuplan-devkit versions.
    # We only pass arguments that exist in the installed ScenarioFilter __init__ signature.
    sf_kwargs = dict(
        scenario_types=cfg.scenario_types,
        scenario_tokens=cfg.scenario_tokens,
        log_names=cfg.log_names,
        map_names=None,
        shuffle=cfg.shuffle,
        expand_scenarios=cfg.expand_scenarios,
        remove_invalid_goals=True,
        num_scenarios_per_type=cfg.num_scenarios_per_type,
        limit_total_scenarios=cfg.limit_total_scenarios,
        # Optional advanced filters (leave None by default)
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        token_set_path=None,
        fraction_in_token_set_threshold=None,
        ego_route_radius=None,
    )

    import inspect

    sig = inspect.signature(ScenarioFilter.__init__)
    valid_keys = set(sig.parameters.keys())
    # remove 'self'
    valid_keys.discard('self')
    sf_kwargs = {k: v for k, v in sf_kwargs.items() if k in valid_keys}

    scenario_filter = ScenarioFilter(**sf_kwargs)

    worker = _make_worker_pool(cfg.max_workers)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    return scenarios
