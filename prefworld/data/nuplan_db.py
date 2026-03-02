from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter


PathLike = Union[str, Path]
DBFilesLike = Union[str, Sequence[str], None]


def _as_list(x: DBFilesLike) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(p) for p in x]
    return [str(x)]


def _expand_db_files(data_root: str, db_files: DBFilesLike) -> List[str]:
    """Resolve db_files into an explicit list of *.db paths.

    Supports:
      - a single path (file/dir/glob)
      - a comma-separated string of paths
      - a list of paths
      - None: will search under data_root for *.db files (shallow first, then recursive as fallback)
    """
    if isinstance(db_files, str) and "," in db_files:
        items = [p.strip() for p in db_files.split(",") if p.strip()]
        db_files = items

    items = _as_list(db_files)
    db_paths: List[str] = []

    def add_from_path(p: str) -> None:
        if any(ch in p for ch in ["*", "?", "["]):
            for g in glob.glob(p):
                add_from_path(g)
            return
        if os.path.isdir(p):
            # Prefer shallow search; fallback to recursive
            found = list(Path(p).glob("*.db"))
            if not found:
                found = list(Path(p).rglob("*.db"))
            db_paths.extend([str(f) for f in found])
        elif os.path.isfile(p) and p.endswith(".db"):
            db_paths.append(p)

    if not items:
        # Attempt to infer from data_root:
        # - common layout: /dataset/train_boston/*.db etc.
        root = Path(data_root)
        # shallow search up to 2 levels to avoid huge recursion
        found = list(root.glob("*.db"))
        if not found:
            found = list(root.glob("*/*.db"))
        if not found:
            found = list(root.rglob("*.db"))
        db_paths = [str(f) for f in found]
        return sorted(set(db_paths))

    for p in items:
        add_from_path(p)

    return sorted(set(db_paths))


def _resolve_sensor_root(data_root: str, sensor_root: Optional[str]) -> str:
    """Try to resolve a reasonable sensor_root; create directory if absent.

    This repo typically uses `include_cameras=False`, so sensor_root is mainly needed because
    the nuPlan ScenarioBuilder expects a path. An empty directory is acceptable for most use cases.
    """
    if sensor_root:
        os.makedirs(sensor_root, exist_ok=True)
        return sensor_root

    candidates = [
        os.path.join(data_root, "sensor_blobs"),
        os.path.join(os.path.dirname(data_root), "sensor_blobs"),
        os.path.join(data_root, "sensor_data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # fallback: create under data_root
    fallback = candidates[0]
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _resolve_map_version(map_root: str, map_version: str) -> str:
    """If map_root/map_version doesn't exist, fall back to empty version."""
    if not map_version:
        return ""
    if os.path.isdir(os.path.join(map_root, map_version)):
        return map_version
    # Many users store maps directly as: /maps/<city>/<version>/map.gpkg
    # In that case, map_root already points to the version container and map_version should be "".
    return ""

def _build_pool(num_workers: int):
    """Build a worker/pool compatible across nuplan-devkit versions."""
    import inspect

    # Most common across nuplan versions/forks
    try:
        from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
    except ImportError:
        # Some forks/older layouts
        from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

    n = int(num_workers)

    # Build kwargs safely across signature variants
    sig = inspect.signature(SingleMachineParallelExecutor.__init__)
    params = sig.parameters

    kwargs = {}
    if "max_workers" in params:
        kwargs["max_workers"] = n
    elif "num_workers" in params:
        kwargs["num_workers"] = n

    # Some versions expose these knobs
    if "use_process_pool" in params:
        kwargs["use_process_pool"] = (n > 1)
    if "use_thread_pool" in params:
        kwargs["use_thread_pool"] = True

    return SingleMachineParallelExecutor(**kwargs)


def _get_scenarios(builder, scenario_filter, max_workers: int = 4):
    """
    Call NuPlanScenarioBuilder.get_scenarios across different nuplan-devkit APIs.

    Handles variants:
      - get_scenarios(filter, worker=pool)   (worker required)
      - get_scenarios(filter, worker_pool=pool)
      - get_scenarios(filter, num_workers=K)
      - get_scenarios(filter)  (no parallelism)
    Also handles pool lifecycle when pool is NOT a context manager.
    """
    import inspect

    def _start_pool(pool):
        # Different versions use different start semantics
        if hasattr(pool, "start") and callable(getattr(pool, "start")):
            pool.start()

    def _stop_pool(pool):
        # Try common shutdown/close/stop/terminate names
        for name in ("shutdown", "close", "stop", "terminate", "join"):
            if hasattr(pool, name) and callable(getattr(pool, name)):
                try:
                    getattr(pool, name)()
                except TypeError:
                    # some methods require args; ignore
                    pass
                break

    sig = inspect.signature(builder.get_scenarios)
    params = sig.parameters

    # 1) Version where worker is required (your earlier报错就是这个)
    if "worker" in params:
        pool = _build_pool(max_workers)
        _start_pool(pool)
        try:
            # Prefer keyword, but some versions are positional-only
            try:
                return builder.get_scenarios(scenario_filter, worker=pool)
            except TypeError:
                return builder.get_scenarios(scenario_filter, pool)
        finally:
            _stop_pool(pool)

    # 2) Version where worker_pool is used
    if "worker_pool" in params:
        pool = _build_pool(max_workers)
        _start_pool(pool)
        try:
            return builder.get_scenarios(scenario_filter, worker_pool=pool)
        finally:
            _stop_pool(pool)

    # 3) Version where only number of workers is accepted
    if "num_workers" in params:
        return builder.get_scenarios(scenario_filter, num_workers=int(max_workers))
    if "num_worker" in params:
        return builder.get_scenarios(scenario_filter, num_worker=int(max_workers))

    # 4) No parallelism support
    return builder.get_scenarios(scenario_filter)



@dataclass
class NuPlanDataConfig:
    """Config for loading nuPlan DB scenarios."""

    data_root: str
    map_root: str
    map_version: str = "nuplan-maps-v1.0"

    # Can be a file, dir, glob, comma-separated string, or list of these.
    db_files: DBFilesLike = None

    # Optional; if None we'll try to infer.
    sensor_root: Optional[str] = None

    include_cameras: bool = False
    max_workers: int = 4

    # ScenarioFilter fields
    scenario_types: Optional[List[str]] = None
    scenario_tokens: Optional[List[str]] = None
    log_names: Optional[List[str]] = None
    shuffle: bool = True
    expand_scenarios: bool = False
    num_scenarios_per_type: Optional[int] = None
    limit_total_scenarios: Optional[int] = None

    map_names: Optional[List[str]] = None
    timestamp_threshold_s: Optional[float] = None
    ego_displacement_minimum_m: Optional[float] = None
    remove_invalid_goals: bool = True

def build_scenarios(cfg: NuPlanDataConfig):
    """Build a list of nuPlan scenarios according to cfg."""
    db_list = _expand_db_files(cfg.data_root, cfg.db_files)
    if len(db_list) == 0:
        raise FileNotFoundError(
            f"No .db files found. data_root={cfg.data_root!r}, db_files={cfg.db_files!r}. "
            f"If your dataset is split into folders like train_boston/train_pittsburgh/val, set db_files to a list of those directories."
        )

    sensor_root = _resolve_sensor_root(cfg.data_root, cfg.sensor_root)

    map_version = _resolve_map_version(cfg.map_root, cfg.map_version)
    builder = NuPlanScenarioBuilder(
        data_root=str(cfg.data_root),
        map_root=str(cfg.map_root),
        db_files=db_list,
        sensor_root=str(sensor_root),
        map_version=str(map_version),
        include_cameras=bool(cfg.include_cameras),
    )

    scenario_filter = ScenarioFilter(
        scenario_types=cfg.scenario_types,
        scenario_tokens=cfg.scenario_tokens,
        log_names=cfg.log_names,
        map_names=cfg.map_names,
        timestamp_threshold_s=cfg.timestamp_threshold_s,
        ego_displacement_minimum_m=cfg.ego_displacement_minimum_m,
        remove_invalid_goals=cfg.remove_invalid_goals,
        shuffle=cfg.shuffle,
        expand_scenarios=cfg.expand_scenarios,
        num_scenarios_per_type=cfg.num_scenarios_per_type,
        limit_total_scenarios=cfg.limit_total_scenarios,
    )
    return _get_scenarios(builder, scenario_filter, max_workers=cfg.max_workers)
