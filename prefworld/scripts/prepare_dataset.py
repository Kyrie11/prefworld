from __future__ import annotations
import glob
import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from prefworld.data.cache import CachedSampleIndex, save_npz, write_index_jsonl
from prefworld.data.extractor import ExtractionConfig, extract_sample
from prefworld.data.nuplan_db import NuPlanDataConfig, build_scenarios
from prefworld.utils.config import load_config, make_argparser, parse_overrides


def _scenario_token(scenario) -> str:
    for attr in ["token", "scenario_token", "id"]:
        if hasattr(scenario, attr):
            return str(getattr(scenario, attr))
    # fallback
    return str(hash(scenario))


def _log_name(scenario) -> str:
    if hasattr(scenario, "log_name"):
        return str(getattr(scenario, "log_name"))
    if hasattr(scenario, "_log_name"):
        return str(getattr(scenario, "_log_name"))
    return "unknown_log"


def main() -> None:
    parser = make_argparser("Prepare nuPlan DB -> cached training samples (no sensors)")
    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    cfg = load_config(args.config, overrides)

    split = str(cfg.dataset.split)
    cache_dir = Path(cfg.dataset.cache_dir)
    out_dir = cache_dir / split / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_files = []
    for folder in ['train_boston', 'train_pittsburgh', 'train_singapore']:
        folder_path = Path(f'/coolas-shared/yusz/dataset/nuplan/data/cache/{folder}')
        db_files.extend(list(folder_path.glob('*.db')))

    # Build scenarios
    data_cfg = NuPlanDataConfig(
        data_root=str(cfg.dataset.data_root),
        map_root=str(cfg.dataset.map_root),
        map_version=str(cfg.dataset.map_version),
        db_files=db_files,
        sensor_root=cfg.dataset.get("sensor_root", None),
        include_cameras=False,
        max_workers=int(cfg.dataset.max_workers),
        scenario_types=cfg.dataset.get("scenario_types", None),
        scenario_tokens=cfg.dataset.get("scenario_tokens", None),
        log_names=cfg.dataset.get("log_names", None),
        map_names=cfg.dataset.get("map_names", None),
        timestamp_threshold_s=cfg.dataset.get("timestamp_threshold_s", None),
        ego_displacement_minimum_m=cfg.dataset.get("ego_displacement_minimum_m", None),
        remove_invalid_goals=bool(cfg.dataset.get("remove_invalid_goals", True)),
        shuffle=bool(cfg.dataset.get("shuffle", True)),
        expand_scenarios=bool(cfg.dataset.get("expand_scenarios", False)),
        num_scenarios_per_type=cfg.dataset.get("num_scenarios_per_type", None),
        limit_total_scenarios=cfg.dataset.get("limit_total_scenarios", None),
    )
    scenarios = build_scenarios(data_cfg)

    ext_cfg = ExtractionConfig(**cfg.dataset.extraction)

    stride = int(cfg.dataset.sample_stride)
    max_per_scenario = cfg.dataset.get("max_samples_per_scenario", None)
    max_total = cfg.dataset.get("max_total_samples", None)

    index_items: List[CachedSampleIndex] = []
    total_written = 0

    for sc in tqdm(scenarios, desc=f"Extracting {split}"):
        token = _scenario_token(sc)
        log_name = _log_name(sc)
        n_it = int(sc.get_number_of_iterations())

        # Determine valid iteration range
        min_it = int(cfg.dataset.get("min_iteration", 0))
        # ensure enough past and future
        start_it = max(min_it, ext_cfg.past_num_samples - 1)
        end_it = n_it - ext_cfg.future_num_samples - 1
        if end_it <= start_it:
            continue

        its = list(range(start_it, end_it, stride))
        if max_per_scenario is not None:
            its = its[: int(max_per_scenario)]

        for it in its:
            sample = extract_sample(sc, it, ext_cfg, include_future_agents=True)
            filename = f"{log_name}__{token}__it{it:05d}.npz"
            path = out_dir / filename
            save_npz(path, sample)
            index_items.append(
                CachedSampleIndex(
                    path=str(path),
                    log_name=log_name,
                    scenario_token=token,
                    iteration=int(it),
                )
            )
            total_written += 1
            if max_total is not None and total_written >= int(max_total):
                break
        if max_total is not None and total_written >= int(max_total):
            break

    # Write index
    index_path = cache_dir / split / "index.jsonl"
    write_index_jsonl(index_path, index_items)

    # Save a copy of the effective config for reproducibility
    (cache_dir / split).mkdir(parents=True, exist_ok=True)
    (cache_dir / split / "extraction_config.yaml").write_text(str(cfg), encoding="utf-8")

    print(f"Wrote {len(index_items)} samples to {out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
