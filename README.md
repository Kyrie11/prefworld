# PrefWorld (PyTorch) — nuPlan DBs + HD Map (No Sensors)

This repository implements a **preference-conditioned structured world model** designed for:
- **Training** on nuPlan DB logs (SQLite `.db`) and HD maps (nuplan-maps-v1.0)
- **Validation** on held-out logs/scenarios
- **No raw sensor input** (no images / point clouds)

It provides:
- Streaming **Preference Completion** (Gaussian belief update in natural-parameter form)
- **Intention (maneuver) prediction** conditioned on preferences and ego candidate plan
- **EB-STM** (Energy-Based Structured Transition Model) over discrete interaction structures
- **PCI** (Preference Criticality Index) for selecting critical agents (information-value based)

> ⚠️ This repo depends on `nuplan-devkit` for reading DB logs and maps. Install nuPlan first.

---

## 1) Installation

### 1.1 Install nuPlan devkit (recommended: from source)
```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
git checkout v1.2
pip install -e .
```

### 1.2 Install this repo
```bash
cd prefworld
pip install -e .
```

---

## 2) Data Layout

We assume nuPlan v1.1 dataset layout similar to the official devkit docs. A typical layout:
```
NUPLAN_DATA_ROOT/
  nuplan-v1.1/
    splits/
      train/   (db files)
      val/
      test/
    maps/
      nuplan-maps-v1.0/
        <city>/
          map.gpkg
```

You will need:
- `data_root`: directory containing the `.db` logs (or a directory that contains split subdirs)
- `map_root`: directory containing `nuplan-maps-v1.0/`

---

## 3) Preprocessing (cache features/labels)

We recommend caching samples to disk for fast training.

```bash
python -m prefworld.scripts.prepare_dataset \
  --config prefworld/configs/dataset/nuplan_db.yaml \
  dataset.data_root=/path/to/nuplan-v1.1/splits/train \
  dataset.map_root=/path/to/nuplan-v1.1/maps \
  dataset.cache_dir=/path/to/cache/prefworld \
  dataset.split=train

python -m prefworld.scripts.prepare_dataset \
  --config prefworld/configs/dataset/nuplan_db.yaml \
  dataset.data_root=/path/to/nuplan-v1.1/splits/val \
  dataset.map_root=/path/to/nuplan-v1.1/maps \
  dataset.cache_dir=/path/to/cache/prefworld \
  dataset.split=val
```

This will create:
```
cache_dir/
  train/
    index.jsonl
    samples/
      <log>__<scenario_token>__itXXXX.npz
  val/
    index.jsonl
    samples/
      ...
```

---

## 4) Training

```bash
python -m prefworld.scripts.train \
  --config prefworld/configs/train/default.yaml \
  dataset.cache_dir=/path/to/cache/prefworld \
  train.output_dir=/path/to/outputs/run1
```

Checkpoint will be saved under `output_dir/checkpoints/`.

---

## 5) Validation

```bash
python -m prefworld.scripts.validate \
  --config prefworld/configs/eval/default.yaml \
  dataset.cache_dir=/path/to/cache/prefworld \
  eval.checkpoint=/path/to/outputs/run1/checkpoints/best.pt
```

---

## 6) Notes / Design choices

- **No sensors**: We use only ego + tracked objects + traffic lights from DBs and HD maps.
- **Pseudo-labels**:
  - Maneuver labels are computed from future kinematics (lane-change/turn/stop heuristics).
  - Interaction structure labels are computed from conflict-region overlap and temporal ordering.
- You can improve labels using more advanced map-aware logic; the code is modular for that.

---

## 7) Directory structure

- `prefworld/data/` — nuPlan DB reading, feature extraction, caching
- `prefworld/models/` — PCNet, IntentionNet, EFEN, EB-STM, combined model
- `prefworld/planning/` — PCI and (optional) inference utilities
- `prefworld/training/` — losses, metrics, training loops
- `prefworld/scripts/` — CLI entrypoints

