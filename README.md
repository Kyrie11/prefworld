# PrefWorld (ICLR 2026 submission repo)

This repo contains a *world model* for multi-agent driving scenes on **nuPlan**:

- **Preference completion**: infer latent per-agent preferences from partial observations.
- **EB-STM**: energy-based structured transition model for interaction graphs.
- **PCI**: identify preference-critical agents via counterfactual rollout divergence.
- **(Optional) planner**: structure sampling + risk aggregation + MPC-style tracking.

Key implementation updates in this version:
- Preference completion uses an **autoregressive motion-primitive likelihood** (maneuver is treated as a *latent*; no maneuver labels required for training).
- EB-STM supports **templateized edit-token candidates**, **beam rollout**, and **predictability-adaptive horizon** (entropy-based early stop).
- PCI is implemented with **shared rollout support** + **counterfactual reweighting** and includes a **risk tie-break** (CVaR-style interaction frequency with ego).

---

## 1) Environment

- Python 3.10+
- PyTorch 2.x
- nuPlan devkit (for scenario reading / map access)

> Note: the dev container used for code review may not include the nuPlan devkit. Runtime usage requires installing nuPlan on your machine.

---

## 2) Dataset layout (your structure)

This repo now supports the following layout directly:

**DBs**
```
/dataset/train_boston
/dataset/train_pittsburgh
/dataset/train_singapore
/dataset/val
```
Each folder should contain one or more `*.db` files.

**Maps**
Either of these layouts is supported:
- (A) Official nuPlan layout:
  ```
  /maps/nuplan-maps-v1.0/<city>/<version>/map.gpkg
  ```
- (B) Flat layout (no `nuplan-maps-v1.0` folder):
  ```
  /maps/<city>/<version>/map.gpkg
  ```
Your example looks like (B), e.g.:
```
/maps/sg-one-north/9.17.1964/map.gpkg   # (or map.pkg if that’s how your export is named)
```

If you use layout (B), set:
- `dataset.map_root: /maps`
- `dataset.map_version: ""` (empty)

---

## 3) Build cache

Edit `prefworld/configs/dataset/nuplan_db.yaml` and set:
- `dataset.data_root`
- `dataset.db_files` (a list of train/val directories)
- `dataset.map_root`
- `dataset.map_version`

Then build cached training samples:

```bash
python -m prefworld.scripts.prepare_dataset   --config prefworld/configs/dataset/nuplan_db.yaml   dataset.split=train   dataset.cache_dir=/path/to/cache/prefworld_train
```

Build cached validation samples:

```bash
python -m prefworld.scripts.prepare_dataset   --config prefworld/configs/dataset/nuplan_db.yaml   dataset.split=val   dataset.db_files=[/dataset/val]   dataset.cache_dir=/path/to/cache/prefworld_val
```

---

## 4) Train

```bash
python -m prefworld.scripts.train   --config prefworld/configs/train/default.yaml   dataset.cache_dir=/path/to/cache/prefworld_train   train.val_cache_dir=/path/to/cache/prefworld_val   train.output_dir=/path/to/outputs/run1
```

Important training knobs:
- `train.w_pc`: weight for preference completion objective
- `train.w_kl`: **used** as the weight for PC KL terms (both KL(q_full||q_ctx) and KL(q_ctx||prior))
- `train.w_intent`: default `0.0` (maneuver is latent / unlabeled).  
  If you want to use heuristic pseudo-labels as weak supervision, enable:
  - `model.use_pseudo_intent=true`
  - `train.w_intent>0`

---

## 5) Validate

```bash
python -m prefworld.scripts.validate   --config prefworld/configs/eval/default.yaml   dataset.cache_dir=/path/to/cache/prefworld_val   eval.checkpoint=/path/to/outputs/run1/checkpoints/best.pt
```

---

## 6) PCI (Preference-Critical Agents)

The PCI implementation lives in:
- `prefworld/planning/pci.py`

It computes PCI using:
- shared structure rollouts `R_1..R_S` under the full belief
- counterfactual reweighting per agent
- Jensen–Shannon divergence between rollout distributions
- optional tie-break risk (CVaR of ego↔agent interaction count)

You can also select agents with the risk tie-break in:
- `prefworld/planning/critical_agents.py`

---

## 7) Planner (structure sampling + risk + MPC)

A lightweight planner is provided in:
- `prefworld/planning/planner.py`

It demonstrates the paper’s logic flow:
1) condition on ego maneuver
2) sample / beam-rollout interaction structures
3) score candidate maneuvers by a risk metric (expected + CVaR)
4) track the chosen reference with a simple gradient-based MPC

This is intentionally minimal and is meant as an integration scaffold.

---

## 8) Repo structure

- `prefworld/data/` — nuPlan DB reading, feature extraction, caching
- `prefworld/models/` — template encoder, preference completion, EFEN, EB-STM, full model
- `prefworld/planning/` — PCI, critical agent selection, (optional) planner + MPC
- `prefworld/training/` — training loops
- `prefworld/scripts/` — CLI entrypoints
