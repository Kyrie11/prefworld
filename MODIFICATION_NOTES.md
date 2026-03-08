# PrefWorld 论文对齐修改说明

本次修改以论文 `method` 和 `appendix` 为主线，重点把旧版 `maneuver-only` 实现改成论文当前叙事中的 **reference-path branch × longitudinal constraint source** 动作表示，同时尽量保持现有 Stage0/1/2/3/4 的训练入口不变。

## 1. 动作空间与标签体系

修改文件：`prefworld/data/labels.py`

- 保留旧版 6 类 coarse maneuver（KEEP / LCL / LCR / TURN_L / TURN_R / STOP）作为：
  - 兼容原有 cache 标签；
  - 训练日志/指标；
  - planner fallback。
- 新增论文动作空间：
  - `PathType = {KEEP, LCL, LCR, BRANCH_STRAIGHT, BRANCH_LEFT, BRANCH_RIGHT}`
  - `LonConstraint = {FREE_FLOW, FOLLOW(j), STOP_LINE, YIELD_TO(j)}`
- 提供 `path_constraint_to_maneuver()` 映射，把论文动作槽聚合回旧版 coarse maneuver family。

## 2. TemplateEncoder 重构（最核心）

修改文件：`prefworld/models/template_encoder.py`

现在 `TemplateEncoding` 会显式输出：

- `neighbor_mask`
- `topological_edge_type`
- `path_valid`
- `path_features`
- `feasible_actions`（注意：现在是 structured action slots，不再是 6 个 maneuver）
- `action_features`
- `action_family`
- `action_path_type`
- `action_constraint_type`
- `action_source_index`
- `comparable_metrics` / `dynamic_metrics`

实现上对齐论文 appendix 的几个关键点：

- 邻域：几何半径 + 拓扑前向 reachability + conflict proxy
- 候选路径：KEEP / legal LCL / legal LCR / branching connectors
- 动作集合：`Y_path × Y_lon` 的固定槽位化张量实现
- `StopLine` / `Follow(j)` / `YieldTo(j)` 的路径条件化筛选
- `K_cmp` 与 `K_dyn` 分开输出
- 默认超参向论文 appendix 靠拢：
  - `R_geo = 60m`
  - `H_topo = 80m`
  - `T_conf = 5s`

## 3. Preference Completion 改为 structured action decoder

修改文件：

- `prefworld/models/motion_primitives.py`
- `prefworld/models/preference_completion.py`

主要变化：

- 旧版 decoder 直接对 6 类 maneuver 建模；
- 现在 decoder 接收 `action_features`，对 structured action slots 建模；
- 最终通过 `log-sum-exp` 聚合回 6 类 coarse maneuver logits，保证：
  - 训练脚本基本不用改；
  - 原有 intent 指标还能继续看。

输出新增：

- `action_logits_last`
- `action_family_last`
- 同时保留 `maneuver_logits_last`

## 4. 主模型 PrefWorldModel 对齐

修改文件：`prefworld/models/prefworld_model.py`

主要变化：

- Stage1 的 PC 训练现在使用 `TemplateEncoder` 输出的 structured action slots；
- `EB-STM` 的 ego 条件输入不再固定是 maneuver one-hot，而是从当前 deterministic template 中选取与 coarse family 对应的 **代表性 structured action feature**；
- `aux` 中新增：
  - `action_logits_last`
  - `action_family_last`
  - `ego_feasible_actions_last`
  - `ego_action_features_last`
  - `ego_action_family_last`

## 5. Planner 兼容更新

修改文件：`prefworld/planning/planner.py`

- 将 planner 中的 feasibility 检查从“按 6 类 maneuver 直接索引”改成：
  - 先把 structured action feasibility 聚合成 coarse family feasibility；
  - 再做 KEEP/LCL/LCR/...
- EB-STM 评估 ego action 时，传入当前 ego 的 structured action catalogue，用于选择 family 对应的代表性动作特征。
- 顺手修复了旧版 `_ego_reference_primitive()` 在 batch=1 时 lane change / turn 参考轨迹维度可能塌成一维的问题。

## 6. 训练/脚本兼容

修改文件：

- `prefworld/scripts/sanity_check_feasible_actions.py`
- 新增 `prefworld/configs/train/stage2_ebstm.yaml`

说明：

- 你的执行顺序文档里使用 `stage2_ebstm.yaml`，仓库原本只有 `stage2_eb.yaml`，现在已补一个同内容别名文件，直接按你的执行顺序文档使用即可。
- sanity-check 脚本也已改成先把 structured action feasibility 聚合为 coarse family 再做离线检查。

## 7. 兼容性说明

- **缓存格式没有强制改动**：仍可继续使用你现有的 nuPlan cache。
- 原有 `ego_maneuver / agents_maneuver` 未来轨迹标签仍保留，只作为：
  - 兼容旧指标；
  - 可选的 weak supervision / ablation；
  - planner coarse fallback。
- 论文中的完整 reference-path Frenet + `xi = [delta d, a_adj, rho]` 闭式 / 半闭式求解，在当前仓库 map/cache 结构下很难一次性彻底重写成工程可直接跑的版本；本次实现优先完成了 **离散动作空间、template 表示、PC 解码器、EB 条件接口、planner feasibility** 这条主干的一致性。

## 8. 建议运行顺序

仍按你的文档：

1. `prepare_dataset`
2. `train_stage1_pc`
3. `train_stage2_ebstm`
4. `train_stage3_joint`
5. nuPlan simulation

其中 Stage2 现在可以直接使用：

```bash
python -m prefworld.scripts.train_stage2_ebstm \
  --config prefworld/configs/train/stage2_ebstm.yaml \
  dataset.cache_dir_train=/ABS/PATH/prefworld_train \
  dataset.cache_dir_val=/ABS/PATH/prefworld_val
```

## 9. 进一步对齐论文最新版 Method（本次补充修改）

本次在你给的 `iclr2026_conference.tex` 的 Method / Appendix 基础上，又做了几处关键对齐（主要集中在 `motion_primitives.py` 和 `preference_completion.py`）：

### 9.1 Motion Primitive Likelihood 与 Preference 解耦

- **对齐点**：论文明确写了 $p_n(X|m,\tau^{det})$ 与偏好向量 $z$ **无关**。
- **实现**：`MotionPrimitiveDecoder.token_log_prob()` 现在只依赖 `x/tau/ctx + comparable_metrics/dynamic_metrics + map_polylines/path_polyline_idx`，不再把 `z` 输入到 primitive likelihood。

### 9.2 Choice Evidence（q_χ）驱动的 Preference Completion

- **对齐点**：论文中的 evidence update 使用 $q_\chi(m|X_t,\tau_t^{det})$（soft choice evidence）而非 raw kinematics。
- **实现**：decoder 内部先计算：
  - recognition distribution `recog_probs = q_chi`（每个 token 的 action posterior），
  - confidence gate `recog_conf = 1 - H(q_chi)/log|A_t|`。
  然后 PC 模块用 `E_{m~q_chi}[f(m,\tau^{det})]` 作为证据输入，`alpha_t = recog_conf` 直接作为可靠度门控。

### 9.3 Preference Completion 训练目标替换为论文 Eq.(pc_choice)

- **对齐点**：论文的主要优化项是：
  $$\mathrm{KL}(q_\phi(z|D_C)||p_\theta(z|c)) - \mathbb{E}_{z\sim q_\phi} \sum_{t\in Q} \mathbb{E}_{m\sim q_\chi} \log \pi_\theta(m|\tau^{det},z).$$
- **实现**：`loss_query_nll` 现在是 query token 上的 **soft-target cross entropy**：
  $$-\sum_m q_\chi(m)\log \pi_\theta(m|\tau^{det},z)$$
  并对 `z ~ q(z|C)` 做 Monte-Carlo 平均。

### 9.4 Invariance Regularizer 改为论文 Eq.(rinv_def) 的对称 KL

- **对齐点**：论文的 $\mathcal{R}_{inv}$ 是两个 sub-context posterior 的对称 KL：
  $$\tfrac12\mathrm{KL}(q(z|C_1)||q(z|C_2)) + \tfrac12\mathrm{KL}(q(z|C_2)||q(z|C_1)).$$
- **实现**：`loss_contrastive` 现在对应这个对称 KL（而不是 InfoNCE）。
  Sub-context 采样做了一个工程近似：在 `ctx_mask` 内用 `recog_conf` 作为 evidence strength，贪心选 token 使得两份子集的 evidence strength 近似匹配。

### 9.5 u_ctx baseline 与正则

- **对齐点**：论文指出 `u_ctx(τ)` 是 preference-independent baseline，并应保持 mean-zero（对可行动作）。
- **实现**：`PaperStructuredActionUtility` 里每个时间步会对 `u_ctx` 做 feasible-set mean subtraction；PC 端额外提供 `loss_u_ctx`（默认权重 `lambda_u_ctx`）。

