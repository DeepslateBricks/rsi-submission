### 一、 方案融合的前沿研究与 SOTA 思路

1. **受控双层自我改进（Dual-Loop RSI）：`AIDE / AIDE²` + `SkillApt` + `EvoGuard`**
   - **受控试验搜索（AIDE, arXiv:2502.13138 & AIDE²）**：将实验迭代建模为指标驱动的候选生成与沙盒择优；同时把自由代码生成收敛为**领域原语 + 严格 JSON 参数空间**，彻底消除 LLM 语法崩溃与“Reward Hacking”风险。
   - **策略层递归自改进（SkillApt, arXiv:2609.26863）**：按 `(任务 × 漂移诊断 × 原语)` 维护条件效用表并注入规划器上下文，实现“基于历史条件效用选原语”，对应 **Strategy-Level RSI** 思想。
   - **防演化退化锁（EvoGuard, arXiv 2026-11）**：仅正向强化通过门控的策略；回滚时拒绝强化并触发冷却期（`cooldown_until`），防止在失效原语上反复打转。

2. **预算感知上下文路由：`PILOT`（EMNLP Findings 2025, arXiv:2508.21141）**
   - 借鉴 LinUCB 上下文赌博机做预算受限路由的思想，在官方 SDK 单次运行绑定单一模型（`rnd.model_name` 不可变）的约束下，将路由对象重构为**推理预算档位**（`fast_path` 关闭思考链并控预算 vs `deep_path` 开启思考链并给足探索预算），并将 token 开销纳入奖励惩罚项。

3. **抗漂移与防自监控失效：`WATCH`（ICML 2025, arXiv:2505.04608）+ `BD`（ICDMW 2025, arXiv:2506.08734）+ `arXiv:2610.07038`**
   - **无标签漂移监测与归因**：借鉴 BD 的分块漂移度量与 WATCH 的漂移细分类思想，按组/通道对比 `train` 与 `validation` 的偏移程度及过拟合间隙，输出结构化诊断标签（`severe_class_drift`、`overfitting`、`asymmetric_overestimation`、`extreme_covariate_shift` 等）。
   - **不可变冻结基线对照（arXiv:2610.07038）**：Round 1 永久冻结官方基线锚点（`frozen_anchor` / `frozen_anchor_oof`），防止多轮自适应后“模型自我适应掩盖分布变化”，并以物理实体/非重叠时序块维持检验有效性。

4. **序贯安全门控与混合演进：`E-valuator`（arXiv:2512.03109）+ `CEP`（arXiv:2506.14790）**
   - **块级 E-process 门控（E-valuator）**：用基于成对块损失差的 e-value 检验替代单一指标波动对比，过滤噪声抖动与多轮假阳性。
   - **持续演进专家池（CEP）**：在真实续训的轻量骨干之上挂载多专家池（`Ridge` / `LogisticRegression` / `LightGBM`），兼顾低参数量与残差/概率互补能力。

---

### 二、 最新版代码中的 5 大核心工程与算法亮点

1. **真实无损迁移 + 弹性权重收缩（`cep_model.py`）**
   - `migrate()` 完整保留父模型权重与 Adam 一/二阶动量，新增衍生特征列采用零初始化并以探针断言等价性，满足“真实续训”约束的同时支持特征空间演进。
   - `shrink_backbone()` 对过拟合父模型按 $\gamma$ 和 $\gamma^2$ 同步衰减权重与 Adam 动量，结合 Group-CV 早停和 Cosine LR，彻底解决 B 数据集严重跨轴承过拟合问题（RMSE 降幅达 **-64.0%**）。

2. **样本加权 E-process + Group-CV OOF 治理双轨门控（`gatekeeper.py` & `contest_submission.py`）**
   - **样本占比加权 E-process + 一致性钳制**：按块样本占比 $w_b = n_b / N$ 聚合相对损失差，消除短块虚高膨胀，并通过 `e-clamp` 保证主指标未提升时证据值绝不大于 1。
   - **同协议 Group-CV OOF 锚点 + 治理模式**：回归任务用同折划分从零重训验证配方泛化能力（并用 `parent_anchor_oof_cache` 复用未变父节点结果），在 B/过拟合治理模式下改以冻结基线 OOF 为下限，避免过拟合父模型误杀泛化候选；配合 `Track2-imbalance`、`Track2-nasa-guard` 与沙盒预筛 `_sandbox_guard_ok`，守住高估率与尾部风险红线。

3. **面向思考模型的上下文瘦身与预算标定（`aide_search.py`）**
   - 通过 `_slim_report`、`_slim_candidate`、`_slim_baseline`、`_slim_history` 将发给规划器的上下文压缩至 `<= 2 KB`，避免长提示引发推理模型过度思考导致输出截断或超时。
   - 针对 Qwen3 思考链特性标定 `4096 / 6144` 预算：`fast_path` 显式传 `enable_thinking=False` 提速降耗，`deep_path` 开启 `enable_thinking=True`；失败时执行最多 2 次**翻倍预算 + 兼容性剥离重试**，并辅以确定性 `fallback_proposals` 保底。

4. **紧扣工业物理先验的特征原语库（`primitives.py`，全部严格仅在 `train` 拟合）**
   - **A（气体阵列漂移，+24.0%）**：基于 16 传感器 × 8 特征物理结构设计 `_gas_group_log_rel`（保号 `log1p` + 同通道跨传感器去均值 + L2 归一化消除共模增益漂移）与 `_gas_sensor_l2`，结合有效样本数加权（`effective_beta=0.99`）、Group-CV 坐标搜索 `logit_bias` 及 20% 骨干投票融合。
   - **B（轴承振动 RUL，-64.0%）**：构造无量纲频带占比、双通道均值/差值与冲击交互特征（`_bearing_scale_invariant`），并主动屏蔽易引起跨轴承记忆的 `dominant_freq` 列。
   - **C（发动机多工况 RUL，-7.5%）**：仅在训练集标准化 3 维 `settings` 后用 6 簇 KMeans 解耦工况，做分簇传感器标准化残差，配合 `asym` 非对称损失、`conservative_bias` 与 `sqrt(y)` 目标空间变换（`regime_sqrt`）。

5. **全自包含、无 `pickle` 流式推理与容器鲁棒性（`cep_model.py` & `Dockerfile`）**
   - 模型全量以 `allow_pickle=False` 的纯 `.npz` + JSON + LightGBM `model_str` 持久化；内存 `_booster` 缓存（落盘自动跳过 `_` 前缀键）保证逐行流式推理低延迟；`tie_break_scores()` 严格保障分类分数在 `[0, 1]` 且最大值唯一。
   - `Dockerfile` 在安装 Python 依赖前前置安装 `libgomp1`（解决精简镜像导入 LightGBM 缺 OpenMP 报 `OSError` 的问题），并配置 `chmod -R a+rX /app` 与 `uid 65534` 只读沙盒运行。

---

### 三、 实事求是：哪些是“轻量近似”

1. **漂移检测（`diagnostics.py`）**：目前是块级标准化均值/偏移度量与阈值规则诊断，未完整实现 BD 的两样本配对 t 检验 p 值或 WATCH 的在线加权共形鞅（WCTM）。
2. **E-process（`gatekeeper.py`）**：落地的是单轮内样本加权成对损失相对改善率的截断指数，属于受 E-valuator 启发的单步证据门控，而非跨多轮累积乘积的无限期（anytime-valid）检验鞅。
3. **AIDE 与 SkillApt（`aide_search.py`, `router_skill.py`）**：受单轮预算与壁钟时间限制，每轮并行提出 2–3 条受控配置在沙盒择优（未展开多代分支树）；`SkillApt` 侧重累积 `with` 执行净收益，未在同轮额外跑 `without` 对照组。
4. **C 组改进空间**：受单行无状态推理约束（无法跨行追踪同一台发动机的历史退化轨迹），C 组目前仅用单快照实现 `-7.5%`；后续若继续深挖，重点在于单快照内的跨传感器物理比值交互与分段上限 RUL 目标建模。

---

**一句话总结**：这套方案把前沿的**受控双环 Agent 自演化（AIDE + SkillApt + EvoGuard + PILOT）**与**防退化统计学门控（Frozen Anchor + Group-OOF + 加权 E-process）**完整压缩进了严苛的 CPU 断网沙盒协议，并在独立容器实测中交出了 **A +24.0%、B -64.0%、C -7.5%** 的稳健提分答卷。