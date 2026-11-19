# `.eval` — 独立评测端

这个目录模拟竞赛平台，对 `submission_project` 做端到端独立评测。它不是提交的一部分，
不进入镜像，也不影响提交内容。

## 它做什么

`eval.py` 扮演宿主，通过真实线协议（`contest.harness.workspace.v2`）驱动提交镜像：

1. **describe 自检** — 跑 `python3 /app/harness.py describe`，检查协议与任务声明。
2. **静态合规审计** — 扫描 `app/` 源码，查硬编码密钥、写 `/app`、访问隐藏数据、
   绕过网关的外网请求等契约违规。
3. **独立测量父基线** — 用数据包自带的 `baselines/predict.py` 在公开验证集上跑一次，
   得到评测端自己算出的父模型分数，不采信提交方自述。
4. **每一轮**：发送 `start` 帧 → 应答 `history_request` / `model_request` /
   `state_request` / `candidate_request` / `artifact_read_request` → 收到 `round_complete`。
5. **独立测量候选** — 把候选文件挂到 `/model`，按 `predict_argv` 启动预测进程，
   用公开验证集的真实标签算分（A 用 Macro-F1，B/C 用 RMSE/MAE/NASA）。
6. **独立判定** — 只用自己测出的数字判 `improved` / `not_improved` / `kept_parent`，
   并核对协议卫生（`model_call_ids` 是否真实签发给提交方、`history_ids` 是否合法等）。

## 用法

```sh
export PYTRIO_API_KEY=trio_...

# 正式评测：竞赛要求 A/B/C 各 10 轮（默认值即如此，可直接跑）
python3 .eval/eval.py

# 等价写法
python3 .eval/eval.py --datasets A B C --rounds 10

# 冒烟：只验协议链路，不等同于评测成绩
python3 .eval/eval.py --stub --datasets A --rounds 1 --measure-limit 200

# 清理旧运行目录（容器以 uid 65534 写入，宿主可能删不掉）
python3 .eval/eval.py --clean --stub --datasets A --rounds 1
```

默认值按竞赛文档设定（`.workspace/main.md:11`「每个数据集进行 10 轮优化」）：
数据集 `A B C`、每集 `10` 轮、模型自动发现。**少跑轮次不会得到合格结论**，
报告末尾的「竞赛硬性指标」表会直接标出未通过项。

### 模型通道：只允许 PyTRiO

比赛规定只能用 PyTRiO，脚本因此只对接 `https://pytrio.com`：

- 对话端点 `POST /api/openai/v1/chat/completions`（注意 `/api/openai/v1` 段）
- 认证需要**同时**带 `Authorization: Bearer <key>` 和 `X-API-Key: <key>`，
  只带 Bearer 会 401
- 模型从 `GET /api/models` 发现；未指定 `--model` 时优先选 4B
- PyTRiO 是平台内网关，请求强制绕过本地代理（`trust_env = False`），
  否则 `http_proxy` 会让请求挂死

密钥只从 `PYTRIO_API_KEY` 读取，不落盘、不传入容器。脚本启动时会检查 key 是否
以 `trio_` 开头，避免把第三方 key 发到 PyTRiO 白跑一轮。

模型名的来源优先级：`--model` > `PYTRIO_MODEL` > 自动发现。

## 产物

每次运行写到 `.eval/runs/<时间戳>/`：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 人读报告（含上表所有章节与结论） |
| `report.json` | 机读结果（含 `contest_checks` 硬性指标判定） |
| `events.jsonl` | 每一帧协议、每一次测量与判定的完整轨迹 |
| `staging/` | 该轮的 `/data`、`/previous`、`/model` 挂载内容 |
| `workspace/` | 提交方在 `/workspace` 里写下的全部中间产物 |
| `state/` | 跨轮原生状态 |

## 已知前提

- **Docker 镜像**：默认 `rsi-submission:2.0`，可用 `--image` 指定。镜像必须能从
  `app/` 构建（`docker build -t rsi-submission:2.0 .`）。
- **libgomp**：`python:3.12-slim-bookworm` 不自带 `libgomp.so.1`，而 LightGBM 的
  扩展在导入期 dlopen 它。缺库时 `import lightgbm` 抛的是 `OSError` 而非
  `ImportError`，早期版本因此让 B/C 的每个回归 trial 全部失败。
  现在已两层修复：`Dockerfile` 安装 `libgomp1`，`cep_model.py` 同时捕获
  `ImportError` 与 `OSError`（训练侧退化为无树专家组合，推理侧给出可诊断错误）。
  `eval.py` 保留了一个兼容用的自动修补，仅在镜像确实缺库时触发；用
  `--no-libgomp-repair` 可强制走干净路径来验证镜像本身。
- **验证集**：默认用完整公开验证集评分。`--measure-limit N` 只用于快速冒烟，
  报告会如实写出实际行数，不把截断分数当成全量成绩。

## 勘误：思考模型的 token 预算

开发期的「live 模式」日志显示：选定 `Qwen/Qwen3.8-27B` 后连续出现
**8 次** `live model failed, using stub: The read operation timed out`，成功 **0 次**
（`_live_model()` 的 `except` 分支静默 fallback）。因此开发期所谓「stub 与 live
结果一致」，实际是两种模式跑的都是同一套 stub。历史数据不作为参考，
v1 成绩以本评测端的独立测量为准。

### 根因：reasoning 吃掉全部预算

Qwen3.5/3.8 是思考模型，先写 `reasoning` 再写 `content`。用真实尺寸的诊断
prompt（3550 字符）实测 27B：

| max_tokens | 耗时 | completion | reasoning | content 字符 | JSON 可解析 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 30s | 1024 | 1024（耗尽） | 0 | 否 |
| 2048 | 48s | 2048 | 1555 | 1212（截断） | 否 |
| 4096 | 91s | 3656 | 2772 | 2561 | **是** |

提交端 `fast_path` 的保底是 `min(1024, rnd.max_output_tokens)`，在思考模型上
必然拿不到有效 JSON，于是每轮退回模板提案 —— 这正是「真实提案被采用 = 0」的机制。

评测端因此做三件事，而不是替提交端兜底：

1. `MAX_OUTPUT_TOKENS`（平台授权上限）设为 16384，不人为压低；
2. 逐次记录 `requested_max_tokens / completion_tokens / reasoning_tokens /
   content_chars / content_parsable_json`，写进报告的「模型调用明细」表；
3. 硬性指标新增「模型回执可用」，可用率 < 50% 即判不通过 —— 分数提升若来自
   模板提案而非模型决策，不应算作 RSI 生效。

### 另外两点观察

- **27B 的可用性是波动的。** 曾观察到 `/api/models` 返回
  `sample_available: False` 且调用报 `404 no available actor`，同一 key 在
  其他时段又完全正常。脚本因此按 `sample_available` 过滤并记录模型目录，
  遇到不可用不会误判为「模型不可用」的永久结论。
- **`_live_model()` 的静默 fallback 有风险。** 真实平台上这会掩盖模型故障；
  本评测端遇错重试 3 次后如实报错，不降级成 stub。
