<div align=center>
<h3>在资源受限的环境中落地前沿自我演化（RSI）</h3><i>Frontier-Inspired RSI in Resource-Constrained Environments</i>
</div>

## 解压数据集

将三个数据集放入 `.datasets/`：

```
.datasets/cpu-a-guide-20261110-v1/
.datasets/cpu-b-rul-20261110-v1/
.datasets/cpu-c-guide-20261110-v1/
```

## 接手开发

代码位于 `app/`，入口为 `app/harness.py`，验证工具位于 `.workspace/dev_tools/test_submission.py`。

- 比赛要求可见 `.workspace/main.md`，SDK 文档见 `.workspace/contest-harness-sdk-1.0.2/HARNESS_SDK.md`，开发日志及 HANDOFF 见 `.workspace/HANDOFF.md`。
- 数据集见 `.datasets`。**以每个数据集根目录的 `TASK.md` 为准**——`baselines/README.md` 是三包共用的旧模板，其中「B 为三分类」的描述已过时（B 实为 RUL 回归）。

## 构建镜像

```bash
docker build -t rsi-submission:2.0 .
```

## 评测

```bash
export PYTRIO_API_KEY=trio_...
python3 .eval/eval.py          # A/B/C 各 10 轮，结果写入 .eval/runs/
```
