"""Export local simulation traces to a compact report + SwanLab-style offline JSONL."""
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if (_HERE.parent / "app").exists():
    # 布局一：dev_tools 位于 submission_project 内
    ROOT = _HERE.parent
elif (_HERE.parent.parent / "app").exists():
    # 布局二：.workspace 位于 git 仓库根内（提交仓库布局）
    ROOT = _HERE.parent.parent
else:
    # 布局三：.workspace 与 submission_project 并列
    ROOT = _HERE.parent.parent / "submission_project"
# 数据集根：优先 git 根下 .datasets，回退上级目录
DATA_ROOT = ROOT / ".datasets" if (ROOT / ".datasets").exists() else ROOT.parent

BASELINE = {
    "A": {"macro_f1": 0.7140297220788709},
    "B": {"rmse": 681.1598326043113},
    "C": {"rmse": 52.86857256611151},
}


def main():
    reports = ROOT / "reports" if (ROOT / "reports").exists() else _HERE.parent / "reports"
    summary_path = reports / "summary.json"
    if not summary_path.exists():
        raise SystemExit("run run_10rounds_sim.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = []
    lines = [
        "# 本地 10 轮仿真报告",
        "",
        "所有指标来自本地宿主逐行流式 predict 的实测，不采用训练日志自报分数。",
        "",
        "| 数据集 | 主指标 | 官方基线 | 第 10 轮 | 变化 | 选中产物 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in summary:
        key = item["dataset"]
        name = "macro_f1" if key == "A" else "rmse"
        baseline = item["baseline"][name]
        final = item["final"][name]
        delta = (final - baseline) / max(abs(baseline), 1e-9) * 100
        lines.append("| %s | %s | %.4f | %.4f | %+.2f%% | %s |" % (
            key, name, baseline, final, delta, item["selected_artifact_id"]))
        trace = reports / ("%s_trace.jsonl" % key.lower())
        if trace.exists():
            for raw in trace.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                record = json.loads(raw)
                metrics = record.get("summary", {}).get("metrics", {})
                events.append({"dataset": key, "round": record["round"],
                               "selected_artifact_id": record["selected_artifact_id"],
                               "metrics": metrics})
    if summary and all("round_avg_seconds" in item for item in summary):
        lines.extend(["", "## 单轮耗时审计", "",
                      "| 数据集 | 平均单轮耗时 (s) | 最大单轮耗时 (s) | 总耗时 (s) |",
                      "| --- | ---: | ---: | ---: |"])
        for item in summary:
            lines.append("| %s | %.1f | %.1f | %.1f |" % (
                item["dataset"], item["round_avg_seconds"],
                item["round_max_seconds"], item["seconds"]))
    lines.extend(["", "## 关键机制", "",
                  "- 模型层：父检查点权重/Adam 状态真实继承，新增衍生特征零初始化迁移。",
                  "- 策略层：PILOT 预算档位路由 + SkillApt 条件效用表 + 冷却锁。",
                  "- 门控：样本加权块级 e-process + 冻结基线 + Group-CV OOF 锚点回放（overfitting 治理模式以冻结基线 OOF 为红线）。",
                  "- 预测：逐行无状态前向，本地宿主强制“收到第 1 行才写第 2 行”。", ""])
    (reports / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (reports / "swanlab_offline.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    print("wrote %s and %s" % (reports / "summary.md", reports / "swanlab_offline.jsonl"))


if __name__ == "__main__":
    main()
