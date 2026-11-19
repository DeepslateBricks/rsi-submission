"""Run full 10-round offline simulations for A/B/C with real streaming predict."""
import argparse
import json
import sys
import time
from pathlib import Path

from local_host import BASELINE_METRICS, LocalHost

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

WORKSPACE = ROOT.parent
DATASETS = {
    "A": "cpu-a-guide-20261110-v1",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="A,B,C")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--mode", choices=["stub", "live"], default="stub")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for key in args.datasets.split(","):
        key = key.strip().upper()
        if key not in DATASETS:
            continue
        started = time.monotonic()
        print("=== dataset %s (%s mode) ===" % (key, args.mode), flush=True)
        host = LocalHost(key, DATA_ROOT / DATASETS[key], rounds=args.rounds, mode=args.mode)
        results = host.run_all()
        elapsed = time.monotonic() - started
        trace_path = output / ("%s_trace.jsonl" % key.lower())
        with trace_path.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        last = results[-1]
        metric = last["summary"]["metrics"]
        baseline = BASELINE_METRICS[key]
        round_seconds = [float(item.get("round_seconds", 0.0)) for item in results]
        summary.append({
            "dataset": key, "baseline": baseline, "final": metric,
            "selected_artifact_id": last["selected_artifact_id"],
            "rounds": len(results), "seconds": round(elapsed, 2),
            "round_avg_seconds": round(sum(round_seconds) / max(1, len(round_seconds)), 2),
            "round_max_seconds": round(max(round_seconds or [0.0]), 2),
            "trace": str(trace_path),
        })
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    header = ["dataset", "metric", "baseline", "final", "selected", "rounds",
              "round_avg_s", "round_max_s", "seconds"]
    print("\n" + "| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for item in summary:
        metric_name = "macro_f1" if item["dataset"] == "A" else "rmse"
        print("| %s | %s | %.4f | %.4f | %s | %d | %.1f | %.1f | %.1f |" % (
            item["dataset"], metric_name, item["baseline"][metric_name],
            item["final"][metric_name], item["selected_artifact_id"],
            item["rounds"], item["round_avg_seconds"], item["round_max_seconds"],
            item["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
