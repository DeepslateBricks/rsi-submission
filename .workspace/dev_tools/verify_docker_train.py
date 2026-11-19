"""1-round train acceptance INSIDE the container: --network none + uid 65534 + read-only parent.

Usage: python3 dev_tools/verify_docker_train.py --image rsi-submission:2.0 --dataset A
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_host import LocalHost

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
DATA = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="rsi-submission:2.0")
    parser.add_argument("--dataset", default="A")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    key = args.dataset.upper()
    host = LocalHost(key, DATA_ROOT / DATA[key], rounds=args.rounds, root="/tmp/rsi_docker_train_" + key.lower())
    # 容器内可写 state 目录（uid 65534 需写权限）
    state_host_dir = host.root / "state_mount"
    state_host_dir.mkdir(parents=True, exist_ok=True)
    state_host_dir.chmod(0o777)
    host.state_path_override = "/workspace/state"
    host.data_root_override = "/data"
    host.train_cmd = [
        "docker", "run", "--rm", "--network", "none", "-i",
        "-u", "65534:65534",
        "-e", "CONTEST_WORK=/workspace",
        "-e", "RSI_PARENT_ROOT=/workspace/previous",
        "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "-v", "%s:/data:ro" % (DATA_ROOT / DATA[key]),
        "-v", "%s:/workspace/previous:ro" % host.parent_root,
        "-v", "%s:/workspace/state" % state_host_dir,
        args.image, "python3", "/app/harness.py", "train",
    ]
    results = host.run_all()
    for item in results:
        metrics = item.get("summary", {}).get("metrics", {})
        print("container train round %d selected=%s metrics=%s" % (
            item["round"], item["selected_artifact_id"],
            json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                        if k in ("macro_f1", "rmse", "accuracy", "mae")}, ensure_ascii=False)))
    print("docker train acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
