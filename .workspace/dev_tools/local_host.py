"""Local workspace-v2 host stand-in with real candidate/state/predict measurement."""
import base64
import hashlib
import json
import os
import select
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from contest_harness.protocol import PROTOCOL, dump, emit, read_frame

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
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
BASELINE_METRICS = {
    "A": {"macro_f1": 0.7140297220788709, "accuracy": 0.8786173633440515},
    "B": {"rmse": 681.1598326043113, "mae": 563.1058414514886},
    "C": {"rmse": 52.86857256611151, "mae": 38.89982982666355},
}
TASK_TYPES = {"A": "classification", "B": "regression", "C": "regression"}


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def manifest(directory, descriptor):
    files = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        files.append({"path": path.relative_to(directory).as_posix(),
                      "sha256": digest(raw), "size_bytes": len(raw)})
    encoded = json.dumps({"files": files, "artifact": descriptor}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return files, "sha256:" + hashlib.sha256(encoded).hexdigest()


def baseline_artifact(dataset_key, data_dir, artifact_id):
    source = data_dir / "checkpoint.npz"
    staging = Path(os.environ.get("RSI_LOCAL_STAGE", "/tmp")) / ("baseline_" + dataset_key.lower())
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(source, staging / "checkpoint.npz")
    descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV}
    files, sha = manifest(staging, descriptor)
    return {"artifact_id": artifact_id, "artifact": descriptor, "files": files, "artifact_sha256": sha,
            "root": str(staging)}


def _pythonpath():
    sdk_candidates = [
        ROOT / "sdk_src",
        ROOT / ".workspace" / "contest-harness-sdk-1.0.2" / "contest-harness-sdk-1.0.2",
        WORKSPACE / "submission_project" / "sdk_src",
        WORKSPACE / "contest-harness-sdk-1.0.2" / "contest-harness-sdk-1.0.2",
    ]
    sdk = next((path for path in sdk_candidates if (path / "contest_harness").exists()), None)
    if sdk is None:
        raise RuntimeError("SDK source not found; copy it into submission_project/sdk_src")
    return os.pathsep.join([str(ROOT / "app"), str(sdk), os.environ.get("PYTHONPATH", "")])


def _read_line(process, timeout=20):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError("predict did not reply before next sample (streaming protocol violated)")
    return process.stdout.readline()


def measure_candidate(artifact_root, dataset_key, task, validation_path, log_path):
    context = {
        "protocol": "contest.predict-context.v1", "dataset_key": dataset_key, "task": task,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV},
        "artifact_id": "measured",
    }
    checkpoint = Path(artifact_root) / "checkpoint.npz"
    manifest_path = Path(artifact_root) / "model_spec.json"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": _pythonpath(), "CONTEST_PREDICTION_CONTEXT": dump(context),
        "CONTEST_DATASET_KEY": dataset_key, "CONTEST_TASK": task,
        "CONTEST_ARTIFACT_PATH": str(checkpoint),
        "CONTEST_ARTIFACT_MANIFEST": dump({"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV}),
        "CONTEST_WORK": str(Path(os.environ.get("RSI_LOCAL_WORK", "/tmp")) / "predict_work"),
        "RSI_MODEL_ROOT": str(artifact_root),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    Path(env["CONTEST_WORK"]).mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, str(ROOT / "app" / "harness.py"), "predict"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=open(log_path, "w", encoding="utf-8"),
                               text=True, bufsize=1, env=env)
    truths, predictions = [], []
    try:
        with open(validation_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                process.stdin.write(dump({"task": task, "inputs": row["inputs"]}) + "\n")
                process.stdin.flush()
                reply = json.loads(_read_line(process))
                output = reply["outputs"]
                if task == "classification":
                    scores = {item["label"]: float(item["score"]) for item in output}
                    truths.append(row["answer"])
                    predictions.append(max(scores, key=scores.get))
                else:
                    truths.append(float(row["answer"]))
                    predictions.append(float(output["value"]))
        process.stdin.close()
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode != 0:
        raise RuntimeError("predict exit %s; see %s" % (process.returncode, log_path))
    if task == "classification":
        labels = sorted(set(truths))
        vector = [[1.0 if label == predicted else 0.0 for label in labels] for predicted in predictions]
        truth_index = [labels.index(label) for label in truths]
        from rsi_engine.diagnostics import classification_metrics
        metrics = classification_metrics(truth_index, vector, labels)
    else:
        from rsi_engine.diagnostics import regression_metrics
        metrics = regression_metrics(truths, predictions, dataset_key)
    return {"metrics": metrics, "rows": len(truths), "checkpoint_sha256": digest(checkpoint.read_bytes())}


def _stub_model(frame, round_index):
    try:
        payload = json.loads(frame["messages"][-1]["content"])
        candidates = payload.get("candidates", [])
    except Exception:
        candidates = []
    experiments = []
    for position, candidate in enumerate(candidates[:3]):
        params = dict(candidate)
        params["epochs"] = max(2, int(params.get("epochs", 3)) + ((round_index + position) % 2))
        experiments.append({
            "hypothesis": "针对%s漂移验证%s原语的条件边际效用。" % (payload.get("dataset_key", "?"), candidate.get("id")),
            "change_summary": "迁移父权重、续训并融合轻量残差专家。",
            "primitive_id": candidate.get("id"), "hyperparams": params, "seed": 42 + round_index * 17 + position,
        })
    if not experiments:
        return '{"experiments":[]}'
    return dump({"experiments": experiments})


def _load_dotenv(path="~/rsi-workspace/.env"):
    """开发期密钥加载：读 .env 注入环境变量，绝不打印/落盘密钥。

    兼容 `PYTRiO_API_KEY`（大小写混排）与引号包裹的值，统一映射为 PYTRIO_API_KEY。
    """
    target = Path(path).expanduser()
    if not target.exists():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key.lower() == "pytrio_api_key":
            key = "PYTRIO_API_KEY"
        os.environ.setdefault(key, value)


def _llm_settings():
    """LLM 配置：支持 google_studio / openai / pytrio 三类 OpenAI-compatible 提供方。

    密钥与端点一律来自环境变量（不落盘、不进容器/候选）：
    - PYTRIO_API_KEY / RSI_LLM_API_KEY / OPENAI_API_KEY：API 密钥（任一存在即启用 live）
    - RSI_LLM_BASE_URL / OPENAI_BASE_URL：端点；PyTRiO 默认 https://pytrio.com
    - RSI_LLM_MODEL：模型名；pytrio 未指定时自动发现（优先 4B 档）
    - RSI_LLM_PROVIDER：google_studio | openai | pytrio | auto（默认 auto，按变量推断）
    """
    _load_dotenv()
    config_path = Path(__file__).with_name("llm_config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        config = {}
    env_base = os.environ.get("RSI_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    env_model = os.environ.get("RSI_LLM_MODEL")
    config["base_url"] = (env_base or config.get("base_url", "")).rstrip("/")
    config["model"] = env_model or config.get("model", "")
    config["proxy"] = os.environ.get("RSI_LLM_PROXY", config.get("proxy", ""))
    pytrio_key = os.environ.get("PYTRIO_API_KEY", "")
    config["provider"] = os.environ.get("RSI_LLM_PROVIDER") or (
        "pytrio" if pytrio_key else
        "openai" if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")
        else config.get("provider", "google_studio"))
    if config["provider"] == "pytrio":
        # PyTRiO 通道忽略本地 llm_config 的 google 值：端点/模型只认环境变量或自身默认。
        config["base_url"] = (env_base or os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com")).rstrip("/")
        config["model"] = env_model or ""
    return config


def _provider_chat_url(base_url, provider):
    if provider == "pytrio":
        return base_url.rstrip("/") + "/api/openai/v1/chat/completions"
    return base_url.rstrip("/") + "/chat/completions"


_PYTRIO_MODEL_CACHE = {}


def _pytrio_model(config, start):
    """自动发现该密钥可用的模型（优先 4B 档），结果缓存于进程内存。"""
    if config.get("model"):
        return config["model"]
    if _PYTRIO_MODEL_CACHE.get("model"):
        return _PYTRIO_MODEL_CACHE["model"]
    key = os.environ.get("PYTRIO_API_KEY", "")
    url = config["base_url"].rstrip("/") + "/api/models?limit=100&offset=0"
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + key, "X-API-Key": key})
    opener = urllib.request.build_opener()  # PyTRiO 为平台内网关，默认不走本地代理
    with opener.open(request, timeout=int(config.get("timeout_seconds", 120))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = [item.get("model_id") for item in payload.get("items", []) if isinstance(item, dict)]
    # 偏好顺序：27B（主力推理档）> 4B（轻量兜底）；RSI_LLM_MODEL 可显式覆盖。
    for token in ("27B", "4B"):
        selected = next((name for name in names if isinstance(name, str) and token in name), None)
        if selected:
            break
    selected = selected or next((name for name in names if isinstance(name, str)), None)
    if not selected:
        raise RuntimeError("no model available to this key")
    _PYTRIO_MODEL_CACHE["model"] = selected
    print("pytrio model selected: %s" % selected, file=sys.stderr)
    return selected


def _live_model(frame, start):
    """Host-side live model call via an OpenAI-compatible endpoint.
    Keys never enter the container or candidate."""
    config = _llm_settings()
    key = (os.environ.get("PYTRIO_API_KEY") or os.environ.get("RSI_LLM_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or os.environ.get("PYTRIO_API_KEY")
           or os.environ.get("PYTRiO_API_KEY") or "")
    if not key or not config.get("base_url"):
        return _stub_model(frame, start.get("round_index", 1))
    if config["provider"] == "pytrio":
        model_name = _pytrio_model(config, start)
    else:
        model_name = config.get("model") or start.get("model_name")
    # pytrio（27B 思考模型）：尊重协议侧档位，不再被 config 的 1024 截断；
    # 其他提供方保留 config 上限以防意外超量。
    max_tokens = (int(frame.get("max_tokens", 512)) if config["provider"] == "pytrio"
                  else min(int(frame.get("max_tokens", 512)), int(config.get("max_tokens", 1024))))
    body = {
        "model": model_name,
        "messages": frame["messages"],
        "max_tokens": max_tokens,
        "temperature": float(frame.get("temperature", config.get("temperature", 0.1))),
    }
    if "response_format" in frame and config["provider"] != "pytrio":
        body["response_format"] = frame["response_format"]
    # PyTRiO 是平台内网关，默认不走本地代理；google/openai 提供方沿用配置代理。
    use_proxy = config.get("proxy") and config["provider"] != "pytrio"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": config.get("proxy"), "https": config.get("proxy")})
    ) if use_proxy else urllib.request.build_opener()
    request = urllib.request.Request(_provider_chat_url(config["base_url"], config["provider"]),
                                     data=dump(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key,
                                              **({"X-API-Key": key} if config["provider"] == "pytrio" else {})})
    try:
        with opener.open(request, timeout=int(config.get("timeout_seconds", 120))) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        print("live model failed, using stub: %s" % exc, file=sys.stderr)
        return _stub_model(frame, start.get("round_index", 1))


class LocalHost:
    def __init__(self, dataset_key, data_dir, rounds=10, mode="stub", root=None, train_cmd=None,
                 state_path_override=None):
        self.dataset_key = dataset_key
        self.task = TASK_TYPES[dataset_key]
        self.data_dir = Path(data_dir)
        self.rounds = int(rounds)
        self.mode = mode
        # train_cmd 非空时以该命令拉起参与者（如 docker run --network none -u 65534 ...），
        # 用于容器内 1 轮 train 验收；默认仍为本机 python 子进程。
        self.train_cmd = list(train_cmd) if train_cmd else None
        # 容器内 train 时，start 帧的 state_path 必须指向容器内可写目录。
        self.state_path_override = state_path_override
        # 容器内 train 时，start 帧的 public_projection 必须指向容器内可见的数据根目录。
        self.data_root_override = None
        self.root = Path(root or (Path(os.environ.get("RSI_LOCAL_WORK", "/tmp")) / ("local_host_" + dataset_key.lower())))
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.state_dir = self.root / "state"
        self.state_dir.mkdir()
        self.parent_root = self.root / "mount_previous"
        baseline_id = "baseline-" + dataset_key.lower()
        self.parent = baseline_artifact(dataset_key, self.data_dir, baseline_id)
        self.artifact_roots = {baseline_id: self.parent["root"]}
        self.artifact_summaries = {baseline_id: {"metrics": dict(BASELINE_METRICS[dataset_key])}}
        self.artifacts = [{key: value for key, value in self.parent.items() if key != "root"}]
        self.history = []
        self.round_records = []

    def _start_frame(self, index, work):
        return {
            "protocol": PROTOCOL, "prompt": "本地仿真提示：按公开数据与父检查点续训。",
            "prompt_version": "local-sim-v1", "run_id": "local-%s" % self.dataset_key,
            "dataset_key": self.dataset_key, "round_index": int(index),
            "stage_id": "round-%d" % index,
            "task_contract": {
                "task": self.task, "metric": "macro_f1" if self.task == "classification" else "rmse",
                "metric_direction": "maximize" if self.task == "classification" else "minimize",
                "target_unit": "minutes" if self.dataset_key == "B" else ("cycles" if self.dataset_key == "C" else ""),
                "public_projection": {"train_path": str(self.data_root_override or self.data_dir) + "/public/train.jsonl",
                                      "validation_path": str(self.data_root_override or self.data_dir) + "/public/validation.jsonl"},
            },
            "model_name": "local-stub-model", "max_output_tokens": 2048,
            "parent_checkpoint_sha256": self.parent["files"][0]["sha256"],
            "parent_artifact": {key: value for key, value in self.parent.items() if key != "root"},
            "artifacts": self.artifacts, "wall_seconds": 1800,
            "resource_limits": {"candidate_bytes": 64 * 1024 * 1024, "state_bytes": 8 * 1024 * 1024},
            "state_path": self.state_path_override or str(self.state_dir),
            "state_sha256": self._state_sha(),
        }

    def _state_sha(self):
        digest_value = hashlib.sha256()
        for path in sorted(self.state_dir.rglob("*")):
            if path.is_file():
                digest_value.update(path.relative_to(self.state_dir).as_posix().encode())
                digest_value.update(path.read_bytes())
        return "sha256:" + digest_value.hexdigest()

    def run_round(self, index):
        round_started = time.monotonic()
        work = self.root / ("round_%02d" % index)
        work.mkdir()
        if self.parent_root.exists():
            shutil.rmtree(self.parent_root)
        self.parent_root.mkdir()
        parent_source = Path(self.artifact_roots[self.parent["artifact_id"]])
        for item in self.parent["files"]:
            target = self.parent_root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_source / item["path"], target)
        env = os.environ.copy()
        env.update({"PYTHONPATH": _pythonpath(), "CONTEST_WORK": str(work),
                    "RSI_PARENT_ROOT": str(self.parent_root),
                    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        log_path = work / "harness.log"
        if self.train_cmd:
            cmd = list(self.train_cmd)
        else:
            cmd = [sys.executable, str(ROOT / "app" / "harness.py"), "train"]
        process = subprocess.Popen(cmd,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=open(log_path, "w", encoding="utf-8"),
                                   text=True, bufsize=1, env=env)
        start = self._start_frame(index, work)
        process.stdin.write(dump(start) + "\n")
        process.stdin.flush()
        sequence = 0
        record = None
        response_number = 0
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("participant closed stdout; see " + str(log_path))
                frame = json.loads(line)
                kind = frame.get("type")
                if kind == "round_failed":
                    raise RuntimeError("participant failed: %s; see %s" % (frame.get("error"), log_path))
                if kind == "round_complete":
                    record = frame["record"]
                    process.stdin.write(dump({"type": "round_ack", "accepted": True}) + "\n")
                    process.stdin.flush()
                    process.stdin.close()
                    break
                if frame.get("id") != sequence:
                    raise RuntimeError("rpc sequence mismatch")
                sequence += 1
                if kind == "history_request":
                    reply = {"type": "history_response", "id": frame["id"], "records": self.history}
                elif kind == "model_request":
                    response_number += 1
                    content = (_live_model(frame, start) if self.mode == "live" else _stub_model(frame, index))
                    reply = {
                        "type": "model_response", "id": frame["id"],
                        "call_id": "call-%s-%d-%d" % (self.dataset_key, index, response_number),
                        "response": {
                            "id": "resp-%d" % response_number, "object": "chat.completion",
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                         "finish_reason": "stop"}],
                            "usage": {"completion_tokens": min(256, len(content) // 3), "total_tokens": len(content) // 2},
                        },
                    }
                elif kind == "state_request":
                    incoming = self.root / ("incoming_state_%02d" % index)
                    if incoming.exists():
                        shutil.rmtree(incoming)
                    incoming.mkdir()
                    for item in frame.get("files", []):
                        target = incoming / item["path"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(base64.b64decode(item["data_base64"], validate=True))
                    if self.state_dir.exists():
                        shutil.rmtree(self.state_dir)
                    shutil.move(str(incoming), str(self.state_dir))
                    reply = {"type": "state_response", "id": frame["id"], "state_sha256": self._state_sha()}
                elif kind == "candidate_request":
                    artifact_id = "cand-%s-%02d" % (self.dataset_key.lower(), index)
                    destination = self.root / ("artifact_%02d" % index)
                    if destination.exists():
                        shutil.rmtree(destination)
                    destination.mkdir()
                    for item in frame["files"]:
                        target = destination / item["path"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(base64.b64decode(item["data_base64"], validate=True))
                    descriptor = frame["artifact"]
                    files, sha = manifest(destination, descriptor)
                    receipt = {"artifact_id": artifact_id, "artifact": descriptor,
                               "files": files, "artifact_sha256": sha, "root": str(destination)}
                    summary = measure_candidate(destination, self.dataset_key, self.task,
                                                self.data_dir / "public" / "validation.jsonl",
                                                work / ("predict_%02d.log" % index))
                    reply = {"type": "candidate_response", "id": frame["id"],
                             "artifact_id": artifact_id, "summary": summary}
                    self.artifacts.append({key: value for key, value in receipt.items() if key != "root"})
                    self.artifact_roots[artifact_id] = str(destination)
                    self.artifact_summaries[artifact_id] = summary
                    self.round_records.append({"round": index, "artifact_id": artifact_id, "summary": summary})
                elif kind == "artifact_read_request":
                    artifact_id = frame["artifact_id"]
                    match = next((item for item in self.artifacts if item["artifact_id"] == artifact_id), None)
                    if match is None:
                        raise RuntimeError("unknown artifact " + str(artifact_id))
                    artifact_root = Path(self.artifact_roots.get(artifact_id, match.get("root") or self.root / ("artifact_%02d" % index)))
                    raw = (artifact_root / frame["path"]).read_bytes()
                    offset = int(frame["offset"])
                    length = int(frame["length"])
                    chunk = raw[offset:offset + length]
                    reply = {"type": "artifact_read_response", "id": frame["id"],
                             "data_base64": base64.b64encode(chunk).decode(),
                             "eof": offset + len(chunk) == len(raw)}
                else:
                    raise RuntimeError("unexpected frame " + str(kind))
                process.stdin.write(dump(reply) + "\n")
                process.stdin.flush()
            process.wait(timeout=60)
        finally:
            if process.poll() is None:
                process.kill()
        if process.returncode != 0:
            raise RuntimeError("train exit %s; see %s" % (process.returncode, log_path))
        selected = record["selected_artifact_id"]
        chosen = next((item for item in self.artifacts if item["artifact_id"] == selected), None)
        if chosen is None:
            raise RuntimeError("selected artifact unknown: " + selected)
        selected_summary = self.artifact_summaries.get(selected, {"metrics": {}})
        self.history.append({"id": "history-%s-%02d" % (self.dataset_key.lower(), index),
                             "stage_id": "round-%d" % index, "selected_artifact_id": selected,
                             "metrics": selected_summary.get("metrics", {})})
        self.parent = chosen
        return {"round": index, "selected_artifact_id": selected, "record": record,
                "summary": selected_summary, "log": str(log_path), "state_sha256": self._state_sha(),
                "round_seconds": round(time.monotonic() - round_started, 2)}

    def run_all(self):
        results = []
        for index in range(1, self.rounds + 1):
            result = self.run_round(index)
            results.append(result)
            metrics = result.get("summary", {}).get("metrics", {})
            print("round %02d selected=%s seconds=%.1f metrics=%s" % (
                index, result["selected_artifact_id"], result.get("round_seconds", 0.0),
                json.dumps(metrics, ensure_ascii=False)), flush=True)
        return results
