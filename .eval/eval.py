#!/usr/bin/env python3
"""Virtual contest host for an independent evaluation of this submission.

This script pretends to be the platform. It speaks the real wire protocol
(``contest.harness.workspace.v2``) to the submission running inside the
submitted Docker image, answers its RPC frames, forwards its model requests to
a real LLM through the PyTRiO-compatible endpoint, measures each submitted
candidate on the *public validation split*, and decides accept/rollback on its
own -- never trusting numbers reported by the submission.

Layout produced next to this file::

    .eval/
      eval.py                  this host
      runs/<timestamp>/        one directory per evaluation run
        events.jsonl           full protocol/decision trace
        report.md              human readable result
        report.json            machine readable result
        staging/               /data, /previous, /model mounts
        state/                 round-to-round native state

Usage::

    export PYTRIO_API_KEY=...
    python3 .eval/eval.py --model <model-name> [--datasets A B C] [--rounds 3]

The API key is read from ``PYTRIO_API_KEY`` only; it is never written to disk
or passed into the container. The model name may also come from
``PYTRIO_MODEL``; the endpoint from ``PYTRIO_BASE_URL``.

Only ``numpy`` and ``requests`` are required on the host.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import select
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard requirement
    sys.stderr.write("eval.py requires numpy on the host\n")
    raise

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROTOCOL = "contest.harness.workspace.v2"
PREDICT_CONTEXT_PROTOCOL = "contest.predict-context.v1"
PROMPT_VERSION = "industrial-workspace-flexible-v3"
SDK_VERSION = "1.0.2"

# The contest permits PyTRiO only. This is the single gateway the host talks to.
PYTRIO_BASE_URL = "https://pytrio.com"

# The platform runs 10 optimisation rounds per dataset before hidden testing
# (main.md: 每个数据集进行 10 轮优化). Fewer rounds cannot show the
# "read history -> experiment -> decide" loop the contest requires (>=3 times)
# and cannot expose multi-round drift/rollback behaviour.
CONTEST_ROUNDS = 10

# A single dropped connection must not fail a whole round: platforms retry the
# gateway. Only exhausted retries surface as an error.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0
MIN_DECISION_ROUNDS = 3

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_DATA_ROOT = REPO / ".datasets"
DEFAULT_IMAGE = "rsi-submission:2.0"
CONTAINER_WORK = "/workspace"
CONTAINER_APP = "/app"

DATA_PACKAGES = {
    "A": "cpu-a-guide-20261110-v1",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}
TASK_TYPES = {"A": "classification", "B": "regression", "C": "regression"}
TARGET_UNITS = {"A": "class", "B": "minute", "C": "cycle"}
PRIMARY_METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
METRIC_DIRECTION = {"A": "higher", "B": "lower", "C": "lower"}

REQUIRED_INPUTS = {"id", "inputs", "answer"}
FILE_LIMIT = 64
BYTE_LIMIT = 64 * 1024 * 1024
TEXT_LIMIT = 4000
CONFIG_LIMIT = 65536
# Platform-granted ceiling for one call. Qwen3.5 is a reasoning model: it writes
# a ``reasoning`` trace before ``content``, and traces of 14k characters were
# observed, so a tight ceiling makes ``content`` come back null with
# finish_reason=length. The submission chooses its own per-call budget within
# this ceiling; the number is deliberately generous.
MAX_OUTPUT_TOKENS = 16384

SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]{1,180}$")
FORBIDDEN_INPUT_FIELDS = {
    "truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_relative(path: str) -> bool:
    if not isinstance(path, str) or not SAFE_PATH.match(path) or path.startswith("/"):
        return False
    parts = path.split("/")
    return len(parts) <= 8 and all(part not in ("", ".", "..") for part in parts)


def _looks_like_json_object(content) -> bool:
    """True when the reply body carries a JSON object the planner can parse."""
    if not isinstance(content, str):
        return False
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        return isinstance(json.loads(text[start:end + 1]), dict)
    except (TypeError, ValueError):
        return False


class EvaluatorError(RuntimeError):
    """A failure that should be reported, not crash the whole run."""


# --------------------------------------------------------------------------- #
# Effective evaluation context
# --------------------------------------------------------------------------- #

class RunLog:
    """Append-only JSONL trace of every protocol frame and host decision."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stream = self.path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields):
        record = {"ts": round(time.time(), 3), "kind": kind, **fields}
        with self._lock:
            self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()

    def close(self):
        self._stream.close()


# --------------------------------------------------------------------------- #
# LLM client (PyTRiO compatible, OpenAI chat-completions shape)
# --------------------------------------------------------------------------- #

class LLMClient:
    """Model gateway client. The contest allows **PyTRiO only**.

    PyTRiO is the platform's own gateway, not a generic OpenAI-compatible SaaS:

    - base ``https://pytrio.com``, chat path ``/api/openai/v1/chat/completions``
      (note the ``/api/openai/v1`` segment; ``<base>/chat/completions`` is wrong)
    - every request carries both ``Authorization: Bearer <key>`` and
      ``X-API-Key: <key>``; Bearer alone returns 401
    - model ids come from ``GET /api/models``; ``Qwen/Qwen3.5-4B`` is a
      reasoning model whose ``message.content`` is ``null`` when the token
      budget is consumed by ``reasoning``

    ``use_stub`` keeps the protocol path exercisable offline without a key.
    """

    def __init__(self, model: str, base_url: str, api_key: str | None, *,
                 use_stub: bool, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.use_stub = use_stub
        self.timeout = timeout
        self.calls = 0
        self.failures = 0
        self.total_tokens = 0
        self.empty_content = 0
        self.unusable_content = 0
        self.replies: list[dict] = []
        self._explicit_model = model
        self.model = model or ""
        self._lock = threading.Lock()

    # -- endpoint / auth ---------------------------------------------------- #
    @property
    def chat_url(self) -> str:
        return self.base_url + "/api/openai/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url + "/api/models?limit=100&offset=0"

    def _session(self):
        """A proxyless session: PyTRiO is an in-platform gateway.

        ``requests`` honours ``http_proxy``/``https_proxy`` from the environment,
        which makes calls to PyTRiO hang or 401 behind a local proxy. ``trust_env
        = False`` bypasses proxy, CA bundle and netrc discovery entirely.
        """
        session = requests.Session()
        session.trust_env = False
        return session

    def _headers(self) -> dict:
        # PyTRiO needs both headers; Bearer alone is rejected with 401.
        return {"Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
                "X-API-Key": "%s" % self.api_key}

    def discover_model(self) -> str:
        """Resolve the model id this key may actually use."""
        if self._explicit_model:
            self.model = self._explicit_model
            return self.model
        if requests is None:
            raise EvaluatorError("requests is required to reach PyTRiO")
        try:
            response = self._session().get(self.models_url, headers=self._headers(),
                                           timeout=self.timeout)
        except Exception as exc:
            raise EvaluatorError("PyTRiO /api/models unreachable: %s" % exc) from exc
        if response.status_code != 200:
            raise EvaluatorError("PyTRiO /api/models returned %d: %s"
                                 % (response.status_code, response.text[:300]))
        items = [item for item in ((response.json() or {}).get("items") or [])
                 if isinstance(item, dict) and isinstance(item.get("model_id"), str)]
        if not items:
            raise EvaluatorError("PyTRiO returned no models for this key")
        # The gateway advertises per-key availability. A model listed with
        # sample_available=False answers every request with
        # 404 "no available actor", so it must not be selected even though it
        # appears in the catalogue.
        available = [item for item in items if item.get("sample_available") is not False]
        self.model_catalogue = [
            {"model_id": item.get("model_id"),
             "sample_available": item.get("sample_available"),
             "train_available": item.get("train_available")}
            for item in items]
        if not available:
            raise EvaluatorError(
                "no PyTRiO model is currently sample_available for this key: %s"
                % json.dumps(self.model_catalogue, ensure_ascii=False))
        names = [item["model_id"] for item in available]
        # Prefer the smallest capable tier: 4B answers in ~1s, while the 27B
        # tier has shown multi-minute stalls (and is often unavailable).
        # --model overrides this entirely.
        for token in ("4B", "27B"):
            chosen = next((name for name in names if token in name), None)
            if chosen:
                self.model = chosen
                return chosen
        self.model = names[0]
        return self.model

    # -- stub -------------------------------------------------------------- #
    @staticmethod
    def _stub_body(messages) -> dict:
        """Return a syntactically valid planner response without a model.

        The submission asks for ``{"experiments": [...]}`` with primitive ids
        taken from the candidate list it sends. We echo a generic shape that
        its sanitizer will clip to its own templates; ids it does not know are
        dropped and its deterministic fallback takes over, which is exactly
        the degraded path we want to observe offline.
        """
        ids = []
        for message in messages:
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except (TypeError, ValueError):
                continue
            for candidate in payload.get("candidates") or []:
                if isinstance(candidate, dict) and isinstance(candidate.get("id"), str):
                    ids.append(candidate["id"])
        ids = ids[:2] or ["unknown"]
        body = {"experiments": [
            {
                "hypothesis": "stub: 验证 %s 在当前漂移诊断下的条件效用" % name,
                "change_summary": "stub: 继承父检查点，按候选原语受控微调",
                "primitive_id": name,
                "hyperparams": {"epochs": 3},
                "seed": 42 + index,
            }
            for index, name in enumerate(ids)
        ]}
        return {
            "id": "chatcmpl-stub-%d" % random.randrange(1 << 30),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "stub-offline",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(body)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # -- real call --------------------------------------------------------- #
    def complete(self, messages, options: dict) -> tuple[str, dict]:
        """Return ``(call_id, response_dict)`` for one model request."""
        with self._lock:
            self.calls += 1
            call_index = self.calls
        if self.use_stub:
            return "call-stub-%d" % call_index, self._stub_body(messages)

        if requests is None:
            raise EvaluatorError("requests is required to call a real model")
        if not self.api_key:
            raise EvaluatorError("PYTRIO_API_KEY is not set")

        payload = {"model": self.model, "messages": messages}
        for key in ("max_tokens", "temperature", "top_p", "stop",
                    "frequency_penalty", "presence_penalty", "seed", "n",
                    "response_format", "tool_choice", "tools",
                    "parallel_tool_calls", "enable_thinking"):
            if key in options and options[key] is not None:
                payload[key] = options[key]
        if "max_tokens" not in payload:
            payload["max_tokens"] = MAX_OUTPUT_TOKENS

        response = None
        last_error = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = self._session().post(self.chat_url, json=payload,
                                                headers=self._headers(),
                                                timeout=self.timeout)
            except Exception as exc:  # transient gateway hiccup
                last_error = exc
                self.failures += 1
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                raise EvaluatorError(
                    "model request failed after %d attempts: %s"
                    % (RETRY_ATTEMPTS, exc)) from exc
            # 5xx and 429 are worth retrying; 4xx (bad key/request) is not.
            if response.status_code >= 500 or response.status_code == 429:
                last_error = "HTTP %d" % response.status_code
                self.failures += 1
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
            break
        if response is None:
            raise EvaluatorError("model request produced no response: %s" % last_error)
        if response.status_code != 200:
            self.failures += 1
            detail = response.text[:400]
            if response.status_code == 401:
                detail += ("  [PyTRiO needs both Authorization: Bearer and "
                           "X-API-Key, and the key must be a trio_ key from "
                           "https://pytrio.com — not a third-party API key]")
            raise EvaluatorError("model endpoint returned %d: %s"
                                 % (response.status_code, detail))
        try:
            body = response.json()
        except ValueError as exc:
            self.failures += 1
            raise EvaluatorError("model endpoint returned non-JSON") from exc
        usage = body.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            with self._lock:
                self.total_tokens += usage["total_tokens"]

        # Reasoning models put the answer in ``content`` and the trace in
        # ``reasoning``. Measured on Qwen3.8-27B with a realistic diagnostic
        # prompt: max_tokens=1024 spends all 1024 tokens on reasoning and
        # returns content=null; 2048 returns truncated content; 4096 is the
        # first budget that yields parseable JSON. The submission picks its own
        # budget, so a starved budget shows up here as an empty/partial answer
        # that silently degrades its planner to templates. Record it precisely
        # instead of hiding it.
        try:
            message = body["choices"][0]["message"]
            content = message.get("content")
            finish = body["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            message, content, finish = {}, None, None
        details = (usage.get("completion_tokens_details") or {})
        reasoning_tokens = details.get("reasoning_tokens")
        reply = {
            "requested_max_tokens": payload.get("max_tokens"),
            "finish_reason": finish,
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "content_chars": len(content or ""),
            "content_empty": not (isinstance(content, str) and content.strip()),
        }
        # Did the participant actually receive something a planner can use?
        reply["content_parsable_json"] = _looks_like_json_object(content)
        with self._lock:
            self.replies.append(reply)
            if reply["content_empty"]:
                self.empty_content += 1
            if not reply["content_parsable_json"]:
                self.unusable_content += 1
        body["_eval_reply"] = reply
        if not reply["content_parsable_json"]:
            sink = getattr(self, "log", None)
            if sink is not None:
                sink.write("model_unusable_content", **reply)
        call_id = str(body.get("id") or "call-%d" % call_index)
        return call_id, body


# --------------------------------------------------------------------------- #
# Public data: read once, measure independently
# --------------------------------------------------------------------------- #

def flatten_inputs(value, depth=0):
    """Recursively flatten ``inputs`` into ``[(path, value), ...]``.

    Mirrors the schema discipline the platform expects: numeric scalars only,
    deterministic path order, and rejection of target-derived field names.
    """
    if depth > 16:
        raise ValueError("input nesting exceeds limit")
    paths, values = [], []

    def visit(node, path):
        if isinstance(node, dict):
            for key in sorted(node):
                if not isinstance(key, str):
                    raise ValueError("input keys must be strings")
                if re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUT_FIELDS:
                    raise ValueError("target-derived input field %r" % key)
                visit(node[key], path + [["key", key]])
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, path + [["index", index]])
        elif isinstance(node, bool) or not isinstance(node, (int, float)):
            raise ValueError("inputs must be finite numeric features")
        elif not math.isfinite(float(node)):
            raise ValueError("inputs must be finite numeric features")
        else:
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(node))

    visit(value, [])
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


class PublicSplit:
    """One dataset's public train/validation projection, held by the host."""

    def __init__(self, dataset_key: str, package_dir: Path):
        self.dataset_key = dataset_key
        self.package = package_dir
        self.task = TASK_TYPES[dataset_key]
        self.train_path = package_dir / "public" / "train.jsonl"
        self.validation_path = package_dir / "public" / "validation.jsonl"
        if not self.train_path.is_file() or not self.validation_path.is_file():
            raise EvaluatorError("missing public split under %s" % package_dir)
        self.train_rows = self._read(self.train_path)
        self.validation_rows = self._read(self.validation_path)
        self.classes = self._classes()
        self.train_id_index = {row["id"]: row for row in self.train_rows}

    @staticmethod
    def _read(path: Path) -> list[dict]:
        rows, seen = [], set()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not REQUIRED_INPUTS <= set(row):
                raise EvaluatorError("%s:%d missing id/inputs/answer" % (path.name, number))
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
                raise EvaluatorError("%s:%d duplicate or empty id" % (path.name, number))
            seen.add(row["id"])
            rows.append(row)
        if not rows:
            raise EvaluatorError("%s is empty" % path.name)
        return rows

    def _classes(self) -> list[str]:
        if self.task != "classification":
            return []
        names = sorted({row["answer"] for row in self.validation_rows})
        # Gas sensor task order is fixed by the platform contract, keep it.
        preferred = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
        return [name for name in preferred if name in names] or names


# --------------------------------------------------------------------------- #
# Metrics -- computed by the host, never taken from the submission
# --------------------------------------------------------------------------- #

def classification_metrics(truth, predicted_labels, classes):
    labels = np.asarray(predicted_labels, dtype=int)
    truth = np.asarray(truth, dtype=int)
    f1 = []
    for index in range(len(classes)):
        tp = int(((labels == index) & (truth == index)).sum())
        fp = int(((labels == index) & (truth != index)).sum())
        fn = int(((labels != index) & (truth == index)).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    confusion = defaultdict(int)
    for expected, actual in zip(truth.tolist(), labels.tolist()):
        if expected != actual and 0 <= expected < len(classes) and 0 <= actual < len(classes):
            confusion["%s->%s" % (classes[expected], classes[actual])] += 1
    return {
        "macro_f1": float(np.mean(f1)) if f1 else 0.0,
        "accuracy": float(np.mean(labels == truth)),
        "per_class_f1": {name: round(float(value), 4) for name, value in zip(classes, f1)},
        "top_confusions": sorted(confusion.items(), key=lambda item: -item[1])[:3],
    }


def regression_metrics(truth, predicted, dataset_key):
    truth = np.asarray(truth, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    diff = predicted - truth
    nasa = 0.0
    if dataset_key == "C":
        clipped = np.clip(diff, -100.0, 100.0)
        nasa = float(np.sum(np.where(clipped < 0, np.exp(-clipped / 13.0) - 1.0,
                                     np.exp(clipped / 10.0) - 1.0)))
    result = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "overestimation_rate": float(np.mean(diff > 0)),
        "nasa_score": nasa,
    }
    if dataset_key == "B":
        early = truth > 300
        late = truth <= 100
        result["early_rmse"] = float(np.sqrt(np.mean(diff[early] ** 2))) if early.any() else 0.0
        result["late_rmse"] = float(np.sqrt(np.mean(diff[late] ** 2))) if late.any() else 0.0
    return result


def scores_from_predictions(split: PublicSplit, predictions: list[dict],
                            rows: list[dict] | None = None) -> dict:
    """Turn raw prediction records into metrics on the validation split.

    ``rows`` allows scoring a prefix of the split (a smoke-test budget) without
    pretending the whole split was measured.
    """
    rows = split.validation_rows if rows is None else rows
    if len(predictions) != len(rows):
        raise EvaluatorError("prediction count %d != validation rows %d"
                             % (len(predictions), len(rows)))
    if split.task == "classification":
        index = {name: position for position, name in enumerate(split.classes)}
        labels = []
        for record in predictions:
            outputs = record.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise EvaluatorError("classification output must be a non-empty list")
            best_label, best_score = None, -math.inf
            for item in outputs:
                if not isinstance(item, dict):
                    raise EvaluatorError("classification items must be objects")
                label, score = item.get("label"), item.get("score")
                if not isinstance(label, str) or label not in index:
                    raise EvaluatorError("unknown class label %r" % (label,))
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise EvaluatorError("score must be a number")
                if not 0.0 <= float(score) <= 1.0:
                    raise EvaluatorError("score out of [0,1]")
                if float(score) > best_score:
                    best_label, best_score = label, float(score)
            labels.append(index[best_label])
        truth = [index[row["answer"]] for row in rows]
        return classification_metrics(truth, labels, split.classes)

    values = []
    for record in predictions:
        outputs = record.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {"value"}:
            raise EvaluatorError("regression output must be {'value': n}")
        value = outputs["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluatorError("regression value must be a number")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise EvaluatorError("regression value must be finite and non-negative")
        values.append(float(value))
    truth = [float(row["answer"]) for row in rows]
    return regression_metrics(truth, values, split.dataset_key)


# --------------------------------------------------------------------------- #
# Docker sandbox: the submission runs in the image the platform would use
# --------------------------------------------------------------------------- #

class Sandbox:
    """Runs the submission image with the platform's documented mounts.

    The real platform mounts ``/app`` (read-only), ``/data`` (public split),
    ``/previous`` (parent artifact), ``/state`` (previous native state),
    ``/model`` (selected artifact, predict only) and a writable ``/workspace``.
    We reproduce that with bind mounts, which the host can create even where
    ``/`` itself is not writable.

    ``repair_libgomp`` is reported in the run summary: the shipped image can
    miss ``libgomp.so.1`` while its own dependency install pulls a copy inside
    ``scikit_learn.libs``. Repairing it here makes the submission *runnable*
    but is a deviation from a clean-image evaluation and is flagged as such.
    """

    def __init__(self, image: str, run_dir: Path, log: RunLog, *,
                 repair_libgomp: bool = True):
        self.image = image
        self.run_dir = run_dir
        self.log = log
        self.repair_libgomp = repair_libgomp
        self.staging = run_dir / "staging"
        self.workspace = run_dir / "workspace"
        self.state_dir = run_dir / "state"
        for path in (self.staging / "data", self.staging / "previous",
                     self.staging / "model", self.workspace, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)
        # The image runs as uid 65534; the bind-mounted workspace must accept it.
        os.chmod(self.workspace, 0o777)
        self._libgomp = self._prepare_libgomp() if repair_libgomp else None

    # -- capability probes -------------------------------------------------- #
    def docker_available(self) -> bool:
        try:
            probe = subprocess.run(["docker", "image", "inspect", self.image],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=30)
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _prepare_libgomp(self) -> Path | None:
        """Extract the OpenMP runtime the image itself already ships."""
        target = self.run_dir / "runtime" / "libgomp.so.1"
        target.parent.mkdir(parents=True, exist_ok=True)
        # The image runs as uid 65534 and must be able to write the extraction
        # target through the bind mount.
        os.chmod(target.parent, 0o777)
        # Probe first: a correctly built image already ships libgomp1 and needs
        # no repair. Only a genuinely broken image gets the extracted copy, so
        # the report's ``libgomp_repair`` flag means what it says.
        script = (
            "python3 - <<'EOF'\n"
            "import ctypes, glob, shutil, sys\n"
            "try:\n"
            "    ctypes.CDLL('libgomp.so.1')\n"
            "    print('PRESENT'); sys.exit(0)\n"
            "except OSError:\n"
            "    pass\n"
            "hits = sorted(glob.glob('/usr/local/lib/python3.12/site-packages/**/libgomp*.so*',\n"
            "                        recursive=True))\n"
            "if not hits:\n"
            "    print('MISSING'); sys.exit(0)\n"
            "shutil.copyfile(hits[0], '/out/libgomp.so.1')\n"
            "print('REPAIRED', hits[0])\n"
            "EOF")
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "sh",
                 "-v", "%s:/out" % target.parent, self.image, "-c", script],
                capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError):
            return None
        output = proc.stdout or ""
        if "PRESENT" in output:
            # The image is fine; record that no repair was needed.
            return None
        if "REPAIRED" in output and target.is_file():
            return target
        return None

    # -- mount plan --------------------------------------------------------- #
    def mount_args(self, *, with_model: bool, with_state: bool) -> list[str]:
        args = [
            "-v", "%s:%s" % (self.staging / "data", "/data:ro"),
            "-v", "%s:/previous:ro" % (self.staging / "previous"),
            "-v", "%s:%s" % (self.workspace, CONTAINER_WORK),
        ]
        if with_state:
            args += ["-v", "%s:/state:ro" % self.state_dir]
        if with_model:
            args += ["-v", "%s:/model:ro" % (self.staging / "model")]
        if self._libgomp is not None:
            args += ["-v", "%s:/usr/local/lib/libgomp.so.1:ro" % self._libgomp]
            args += ["-e", "LD_LIBRARY_PATH=/usr/local/lib"]
        return args

    def write_data(self, split: PublicSplit):
        """Expose the public split at ``/data`` exactly like the platform."""
        for target, source in ((self.staging / "data" / "train.jsonl", split.train_path),
                               (self.staging / "data" / "validation.jsonl",
                                split.validation_path)):
            shutil.copyfile(source, target)

    def describe(self) -> dict:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--network", "none",
             "-e", "PYTHONDONTWRITEBYTECODE=1",
             "--entrypoint", "python3", self.image,
             "/app/harness.py", "describe"],
            capture_output=True, text=True, timeout=300)
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        record = None
        if lines:
            try:
                record = json.loads(lines[-1])
            except ValueError:
                record = None
        self.log.write("describe", exit=proc.returncode, stdout=proc.stdout[-4000:],
                       stderr=proc.stderr[-4000:])
        return {"exit": proc.returncode, "record": record,
                "stdout": proc.stdout, "stderr": proc.stderr}


# --------------------------------------------------------------------------- #
# The host: drives one train round over stdin/stdout JSONL
# --------------------------------------------------------------------------- #

class FrameError(EvaluatorError):
    pass


class VirtualHost:
    """One training round: send ``start``, answer RPCs, measure candidates."""

    def __init__(self, *, sandbox: Sandbox, split: PublicSplit, llm: LLMClient,
                 log: RunLog, run_id: str, round_index: int, prompt: str,
                 model_name: str, wall_seconds: float,
                 parent_receipt: dict, artifacts: list[dict],
                 state_sha256: str, state_present: bool,
                 timeout: float, baseline_metrics: dict | None = None,
                 predict_read_timeout: float = 60.0,
                 rounds_per_dataset: int = CONTEST_ROUNDS):
        self.sandbox = sandbox
        self.split = split
        self.llm = llm
        self.log = log
        self.run_id = run_id
        self.round_index = round_index
        self.prompt = prompt
        self.model_name = model_name
        self.wall_seconds = wall_seconds
        self.parent_receipt = parent_receipt
        self.artifacts = {item["artifact_id"]: item for item in artifacts}
        self.state_sha256 = state_sha256
        self.state_present = state_present
        self.timeout = timeout
        # Per-sample ceiling: the platform treats a stalled prediction line as a
        # failed measurement rather than waiting for EOF.
        self.predict_read_timeout = predict_read_timeout
        self.rounds_per_dataset = rounds_per_dataset
        self.baseline_metrics = baseline_metrics

        self.candidates: dict[str, dict] = {}      # artifact_id -> receipt + bytes
        # Carried across rounds: a round may validly select an artifact that was
        # produced earlier, so its bytes must outlive this VirtualHost instance.
        self.artifact_store: dict[str, dict] = {}
        self.measured: dict[str, dict] = {}
        self.model_calls: list[dict] = []
        self.history: list[dict] = []
        self.errors: list[str] = []
        self.prediction_runs: list[dict] = []
        self.decision: dict | None = None
        self.sequence = 0

    # -- start frame -------------------------------------------------------- #
    def start_frame(self) -> dict:
        projection = {
            "train_path": "/data/train.jsonl",
            "validation_path": "/data/validation.jsonl",
            "row_fields": ["id", "inputs", "answer"],
            "source_metadata": False,
        }
        return {
            "protocol": PROTOCOL,
            "run_id": self.run_id,
            "dataset_key": self.split.dataset_key,
            "round_index": self.round_index,
            "stage_id": "independent-eval",
            "prompt": self.prompt,
            "prompt_version": PROMPT_VERSION,
            "rounds_per_dataset": self.rounds_per_dataset,
            "parent_checkpoint_sha256": self.parent_receipt.get("artifact_sha256", ""),
            "task_contract": {
                "dataset_key": self.split.dataset_key,
                "task": self.split.task,
                "metric": PRIMARY_METRIC[self.split.dataset_key],
                "metric_direction": METRIC_DIRECTION[self.split.dataset_key],
                "target_unit": TARGET_UNITS[self.split.dataset_key],
                "public_projection": projection,
            },
            "model_name": self.model_name,
            "parent_artifact": self.parent_receipt,
            "artifacts": list(self.artifacts.values()),
            "state_sha256": self.state_sha256,
            "state_path": "/state" if self.state_present else None,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "resource_limits": {
                "candidate_upload_bytes_per_round": BYTE_LIMIT,
                "candidate_files_per_round": FILE_LIMIT,
                "hypothesis_bytes": TEXT_LIMIT,
                "configuration_bytes": CONFIG_LIMIT,
            },
            "current_validation": {},
            "wall_seconds": self.wall_seconds,
        }

    # -- prompt binding ----------------------------------------------------- #
    def bind_prompt(self, messages: list) -> list:
        """Prepend the official prompt (host duty), keep participant's system."""
        metadata = canonical({
            "run_id": self.run_id,
            "dataset_key": self.split.dataset_key,
            "round_index": self.round_index,
            "stage_id": "independent-eval",
            "prompt_version": PROMPT_VERSION,
            "task_contract": self.start_frame()["task_contract"],
        }).decode()
        block = self.prompt + "\n" + metadata
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            first = dict(messages[0])
            first["content"] = block + "\n\n" + str(first.get("content") or "")
            return [first, *messages[1:]]
        return [{"role": "system", "content": block}, *messages]

    # -- candidate measurement ---------------------------------------------- #
    def materialize(self, files: list[dict]) -> dict[str, bytes]:
        out = {}
        for item in files:
            path = item.get("path")
            if not safe_relative(path):
                raise FrameError("unsafe candidate path %r" % (path,))
            try:
                raw = base64.b64decode(item.get("data_base64") or "", validate=True)
            except Exception as exc:
                raise FrameError("bad base64 for %s" % path) from exc
            out[path] = raw
        return out

    def receipt_for(self, artifact_id: str, descriptor: dict,
                    blobs: dict[str, bytes]) -> dict:
        files = [{"path": path, "sha256": sha256_bytes(raw), "size_bytes": len(raw)}
                 for path, raw in sorted(blobs.items())]
        artifact_sha = sha256_bytes(canonical({"files": files, "artifact": descriptor}))
        return {"artifact_id": artifact_id, "artifact_sha256": artifact_sha,
                "artifact": descriptor, "files": files}

    def measure(self, artifact_id: str, descriptor: dict, blobs: dict[str, bytes],
                limit: int | None = None) -> dict:
        """Mount a candidate at ``/model`` and score it on the validation split."""
        if artifact_id in self.measured:
            return self.measured[artifact_id]
        model_dir = self.sandbox.staging / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        for path, raw in blobs.items():
            target = model_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

        checkpoint = None
        for name in descriptor.get("checkpoint_paths") or []:
            if name.endswith(".npz"):
                checkpoint = name
                break
        if checkpoint is None:
            checkpoint = (descriptor.get("checkpoint_paths") or ["checkpoint.npz"])[0]

        context = {
            "protocol": PREDICT_CONTEXT_PROTOCOL,
            "dataset_key": self.split.dataset_key,
            "task": self.split.task,
            "artifact": descriptor,
            "artifact_id": artifact_id,
        }
        rows = self.split.validation_rows if limit is None else self.split.validation_rows[:limit]
        argv = descriptor.get("predict_argv") or ["python3", "/app/harness.py", "predict"]
        command = ["docker", "run", "--rm", "-i", "--network", "none",
                   *self.sandbox.mount_args(with_model=True, with_state=False),
                   "-e", "CONTEST_ARTIFACT_PATH=/model/%s" % checkpoint,
                   "-e", "CONTEST_ARTIFACT_MANIFEST=%s" % json.dumps(descriptor),
                   "-e", "CONTEST_PREDICTION_CONTEXT=%s" % json.dumps(context),
                   "-e", "CONTEST_DATASET_KEY=%s" % self.split.dataset_key,
                   "-e", "CONTEST_TASK=%s" % self.split.task,
                   "--entrypoint", argv[0], self.sandbox.image, *argv[1:]]
        started = time.monotonic()
        # Streamed one sample at a time: the protocol requires the process to
        # write and flush each result *before* reading the next line. Feeding
        # the whole split at once would hide a missing flush and let a
        # submission that only works on complete input pass.
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        records, failure = [], None
        stderr_text = ""
        try:
            for position, row in enumerate(rows):
                proc.stdin.write(json.dumps({"task": self.split.task, "inputs": row["inputs"]},
                                            ensure_ascii=False) + "\n")
                proc.stdin.flush()
                if not select.select([proc.stdout], [], [], self.predict_read_timeout)[0]:
                    failure = ("no reply for sample %d of %d within %.0fs "
                               "(streaming protocol violated)" %
                               (position + 1, len(rows), self.predict_read_timeout))
                    break
                line = proc.stdout.readline()
                if not line:
                    failure = "predict closed stdout at sample %d of %d" % (position + 1, len(rows))
                    break
                try:
                    records.append(json.loads(line))
                except ValueError:
                    failure = "sample %d is not JSON" % (position + 1)
                    break
            if failure is None:
                proc.stdin.close()
            stderr_text = proc.stderr.read() or ""
        except Exception as exc:
            failure = "measurement error: %s" % exc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        elapsed = time.monotonic() - started
        result = {"ok": False, "rows": len(rows), "elapsed_seconds": round(elapsed, 2),
                  "exit": proc.returncode, "stdout_lines": len(records),
                  "stderr_tail": stderr_text[-3000:]}
        if failure:
            result["error"] = failure
        elif proc.returncode != 0:
            result["error"] = "predict exited %d" % proc.returncode
        elif len(records) != len(rows):
            result["error"] = "expected %d prediction lines, got %d" % (len(rows), len(records))
        else:
            try:
                result["metrics"] = scores_from_predictions(self.split, records, rows)
                result["ok"] = True
            except EvaluatorError as exc:
                result["error"] = str(exc)
        self.log.write("measure", artifact_id=artifact_id, **{
            key: value for key, value in result.items() if key != "stderr_tail"})
        self.prediction_runs.append({"artifact_id": artifact_id, **result})
        if result.get("ok"):
            self.measured[artifact_id] = result
        return result

    # -- RPC dispatch ------------------------------------------------------- #
    def reply(self, frame: dict) -> dict | None:
        """Build the host reply for one participant frame."""
        kind = frame.get("type")
        if kind == "history_request":
            return {"type": "history_response", "id": frame["id"], "records": self.history}

        if kind == "model_request":
            messages = self.bind_prompt(frame.get("messages") or [])
            options = {key: value for key, value in frame.items()
                       if key not in ("type", "id", "messages")}
            try:
                call_id, response = self.llm.complete(messages, options)
            except EvaluatorError as exc:
                self.errors.append(str(exc))
                self.log.write("model_error", error=str(exc))
                raise
            self.model_calls.append({"call_id": call_id,
                                     "max_tokens": options.get("max_tokens"),
                                     "temperature": options.get("temperature")})
            self.log.write("model_call", call_id=call_id,
                           max_tokens=options.get("max_tokens"))
            return {"type": "model_response", "id": frame["id"],
                    "call_id": call_id, "response": response}

        if kind == "state_request":
            files = frame.get("files")
            if not isinstance(files, list):
                raise FrameError("state files must be a list")
            blobs = self.materialize(files)
            digest = sha256_bytes(canonical({path: sha256_bytes(raw)
                                             for path, raw in sorted(blobs.items())}))
            if self.sandbox.state_dir.exists():
                shutil.rmtree(self.sandbox.state_dir)
            self.sandbox.state_dir.mkdir(parents=True)
            for path, raw in blobs.items():
                target = self.sandbox.state_dir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            self.state_present = True
            self.log.write("state_saved", files=len(blobs), sha=digest)
            return {"type": "state_response", "id": frame["id"], "state_sha256": digest}

        if kind == "candidate_request":
            return self._candidate(frame)

        if kind == "artifact_read_request":
            return self._artifact_read(frame)

        if kind == "round_complete":
            self.decision = self._decision(frame.get("record") or {})
            return {"type": "round_ack", "accepted": True}

        raise FrameError("unexpected frame type %r" % (kind,))

    def _candidate(self, frame: dict) -> dict:
        hypothesis = frame.get("hypothesis")
        change = frame.get("change_summary")
        configuration = frame.get("configuration")
        seed = frame.get("seed")
        descriptor = frame.get("artifact")
        files = frame.get("files")

        for label, text in (("hypothesis", hypothesis), ("change_summary", change)):
            if not isinstance(text, str) or not text.strip():
                raise FrameError("%s is required" % label)
            if len(text.encode("utf-8")) > TEXT_LIMIT:
                raise FrameError("%s exceeds %d bytes" % (label, TEXT_LIMIT))
        if not isinstance(configuration, dict):
            raise FrameError("configuration must be an object")
        if len(canonical(configuration)) > CONFIG_LIMIT:
            raise FrameError("configuration exceeds %d bytes" % CONFIG_LIMIT)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise FrameError("seed must be an integer")
        if not isinstance(descriptor, dict):
            raise FrameError("artifact descriptor missing")
        if set(descriptor) != {"checkpoint_paths", "code_paths", "predict_argv"}:
            raise FrameError("artifact descriptor has wrong keys")
        checkpoints = descriptor.get("checkpoint_paths")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise FrameError("at least one checkpoint path is required")
        if not isinstance(files, list) or not files:
            raise FrameError("candidate must contain files")
        if len(files) > FILE_LIMIT:
            raise FrameError("candidate exceeds %d files" % FILE_LIMIT)

        blobs = self.materialize(files)
        total = sum(len(raw) for raw in blobs.values())
        if total > BYTE_LIMIT:
            raise FrameError("candidate exceeds %d bytes" % BYTE_LIMIT)
        for path in checkpoints:
            if path not in blobs:
                raise FrameError("declared checkpoint %s is missing" % path)

        artifact_id = "cand-%d-%d" % (self.round_index, len(self.candidates) + 1)
        receipt = self.receipt_for(artifact_id, descriptor, blobs)
        self.candidates[artifact_id] = {"receipt": receipt, "blobs": blobs}
        self.artifacts[artifact_id] = receipt

        measured = self.measure(artifact_id, descriptor, blobs)
        if measured.get("ok"):
            metrics = measured["metrics"]
            self.log.write("candidate_accepted", artifact_id=artifact_id,
                           metrics=metrics, hypothesis=hypothesis[:200])
        else:
            self.errors.append("measurement failed for %s: %s"
                               % (artifact_id, measured.get("error")))
            self.log.write("candidate_measure_failed", artifact_id=artifact_id,
                           error=measured.get("error"))
            metrics = {}
        summary = {"protocol": "contest.candidate-summary.v1",
                   "validation_rows": measured.get("rows"),
                   "metrics": metrics,
                   "measurement_ok": bool(measured.get("ok")),
                   "measurement_error": measured.get("error")}
        return {"type": "candidate_response", "id": frame["id"],
                "artifact_id": artifact_id, "summary": summary}

    def _artifact_read(self, frame: dict) -> dict:
        artifact_id = frame.get("artifact_id")
        # The submission may restore an artifact produced in an *earlier* round
        # (the documented rollback path), so look in this round's candidates
        # first and then in the store carried across rounds.
        entry = self.candidates.get(artifact_id) or self.artifact_store.get(artifact_id)
        if entry is None:
            raise FrameError("unknown artifact %r" % (artifact_id,))
        path = frame.get("path")
        raw = entry["blobs"].get(path)
        if raw is None:
            raise FrameError("unknown artifact path %r" % (path,))
        offset, length = int(frame.get("offset")), int(frame.get("length"))
        chunk = raw[offset:offset + min(length, 65536)]
        if not chunk:
            raise FrameError("empty read")
        return {"type": "artifact_read_response", "id": frame["id"],
                "data_base64": base64.b64encode(chunk).decode(),
                "eof": offset + len(chunk) == len(raw)}

    # -- accept / rollback -------------------------------------------------- #
    def _decision(self, record: dict) -> dict:
        """Judge the round using host-measured numbers only."""
        task = self.split.task
        primary = PRIMARY_METRIC[self.split.dataset_key]
        direction = METRIC_DIRECTION[self.split.dataset_key]
        selected = record.get("selected_artifact_id")

        def metric_of(artifact_id):
            cached = self.measured.get(artifact_id)
            if cached is not None:
                return cached.get("metrics")
            run = next((item for item in self.prediction_runs
                        if item["artifact_id"] == artifact_id and item.get("ok")), None)
            return (run or {}).get("metrics")

        # Re-measure the selected candidate independently if it was not measured
        # (e.g. the submission selected the parent, keep None then).
        chosen_metrics = metric_of(selected)
        parent_metrics = None
        parent_id = self.parent_receipt.get("artifact_id")
        if selected != parent_id:
            # Prefer the host's own baseline measurement; fall back to whatever
            # this round independently measured for a previously submitted
            # parent candidate.
            parent_metrics = self.baseline_metrics
            if parent_metrics is None:
                run = next((item for item in self.prediction_runs
                            if item["artifact_id"] == parent_id and item.get("ok")), None)
                parent_metrics = (run or {}).get("metrics")
        else:
            # The submission kept the parent model. That is a legitimate choice,
            # so score the parent with the host's own baseline measurement and
            # report it as a retained round rather than an unmeasured one.
            parent_metrics = chosen_metrics or self.baseline_metrics
            chosen_metrics = chosen_metrics or self.baseline_metrics

        verdict = "no_measurement"
        delta = None
        if chosen_metrics and chosen_metrics.get(primary) is not None:
            if parent_metrics and parent_metrics.get(primary) is not None:
                cand_value = float(chosen_metrics[primary])
                base_value = float(parent_metrics[primary])
                delta = cand_value - base_value
                improved = delta > 0 if direction == "higher" else delta < 0
                verdict = "improved" if improved else "not_improved"
            else:
                verdict = "measured_no_parent_baseline"
            if selected == parent_id:
                verdict = "kept_parent"
                delta = 0.0

        # Protocol hygiene, independent of numbers.
        model_call_ids = record.get("model_call_ids") or []
        issued = {call["call_id"] for call in self.model_calls}
        real_calls = [call_id for call_id in model_call_ids if call_id in issued]
        hygiene = {
            "model_call_ids_nonempty": bool(model_call_ids),
            "model_call_ids_issued_by_host": len(real_calls) == len(model_call_ids),
            "history_ids_subset": set(record.get("history_ids") or []) <= {
                item["id"] for item in self.history},
            "reason_present": bool((record.get("reason") or "").strip()),
            "selected_artifact_known": selected in self.artifacts,
            "hypothesis_present": bool((record.get("hypothesis") or "").strip()),
        }

        decision = {
            "selected_artifact_id": selected,
            "verdict": verdict,
            "primary_metric": primary,
            "selected_metrics": chosen_metrics,
            "parent_metrics": parent_metrics,
            "parent_metrics_source": ("host_baseline" if parent_metrics is self.baseline_metrics
                                      else "host_measurement"),
            "delta": delta,
            "hygiene": hygiene,
            "record": {key: value for key, value in record.items()
                       if key in ("hypothesis", "change_summary", "reason", "seed")},
        }
        self.log.write("decision", **decision)
        return decision

    # -- run ---------------------------------------------------------------- #
    def run(self) -> dict:
        frame = self.start_frame()
        command = ["docker", "run", "--rm", "-i",
                   *self.sandbox.mount_args(with_model=False,
                                            with_state=self.state_present),
                   "-e", "PYTHONUNBUFFERED=1",
                   "--entrypoint", "python3", self.sandbox.image,
                   "/app/harness.py", "train"]
        # Dump the start frame before booting, so a crash still leaves evidence
        # of what the host intended to send.
        self.log.write("start_frame", frame=frame)
        self.log.write("round_start", dataset=self.split.dataset_key,
                       round=self.round_index)
        started = time.monotonic()
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)

        stderr_chunks: list[str] = []

        def drain():
            for line in proc.stderr:
                stderr_chunks.append(line)

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()

        outcome = {"status": "unknown"}
        try:
            proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                if not line.strip():
                    continue
                try:
                    incoming = json.loads(line)
                except ValueError:
                    self.errors.append("non-JSON on protocol stream: %r" % line[:300])
                    continue
                self.log.write("frame_in", frame=incoming if incoming.get("type") != "round_complete"
                               else {**incoming, "record": {**incoming.get("record", {}),
                                                            "reason": "<omitted>"}})
                if incoming.get("type") == "round_failed":
                    self.errors.append("submission failed: %s" % incoming.get("error"))
                    outcome = {"status": "failed", "error": incoming.get("error")}
                    break
                try:
                    response = self.reply(incoming)
                except FrameError as exc:
                    self.errors.append("host rejected %s: %s" % (incoming.get("type"), exc))
                    outcome = {"status": "host_rejected", "error": str(exc)}
                    break
                if response is not None:
                    proc.stdin.write(json.dumps(response, ensure_ascii=False,
                                                allow_nan=False) + "\n")
                    proc.stdin.flush()
                    if response.get("type") == "round_ack":
                        # Terminal exchange: the round is complete. Drain the
                        # remaining stdout so the process exits cleanly.
                        outcome = {"status": "completed"}
                        break
            # A clean `break` after round_ack is success; a closed stream with
            # no ack is not.
            if outcome.get("status") not in ("completed", "failed", "host_rejected"):
                outcome = {"status": "stream_closed"}
        except Exception as exc:  # keep the report useful even on host bugs
            self.errors.append("host exception: %s" % exc)
            outcome = {"status": "host_exception", "error": str(exc),
                       "traceback": traceback.format_exc()[-3000:]}
        finally:
            elapsed = time.monotonic() - started
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            thread.join(timeout=5)

        stderr_text = "".join(stderr_chunks)
        outcome.update({
            "elapsed_seconds": round(elapsed, 2),
            "exit": proc.returncode,
            "stderr_tail": stderr_text[-12000:],
            "model_calls": len(self.model_calls),
            "candidates": list(self.artifacts.values()),
            "predictions": self.prediction_runs,
            "decision": self.decision,
            "errors": self.errors,
        })
        self.log.write("round_end", **{key: value for key, value in outcome.items()
                                       if key not in ("stderr_tail", "candidates")})
        return outcome


def measure_baseline(sandbox: "Sandbox", split: PublicSplit, log: RunLog,
                     timeout: float, limit: int | None = None) -> dict:
    """Score the shipped baseline independently, using its own predict entry.

    The baseline package under ``/previous/baselines`` is the platform's
    reference implementation. Running it host-side gives a parent number the
    submission never touched, so accept/rollback is judged against a real
    baseline instead of a self-reported one.
    """
    rows = split.validation_rows if limit is None else split.validation_rows[:limit]
    stdin_lines = "\n".join(json.dumps({"task": split.task, "inputs": row["inputs"]},
                                       ensure_ascii=False) for row in rows) + "\n"
    command = ["docker", "run", "--rm", "-i", "--network", "none",
               "-e", "BASELINE_CPU_THREADS=1",
               *sandbox.mount_args(with_model=False, with_state=False),
               "--entrypoint", "sh", sandbox.image, "-c",
               "cd /previous/baselines && exec python3 predict.py "
               "--checkpoint /previous/checkpoint.npz"]
    try:
        proc = subprocess.run(command, input=stdin_lines, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.write("baseline_timeout", rows=len(rows))
        return {"ok": False, "error": "baseline prediction timed out"}
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if proc.returncode != 0:
        log.write("baseline_error", exit=proc.returncode,
                  stderr=(proc.stderr or "")[-2000:])
        return {"ok": False, "error": "baseline exited %d: %s"
                % (proc.returncode, (proc.stderr or "")[-500:])}
    if len(lines) != len(rows):
        return {"ok": False, "error": "baseline produced %d lines for %d rows"
                % (len(lines), len(rows))}
    try:
        records = [json.loads(line) for line in lines]
        metrics = scores_from_predictions(split, records, rows)
    except (ValueError, EvaluatorError) as exc:
        return {"ok": False, "error": "baseline output invalid: %s" % exc}
    log.write("baseline_measured", metrics=metrics, rows=len(rows))
    return {"ok": True, "metrics": metrics, "rows": len(rows)}


# --------------------------------------------------------------------------- #
# Static audit: does the shipped code respect the platform contract?
# --------------------------------------------------------------------------- #

AUDIT_RULES = [
    ("key_hardcoded",
     re.compile(r"(?i)\b(pytrio|pytrio_api_key|api[_-]?key)\b\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
     "疑似硬编码 API 密钥"),
    ("write_app",
     re.compile(r"""open\(\s*['\"]/app|Path\(\s*['\"]/app[^'\"]*['\"]\s*\)\s*\.\s*write|""" 
                r"""['\"]/app/[^'\"]*['\"]\s*,\s*['\"]w"""),
     "疑似向只读的 /app 写入"),
    ("hidden_data",
     re.compile(r"(?i)/data/(hidden|private|test_secret)|/previous/\.\.|hidden_labels"),
     "疑似访问隐藏数据路径"),
    ("outbound_http",
     re.compile(r"requests\.(post|get)\(\s*['\"]https?://(?!127\.0\.0\.1|localhost)"),
     "疑似绕过宿主网关的外部网络请求"),
    ("subprocess_shell",
     re.compile(r"shell\s*=\s*True"),
     "使用了 shell=True 的进程调用"),
]
AUDIT_SUFFIXES = {".py", ".toml", ".txt", ".cfg", ".sh"}


def static_audit(root: Path) -> dict:
    """Scan shipped sources for contract violations. Read-only, fast, no image."""
    findings = defaultdict(list)
    files_scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIT_SUFFIXES:
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, pattern, message in AUDIT_RULES:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings[name].append({"file": path.relative_to(root).as_posix(),
                                       "line": line, "message": message,
                                       "excerpt": match.group(0)[:160]})
    return {"files_scanned": files_scanned,
            "findings": {name: items for name, items in findings.items()},
            "clean": not findings}


# --------------------------------------------------------------------------- #
# Baseline artifact packaging
# --------------------------------------------------------------------------- #

def build_baseline(split: PublicSplit, staging_previous: Path,
                   artifact_id: str, code_dir: Path | None) -> dict:
    """Package the shipped baseline checkpoint as the round's parent artifact.

    The parent must look exactly like a platform artifact: declared
    checkpoint/code paths, a file manifest and a digest over both. The
    submission continues from it through ``rnd.parent``.
    """
    staging_previous.mkdir(parents=True, exist_ok=True)
    blobs: dict[str, bytes] = {}

    checkpoint = split.package / "checkpoint.npz"
    if not checkpoint.is_file():
        raise EvaluatorError("baseline checkpoint missing at %s" % checkpoint)
    blobs["checkpoint.npz"] = checkpoint.read_bytes()

    code_paths: list[str] = []
    if code_dir is not None:
        for source in sorted((code_dir).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = "baselines/" + source.relative_to(code_dir).as_posix()
            if not safe_relative(relative):
                continue
            blobs[relative] = source.read_bytes()
            code_paths.append(relative)

    for path, raw in blobs.items():
        target = staging_previous / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": code_paths,
                  "predict_argv": ["python3", "/app/harness.py", "predict"]}
    files = [{"path": path, "sha256": sha256_bytes(raw), "size_bytes": len(raw)}
             for path, raw in sorted(blobs.items())]
    artifact_sha = sha256_bytes(canonical({"files": files, "artifact": descriptor}))
    return {"artifact_id": artifact_id, "artifact_sha256": artifact_sha,
            "artifact": descriptor, "files": files}


# --------------------------------------------------------------------------- #
# Official prompt (host-authored; the submission never writes it)
# --------------------------------------------------------------------------- #

PROMPT = (
    "你是工业模型竞赛中的受控研究代理。平台已冻结本轮任务合同：数据集、任务类型、"
    "主指标与数据投影都写在 start 帧里，不能更改。你只能使用 /data 下公开的训练与"
    "验证投影，不得访问隐藏测试集，也不得从样本 id、文件名或元数据反推标签。\n"
    "每一轮请：1) 读取父检查点与公开数据；2) 做可证伪的实验并如实记录假设；"
    "3) 提交一个候选产物（checkpoint、必要代码与 predict_argv），等待平台独立测量；"
    "4) 依据真实测量做接受或回滚；5) 保存供下一轮续训的原生状态；6) 调用 finish 并"
    "引用本轮真实的模型回执号。输出协议帧走 stdout，日志走 stderr。"
)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

def _catalogue_text(catalogue) -> str:
    if not catalogue:
        return "未查询（stub 模式或显式指定模型）"
    parts = []
    for item in catalogue:
        flag = "可用" if item.get("sample_available") is not False else "不可用"
        parts.append("`%s` %s" % (item.get("model_id"), flag))
    return "；".join(parts)


def render_report(result: dict) -> str:
    lines = ["# 独立评测报告", ""]
    lines += [f"- 运行 ID：`{result['run_id']}`",
              f"- 评测镜像：`{result['image']}`",
              f"- 模型：`{result['model']}`（{'离线 stub，未调用真实模型' if result['stub'] else 'PyTRiO 真实调用'}）",
              f"- 网关：`{result['base_url']}`（比赛只允许 PyTRiO）",
              f"- 模型可用性：{_catalogue_text(result.get('model_catalogue'))}",
              f"- 数据集：{', '.join(result['datasets'])}",
              f"- 每数据集轮数：{result['rounds']}",
              f"- 开始时间：{result['started_at']}", ""]

    if result.get("libgomp_repair"):
        lines += ["## 环境修补（评测端代劳，说明镜像缺件）", "",
                  "本镜像缺少 `libgomp.so.1`（LightGBM 扩展在导入期 dlopen 它），",
                  "评测端从镜像自带的 `scikit_learn.libs/` 提取并挂载补齐，才让本轮可跑。",
                  "修复后的镜像应安装 `libgomp1`；若要验证镜像本身，用 `--no-libgomp-repair`。", ""]
    if result.get("describe"):
        describe = result["describe"]
        lines += ["## describe 自检", "",
                  f"- 退出码：{describe['exit']}",
                  f"- 输出：`{json.dumps(describe.get('record'), ensure_ascii=False)}`", ""]

    audit = result.get("audit") or {}
    if audit:
        lines += ["## 静态合规审计", "",
                  f"扫描 {audit.get('files_scanned', 0)} 个文件；"
                  + ("未发现契约违规模式。" if audit.get("clean") else "发现可疑模式："), ""]
        for name, items in (audit.get("findings") or {}).items():
            lines.append(f"**{name}**（{len(items)} 处）")
            for item in items[:5]:
                lines.append(f"- `{item['file']}:{item['line']}` {item['message']}："
                             f"`{item['excerpt']}`")
            lines.append("")

    llm = result.get("llm") or {}
    replies = llm.get("replies") or []
    if replies:
        lines += ["## 模型调用明细", "",
                  "思考模型（Qwen3.5/3.8）先写 `reasoning` 再写 `content`。"
                  "预算被 reasoning 吃光时 `content` 为空，提交端只能退回模板提案。", "",
                  "| # | 请求 max_tokens | 完成 tokens | reasoning | content 字符 | 可解析 JSON |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for index, reply in enumerate(replies, 1):
            lines.append("| %d | %s | %s | %s | %s | %s |" % (
                index, reply.get("requested_max_tokens"), reply.get("completion_tokens"),
                reply.get("reasoning_tokens"), reply.get("content_chars"),
                "是" if reply.get("content_parsable_json") else "**否**"))
        lines.append("")

    baselines = result.get("baselines") or {}
    if baselines:
        lines += ["## 宿主独立测量的父基线", "",
                  "| 数据集 | 状态 | 主指标 | 详细指标 |",
                  "| --- | --- | --- | --- |"]
        for key, base in baselines.items():
            if base.get("ok"):
                metrics = base.get("metrics") or {}
                primary = PRIMARY_METRIC[key]
                lines.append("| %s | 成功 | %.4f | `%s` |" % (
                    key, metrics.get(primary, float("nan")),
                    json.dumps(metrics, ensure_ascii=False, default=str)[:220]))
            else:
                lines.append("| %s | 失败 | — | %s |" % (key, base.get("error", "")))
        lines.append("")

    for dataset, rounds in result["rounds_detail"].items():
        lines += [f"## 数据集 {dataset}", ""]
        lines += ["| 轮 | 状态 | 耗时 s | 模型调用 | 候选 | 主指标 | 父指标 | 判定 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for item in rounds:
            decision = item.get("decision") or {}
            chosen = (decision.get("selected_metrics") or {})
            parent = (decision.get("parent_metrics") or {})
            primary = decision.get("primary_metric", PRIMARY_METRIC[dataset])
            fmt = lambda value: "—" if value is None else (f"{value:.4f}")
            lines.append("| %d | %s | %.1f | %d | %d | %s | %s | %s |" % (
                item["round"], item["status"], item.get("elapsed_seconds", 0.0),
                item.get("model_calls", 0), len(item.get("predictions") or []),
                fmt(chosen.get(primary)), fmt(parent.get(primary)),
                decision.get("verdict", "—")))
        issues = []
        for item in rounds:
            for error in item.get("errors") or []:
                issues.append(f"- 第 {item['round']} 轮：{error}")
        if issues:
            lines += ["", "**问题记录**", ""] + issues
        lines.append("")

    lines += ["## 结论", ""]
    for dataset, rounds in result["rounds_detail"].items():
        verdicts = [(item.get("decision") or {}).get("verdict") for item in rounds]
        ok = sum(1 for item in rounds if item["status"] == "completed")
        improved = sum(1 for verdict in verdicts if verdict == "improved")
        lines.append(f"- 数据集 {dataset}：{ok}/{len(rounds)} 轮完成，{improved} 轮独立测量为改进。")

    checks = contest_checks(result)
    lines += ["", "## 竞赛硬性指标", "",
              "| 指标 | 要求 | 实测 | 判定 |",
              "| --- | --- | --- | --- |"]
    for check in checks:
        lines.append("| %s | %s | %s | %s |" % (
            check["name"], check["requirement"], check["actual"],
            "通过" if check["pass"] else "未通过"))
    failed = [check["name"] for check in checks if not check["pass"]]
    lines += ["", ("全部硬性指标通过。" if not failed
                   else "未通过：" + "、".join(failed) + "。")]
    lines.append("")
    result["contest_checks"] = checks
    return "\n".join(lines)


def contest_checks(result: dict) -> list[dict]:
    """Judge the run against the contest's explicit requirements (main.md)."""
    rounds_detail = result.get("rounds_detail") or {}
    checks = []

    # 每个数据集进行 10 轮优化
    per_dataset_ok = all(len(items) == CONTEST_ROUNDS for items in rounds_detail.values())
    checks.append({
        "name": "轮数",
        "requirement": "每数据集 %d 轮" % CONTEST_ROUNDS,
        "actual": ", ".join("%s=%d" % (key, len(items))
                            for key, items in sorted(rounds_detail.items())) or "无",
        "pass": bool(rounds_detail) and per_dataset_ok,
    })

    # 至少 3 次「读历史—做实验—依据结果决策」完整记录
    decision_rounds = 0
    for items in rounds_detail.values():
        for item in items:
            decision = item.get("decision") or {}
            if (item.get("status") == "completed"
                    and (decision.get("record") or {}).get("reason")
                    and (item.get("model_calls") or 0) >= 1
                    and item.get("predictions")):
                decision_rounds += 1
    checks.append({
        "name": "决策记录",
        "requirement": "≥%d 次完整决策" % MIN_DECISION_ROUNDS,
        "actual": "%d 次" % decision_rounds,
        "pass": decision_rounds >= MIN_DECISION_ROUNDS,
    })

    # 效果不低于平台基线（按最后选中产物的独立测量）
    regressions = []
    for key, items in rounds_detail.items():
        primary = PRIMARY_METRIC[key]
        direction = METRIC_DIRECTION[key]
        base = ((result.get("baselines") or {}).get(key) or {}).get("metrics") or {}
        base_value = base.get(primary)
        final = (items[-1].get("decision") or {}).get("selected_metrics") or {}
        final_value = final.get(primary)
        if base_value is None or final_value is None:
            regressions.append("%s 无法比较" % key)
            continue
        worse = (final_value < base_value) if direction == "higher" else (final_value > base_value)
        if worse:
            regressions.append("%s 劣于基线(%.4f vs %.4f)" % (key, final_value, base_value))
    checks.append({
        "name": "不低于基线",
        "requirement": "三个数据集最终产物 ≥ 基线",
        "actual": "全部达标" if not regressions else "；".join(regressions),
        "pass": not regressions,
    })

    # 稳定性：全部轮次均完成，无崩溃/协议违约
    total = sum(len(items) for items in rounds_detail.values())
    completed = sum(1 for items in rounds_detail.values()
                    for item in items if item.get("status") == "completed")
    checks.append({
        "name": "运行稳定性",
        "requirement": "全部轮次无失败",
        "actual": "%d/%d 轮完成" % (completed, total),
        "pass": total > 0 and completed == total,
    })

    # 无干预：每轮都真实调用平台模型（不许零调用蒙混）
    zero_call = ["%s#%s" % (key, item.get("round")) for key, items in rounds_detail.items()
                 for item in items if not (item.get("model_calls") or 0)]
    checks.append({
        "name": "无人干预",
        "requirement": "每轮均有真实模型调用",
        "actual": "全部满足" if not zero_call else "缺调用: " + ", ".join(zero_call),
        "pass": not zero_call and total > 0,
    })

    # 恢复速度（评分项之一）：每轮结束时不应停在历史最差点。
    # 竞赛给「最差阶段与恢复速度」10 分，长期停在低谷是扣分项。
    stalls = []
    for key, items in rounds_detail.items():
        primary = PRIMARY_METRIC[key]
        direction = METRIC_DIRECTION[key]
        worst, stalled = None, False
        for item in items:
            metrics = ((item.get("decision") or {}).get("selected_metrics") or {})
            value = metrics.get(primary)
            if value is None:
                continue
            if worst is None:
                worst = value
                continue
            worse = (value < worst) if direction == "higher" else (value > worst)
            if worse:
                worst, stalled = value, True
            else:
                stalled = False
        if stalled:
            stalls.append(key)
    checks.append({
        "name": "恢复速度",
        "requirement": "每轮结束后不停在历史最差点",
        "actual": "全部恢复" if not stalls else "停在低谷: " + ", ".join(stalls),
        "pass": not stalls,
    })

    # 模型回执可用性：RSI 的「提假设—做实验」环节是否真由模型驱动。
    # 思考模型会把预算耗在 reasoning 上；预算不足时 content 为空或截断，
    # 提交端只能退回模板提案，此时分数提升与模型能力无关。
    llm = result.get("llm") or {}
    calls = llm.get("calls") or 0
    unusable = llm.get("unusable_content") or 0
    usable = max(0, calls - unusable)
    ratio = (float(usable) / calls) if calls else 0.0
    checks.append({
        "name": "模型回执可用",
        "requirement": "回执含可解析 JSON（否则退化为模板）",
        "actual": "%d/%d 次可用（%.0f%%）" % (usable, calls, ratio * 100),
        "pass": calls > 0 and ratio >= 0.5,
    })

    # 资源效率（评分项之一）：单轮耗时不应逼近墙钟上限。
    wall = float(result.get("wall_seconds") or 0.0)
    slow = []
    for key, items in rounds_detail.items():
        durations = [item.get("elapsed_seconds") or 0.0 for item in items]
        if durations and wall and max(durations) > wall * 0.9:
            slow.append("%s 最大 %.0fs/%.0fs" % (key, max(durations), wall))
    checks.append({
        "name": "资源效率",
        "requirement": "单轮不逼近墙钟上限",
        "actual": "全部充裕" if not slow else "；".join(slow),
        "pass": not slow,
    })
    return checks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def clean_runs(run_dir: Path, image: str):
    """Remove a run directory, using the container for uid-65534-owned files."""
    try:
        shutil.rmtree(run_dir)
        return True
    except PermissionError:
        pass
    try:
        subprocess.run(["docker", "run", "--rm", "-u", "0", "--entrypoint", "sh",
                        "-v", "%s:%s" % (run_dir, run_dir), image, "-c",
                        "rm -rf %s/* %s/.[!.]*" % (run_dir, run_dir)],
                       capture_output=True, timeout=300)
        shutil.rmtree(run_dir, ignore_errors=True)
        return not run_dir.exists()
    except (OSError, subprocess.SubprocessError):
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independent virtual contest host for the submission under .eval/")
    parser.add_argument("--model", default=os.environ.get("PYTRIO_MODEL"),
                        help="PyTRiO model id (or PYTRIO_MODEL); omit to auto-discover")
    parser.add_argument("--base-url", default=PYTRIO_BASE_URL,
                        help="PyTRiO gateway root (contest rule: PyTRiO only)")
    parser.add_argument("--datasets", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"], help="datasets to evaluate")
    parser.add_argument("--rounds", type=int, default=CONTEST_ROUNDS,
                        help="optimisation rounds per dataset (contest: %(default)s)")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="submission image (default: %(default)s)")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="dataset packages root (default: %(default)s)")
    parser.add_argument("--wall-seconds", type=float, default=300.0,
                        help="wall clock budget handed to the submission")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="host-side timeout per container step")
    parser.add_argument("--predict-read-timeout", type=float, default=120.0,
                        help="per-sample ceiling while streaming predictions")
    parser.add_argument("--measure-limit", type=int, default=0,
                        help="score only the first N validation rows (0 = all)")
    parser.add_argument("--stub", action="store_true",
                        help="offline stub model, no API calls (protocol smoke test)")
    parser.add_argument("--runs-dir", default=str(HERE / "runs"))
    parser.add_argument("--no-libgomp-repair", action="store_true",
                        help="do not inject libgomp.so.1 (use on images that ship it)")
    parser.add_argument("--clean", action="store_true",
                        help="delete previous run directories before starting")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("PYTRIO_API_KEY")
    if not args.stub and not api_key:
        sys.stderr.write("PYTRIO_API_KEY is not set (or pass --stub for an offline run)\n")
        return 2
    if not args.stub and not api_key.startswith("trio_"):
        # Catching this early beats a 401 after the first round has burned time.
        sys.stderr.write(
            "PYTRIO_API_KEY does not look like a PyTRiO key (expected a 'trio_' "
            "prefix). The contest allows PyTRiO only; keys from other providers "
            "will be rejected by https://pytrio.com.\n")
        return 2

    if args.clean:
        runs_root = Path(args.runs_dir)
        if runs_root.is_dir():
            for old_run in runs_root.iterdir():
                clean_runs(old_run, args.image)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "events.jsonl")

    data_root = Path(args.data_root)
    splits = {}
    for key in args.datasets:
        package = data_root / DATA_PACKAGES[key]
        if not package.is_dir():
            sys.stderr.write("missing dataset package %s\n" % package)
            return 2
        splits[key] = PublicSplit(key, package)

    sandbox = Sandbox(args.image, run_dir, log,
                      repair_libgomp=not args.no_libgomp_repair)
    if not sandbox.docker_available():
        sys.stderr.write("docker image %s is not available\n" % args.image)
        return 2
    describe = sandbox.describe()
    audit = static_audit(REPO / "app")

    llm = LLMClient(args.model, args.base_url, api_key, use_stub=args.stub)
    llm.log = log
    if not args.stub:
        try:
            resolved = llm.discover_model()
        except EvaluatorError as exc:
            sys.stderr.write("PyTRiO model discovery failed: %s\n" % exc)
            return 2
        sys.stderr.write("PyTRiO model: %s (%s)\n" % (resolved, llm.chat_url))
    result = {
        "run_id": run_id,
        "image": args.image,
        "model": llm.model or args.model,
        "requested_model": args.model,
        "base_url": args.base_url,
        "gateway": "pytrio",
        "model_catalogue": getattr(llm, "model_catalogue", None),
        "stub": bool(args.stub),
        "datasets": args.datasets,
        "rounds": args.rounds,
        "wall_seconds": args.wall_seconds,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "libgomp_repair": sandbox._libgomp is not None,
        "describe": describe,
        "audit": audit,
        "rounds_detail": {},
        "llm": {},
    }

    for key, split in splits.items():
        sandbox.write_data(split)
        code_dir = split.package / "baselines"
        parent = build_baseline(split, sandbox.staging / "previous",
                                "parent-%s-000" % key,
                                code_dir if code_dir.is_dir() else None)
        # Independent parent number: measured by the host on the baseline itself.
        limit = args.measure_limit or None
        baseline = measure_baseline(sandbox, split, log, args.timeout, limit)
        baseline_metrics = baseline.get("metrics") if baseline.get("ok") else None
        result.setdefault("baselines", {})[key] = baseline
        # Reset round-to-round state for a fresh dataset.
        if sandbox.state_dir.exists():
            shutil.rmtree(sandbox.state_dir)
        sandbox.state_dir.mkdir(parents=True)
        state_present = False
        state_sha = ""

        rounds = []
        artifact_store: dict[str, dict] = {}
        measured: dict[str, dict] = {}
        known_artifacts: list[dict] = [parent]
        for index in range(1, args.rounds + 1):
            host = VirtualHost(
                sandbox=sandbox, split=split, llm=llm, log=log, run_id=run_id,
                round_index=index, prompt=PROMPT, model_name=llm.model or args.model,
                wall_seconds=args.wall_seconds, parent_receipt=parent,
                artifacts=list(known_artifacts), state_sha256=state_sha,
                state_present=state_present, timeout=args.timeout,
                baseline_metrics=baseline_metrics,
                predict_read_timeout=args.predict_read_timeout,
                rounds_per_dataset=args.rounds)
            host.artifact_store = artifact_store
            host.measured = measured
            outcome = host.run()
            outcome["round"] = index
            rounds.append(outcome)
            # Remember every artifact this round saw, so later rounds can
            # re-measure one the submission selects from history.
            artifact_store.update(host.candidates)
            measured.update(host.measured)
            for receipt in host.artifacts.values():
                if receipt["artifact_id"] not in {item["artifact_id"] for item in known_artifacts}:
                    known_artifacts.append(receipt)
            state_present = host.state_present
            state_sha = host.state_sha256
            # The next round's parent is whatever this round selected.
            picked = (host.decision or {}).get("selected_artifact_id")
            if picked in host.artifacts:
                parent = host.artifacts[picked]
                if picked in host.candidates:
                    for path, raw in host.candidates[picked]["blobs"].items():
                        target = sandbox.staging / "previous" / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw)
        result["rounds_detail"][key] = rounds

    result["llm"] = {"calls": llm.calls, "failures": llm.failures,
                     "total_tokens": llm.total_tokens,
                     "empty_content": llm.empty_content,
                     "unusable_content": llm.unusable_content,
                     "replies": llm.replies}
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    log.close()

    print(render_report(result))
    print("artifacts: %s" % run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
