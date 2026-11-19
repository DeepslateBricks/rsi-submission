"""Input safety, contract parsing, and group/block extraction.

Only `inputs` ever enters the feature matrix. `id` / `meta` are used for grouping
and block statistics exclusively, never as model inputs.
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}
GROUP_BLOCK_SIZE = 50


def flatten_inputs(inputs: Any):
    """Sorted-key recursive flatten, mirroring the official baseline loader."""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def align_paths(paths, expected):
    """Return reordering indices; never infer new feature names at predict time."""
    if list(paths) == list(expected):
        return list(range(len(expected)))
    if len(paths) != len(expected) or set(paths) != set(expected):
        raise ValueError("feature schema changed")
    lookup = {p: i for i, p in enumerate(paths)}
    return [lookup[p] for p in expected]


def extract_group(row, dataset_key, position):
    """Unit / bearing / batch-block key used ONLY for CV and evaluation blocks."""
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    identity = str(row.get("id", ""))
    if dataset_key == "B":
        name = meta.get("bearing")
        if not isinstance(name, str) or not name:
            match = re.search(r"(Bearing\d+_\d+)", identity, re.I)
            name = match.group(1) if match else "unknown_bearing"
        return str(name)
    if dataset_key == "C":
        subset = meta.get("source_subset")
        unit = meta.get("unit")
        if subset is None or unit is None:
            match = re.search(r"(fd\d+)[^u]*u(\d+)", identity, re.I)
            subset = match.group(1).upper() if match else "unknown"
            unit = match.group(2) if match else "0"
        return "%s_u%s" % (subset, unit)
    # A: time-ordered batches; never shuffle across batches.
    batch = meta.get("source_batch")
    index = meta.get("source_index", position + 1)
    if batch is None:
        match = re.search(r"_b(\d+)_", identity, re.I)
        batch = match.group(1) if match else "0"
    try:
        block = (int(index) - 1) // GROUP_BLOCK_SIZE
    except (TypeError, ValueError):
        block = position // GROUP_BLOCK_SIZE
    return "b%s_blk%d" % (batch, block)


@dataclass
class Dataset:
    task: str
    dataset_key: str
    paths: List[str]
    x: np.ndarray
    y: np.ndarray
    ids: List[str]
    groups: List[str]
    metas: List[dict]
    classes: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.ids)


def load_rows(rows, task, dataset_key, classes=None, expected_paths=None):
    ids, metas, groups, values, targets, paths = [], [], [], [], [], expected_paths
    seen = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or "inputs" not in row:
            raise ValueError("invalid data row")
        if row["id"] in seen:
            raise ValueError("duplicate sample id")
        seen.add(row["id"])
        current, vector = flatten_inputs(row["inputs"])
        if paths is None:
            paths = current
        if current != paths:
            raise ValueError("feature schema changed")
        answer = row.get("answer")
        if task == "classification":
            if not isinstance(answer, str) or answer not in (classes or []):
                raise ValueError("label outside configured classes")
            target = classes.index(answer)
        elif task == "regression":
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer) or answer < 0:
                raise ValueError("RUL must be finite and nonnegative")
            target = float(answer)
        else:
            raise ValueError("unsupported task")
        ids.append(row["id"])
        metas.append(row.get("meta") if isinstance(row.get("meta"), dict) else {})
        groups.append(extract_group(row, dataset_key, position))
        values.append(vector)
        targets.append(target)
    if not ids:
        raise ValueError("empty dataset")
    return Dataset(
        task=task, dataset_key=dataset_key, paths=list(paths),
        x=np.asarray(values, dtype=np.float64),
        y=np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
        ids=ids, groups=groups, metas=metas, classes=list(classes or []),
    )


def sample_vector(inputs, expected_paths):
    paths, values = flatten_inputs(inputs)
    order = align_paths(paths, expected_paths)
    return np.asarray(values, dtype=np.float64)[order]


def group_folds(groups, n_splits=4, seed=42):
    """Deterministic group folds (hash-based; no sample leakage across units)."""
    unique = sorted(set(groups))
    buckets = [[] for _ in range(n_splits)]
    for name in unique:
        digest = int.from_bytes(__import__("hashlib").sha256(("%d:%s" % (seed, name)).encode()).digest()[:4], "big")
        buckets[digest % n_splits].append(name)
    member_to_fold = {name: index for index, names in enumerate(buckets) for name in names}
    return np.asarray([member_to_fold[g] for g in groups], dtype=np.int64)


def validate_no_leak(dataset: Dataset):
    for path in dataset.paths:
        lowered = path.lower()
        for token in FORBIDDEN_INPUTS:
            if token in lowered:
                raise ValueError("forbidden target-derived feature: " + path)
    if len(set(dataset.ids)) != len(dataset.ids):
        raise ValueError("duplicate ids")
    return True
