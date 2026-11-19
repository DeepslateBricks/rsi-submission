"""Resumable NumPy MLP backbone + CEP residual expert pool.

The backbone is byte-format compatible with contest.cpu-mlp.v1. Architecture
migration is explicit: old weights and Adam moments are copied losslessly and
new input columns / hidden neurons start at zero, so the migrated model is
numerically identical to its parent at the moment of migration.
"""
import copy
import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np

from . import primitives

FORMAT = "contest.cpu-mlp.v1"
PARAMETERS = ("w1", "b1", "w2", "b2")
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class MLPBackbone:
    def __init__(self, task, paths, classes, x, y, hidden=32, seed=42, learning_rate=0.003, batch_size=128):
        if task not in ("classification", "regression") or not 1 <= hidden <= 128:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= 100000:
            raise ValueError("model must keep fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
            "b2": np.zeros(output),
        }
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {
            "format": FORMAT, "device": "cpu", "task": task, "paths": list(paths),
            "classes": list(classes), "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
            "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
            "target_mean": float(y.mean()) if task == "regression" else 0.0,
            "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0,
        }
        self.validate()

    def validate(self):
        meta = self.meta
        if meta.get("format") != FORMAT or meta.get("device") != "cpu" or meta.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(meta.get(name)) is not int or meta[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= meta["hidden"] <= 128 or not 1 <= meta["batch_size"] <= 65536 or not 0 < meta["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= meta["seed"] < 2**32:
            raise ValueError("seed must fit unsigned 32-bit")
        paths = meta.get("paths")
        if not isinstance(paths, list) or not 1 <= len(paths) <= 4096 or any(not isinstance(p, str) for p in paths) or len(set(paths)) != len(paths):
            raise ValueError("invalid feature schema")
        classes = meta.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (meta["task"] == "classification" and len(classes) < 2) or (meta["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(meta.get(name), bool) or not isinstance(meta.get(name), (float, int)) or not math.isfinite(meta[name]):
                raise ValueError("invalid numeric state")
        if not 0 < meta["learning_rate"] <= 1 or meta["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size, hidden = len(paths), meta["hidden"]
        output = len(classes) if meta["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor invalid")
        if sum(value.size for value in self.params.values()) != meta["parameter_count"]:
            raise ValueError("parameter count mismatch")
        if self.mean.shape != (input_size,) or self.scale.shape != (input_size,) or not np.all(np.isfinite(self.mean)) or np.any(self.scale <= 0):
            raise ValueError("invalid normalization")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")

    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def predict(self, x):
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            return probs / probs.sum(axis=1, keepdims=True)
        return np.maximum(0.0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    def save(self, path):
        self.validate()
        arrays = {
            **self.params,
            **{"adam_first_" + key: value for key, value in self.first.items()},
            **{"adam_second_" + key: value for key, value in self.second.items()},
            "mean": self.mean, "scale": self.scale,
            "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False)),
        }
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),
                    *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist()]
            if len(names) != len(expected) or set(names) != {n + ".npy" for n in expected}:
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            model.meta = json.loads(str(arrays["metadata"]))
            model.params = {key: arrays[key].copy() for key in PARAMETERS}
            model.first = {key: arrays["adam_first_" + key].copy() for key in PARAMETERS}
            model.second = {key: arrays["adam_second_" + key].copy() for key in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.validate()
        return model


def migrate(model: MLPBackbone, spec, parent_sha256=None):
    """Explicit lossless migration, including switching between derived feature views.

    Base-column weights/Adam moments are copied verbatim. Old derived columns are
    dropped and new ones start at zero, so the migrated model is identical to the
    parent on the shared base representation at the moment of migration.
    """
    base_paths = list(spec.get("base_paths", model.meta["paths"]))
    base_dim = len(base_paths)
    derived_paths = list(spec.get("derived_paths", []))
    derived_dim = len(derived_paths)
    target_paths = base_paths + derived_paths
    if list(model.meta["paths"]) == target_paths:
        return model
    old_base_dim = min(len(model.meta["paths"]), base_dim)
    old_hidden = model.meta["hidden"]
    output = len(model.meta["classes"]) if model.meta["task"] == "classification" else 1
    new = copy.deepcopy(model)
    for group_name in ("params", "first", "second"):
        group = getattr(new, group_name)
        w1 = np.zeros((base_dim + derived_dim, old_hidden))
        w1[:old_base_dim, :] = group["w1"][:old_base_dim, :]
        b1 = np.array(group["b1"][:old_hidden], copy=True)
        w2 = np.array(group["w2"][:old_hidden, :], copy=True)
        b2 = np.array(group["b2"], copy=True)
        group.update(w1=w1, b1=b1, w2=w2, b2=b2)
    derived_mean = np.asarray(spec.get("derived_mean", []), dtype=np.float64)
    derived_scale = np.asarray(spec.get("derived_scale", []), dtype=np.float64)
    base_mean = np.zeros(base_dim)
    base_scale = np.ones(base_dim)
    base_mean[:old_base_dim] = model.mean[:old_base_dim]
    base_scale[:old_base_dim] = model.scale[:old_base_dim]
    new.mean = np.concatenate([base_mean, derived_mean])
    new.scale = np.where(np.concatenate([base_scale, derived_scale]) <= 0, 1.0,
                         np.concatenate([base_scale, derived_scale]))
    new.meta["paths"] = target_paths
    new.meta["parameter_count"] = int(sum(value.size for value in new.params.values()))
    new.meta["migrated_from_sha256"] = parent_sha256 or ""
    new.meta["migration"] = {"added_input_columns": derived_dim, "init": "zero", "hidden": old_hidden,
                             "base_columns_preserved": old_base_dim}
    new.validate()
    # Compare at the zero-normalized point of both models: all derived columns
    # normalize to zero there, so dropping/adding views cannot change the output.
    parent_probe = np.tile(model.mean.reshape(1, -1), (1, 1))
    migrated_probe = np.tile(new.mean.reshape(1, -1), (1, 1))
    base_pred = model.predict(parent_probe)
    if not np.allclose(base_pred, new.predict(migrated_probe), atol=1e-6):
        raise ValueError("migration changed parent output")
    return new


def shrink_backbone(model, gamma):
    """Elastic weight shrinkage for overfit parents; Adam moments decay with the weights."""
    gamma = float(gamma)
    if not 0.05 <= gamma <= 1.0:
        return model
    for name in ("w1", "w2"):
        model.params[name] *= gamma
        model.first[name] *= gamma
        model.second[name] *= gamma ** 2
    model.validate()
    model.meta["weight_shrinkage_gamma"] = gamma
    return model


def _snapshot(model):
    return {
        "params": {key: value.copy() for key, value in model.params.items()},
        "first": {key: value.copy() for key, value in model.first.items()},
        "second": {key: value.copy() for key, value in model.second.items()},
        "steps": model.meta["steps"], "epochs": model.meta["epochs"],
    }


def _restore(model, snapshot):
    model.params = {key: value.copy() for key, value in snapshot["params"].items()}
    model.first = {key: value.copy() for key, value in snapshot["first"].items()}
    model.second = {key: value.copy() for key, value in snapshot["second"].items()}
    model.meta["steps"] = snapshot["steps"]
    model.meta["epochs"] = snapshot["epochs"]


def _holdout_metric(model, normalized_x, y, task, index):
    """Evaluate on already-normalized features (avoids double normalization)."""
    if len(index) == 0:
        return 0.0
    _, logits = model.forward(normalized_x[index])
    if task == "regression":
        prediction = np.maximum(0.0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
        return float(np.sqrt(np.mean((prediction - y[index]) ** 2)))
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    labels = probabilities.argmax(axis=1)
    target = y[index].astype(int)
    f1_values = []
    for class_index in np.unique(np.concatenate([target, labels])):
        tp = float(((labels == class_index) & (target == class_index)).sum())
        fp = float(((labels == class_index) & (target != class_index)).sum())
        fn = float(((labels != class_index) & (target == class_index)).sum())
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1_values))


def train_backbone(model, dataset, spec, cfg, seed):
    """Enhanced Adam training with L2, cosine LR, weighted/asymmetric loss and early stop."""
    from .contracts import group_folds

    x_raw = primitives.transform_features(dataset.x, spec)
    x = model.normalized(x_raw)
    y = dataset.y
    meta = model.meta
    target = y if meta["task"] == "classification" else (y - meta["target_mean"]) / meta["target_scale"]
    folds = group_folds(dataset.groups, n_splits=5, seed=seed)
    holdout = np.where(folds == 0)[0]
    train_index = np.where(folds != 0)[0]
    if len(holdout) == 0 or len(train_index) < 2:
        holdout, train_index = np.array([], dtype=int), np.arange(len(x))
        split = max(1, int(0.85 * len(x)))
        order = np.random.default_rng(seed).permutation(len(x))
        train_index, holdout = order[:split], order[split:]
    epochs = int(cfg.get("epochs", 3))
    lr_base = float(cfg.get("lr", 0.001))
    l2 = float(cfg.get("l2", 1e-4))
    loss_name = str(cfg.get("loss", "mse"))
    class_weight_power = float(cfg.get("class_weight", 0.0))
    if meta["task"] == "classification" and cfg.get("oversample"):
        counts = np.bincount(y[train_index].astype(int), minlength=len(meta["classes"]))
        target_count = min(int(counts.max()) * 2, max(3, int(counts.max())))
        repeated = []
        for class_id in range(len(counts)):
            members = train_index[y[train_index].astype(int) == class_id]
            if len(members) == 0:
                continue
            repeats = max(1, target_count // len(members))
            repeated.extend([members] * repeats)
        train_index = np.concatenate(repeated).astype(int)
    batch_size = max(8, min(meta["batch_size"], len(train_index)))
    if meta["task"] == "classification":
        counts = np.bincount(y[train_index].astype(int), minlength=len(meta["classes"])).astype(float)
        counts[counts == 0] = 1
        beta = float(cfg.get("effective_beta", 0.0))
        if 0 < beta < 1:
            weights = (1.0 - beta) / (1.0 - beta ** counts)
        else:
            weights = (1.0 / counts) ** class_weight_power
        weights = weights / weights.mean()
    else:
        weights = None
    best_snapshot, best_score, stale = None, float("inf"), 0
    for epoch in range(epochs):
        progress = epoch / max(1, epochs - 1)
        lr = lr_base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), epoch]))
        order = train_index[rng.permutation(len(train_index))]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = x[indices]
            expected = target[indices]
            hidden, logits = model.forward(batch)
            if meta["task"] == "classification":
                shifted = logits - logits.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                onehot = np.zeros_like(probabilities)
                onehot[np.arange(len(indices)), expected.astype(int)] = 1.0
                onehot = 0.98 * onehot + 0.02 / onehot.shape[1]
                sample_weights = weights[expected.astype(int)] if weights is not None else 1.0
                delta = (probabilities - onehot) * np.asarray(sample_weights)[:, None]
                delta /= len(indices)
            else:
                error = logits[:, 0] - expected
                sample_weights = np.ones_like(error)
                if loss_name == "asym":
                    sample_weights = np.where(error < 0, float(cfg.get("asym_weight", 2.0)), 1.0)
                delta = (2.0 * error * sample_weights / len(indices))[:, None]
                if loss_name == "huber":
                    delta = np.clip(delta, -0.5, 0.5)
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            gradients["w1"] += l2 * model.params["w1"]
            gradients["w2"] += l2 * model.params["w2"]
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            meta["steps"] += 1
            for name, gradient in gradients.items():
                gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * gradient
                model.second[name] = 0.999 * model.second[name] + 0.001 * gradient ** 2
                first = model.first[name] / (1 - 0.9 ** meta["steps"])
                second = model.second[name] / (1 - 0.999 ** meta["steps"])
                model.params[name] -= lr * first / (np.sqrt(second) + 1e-8)
        meta["epochs"] += 1
        model.validate()
        score = _holdout_metric(model, x, y if meta["task"] == "classification" else y, meta["task"], holdout)
        score = score if meta["task"] == "regression" else -score
        if score < best_score - 1e-9:
            best_score, best_snapshot, stale = score, _snapshot(model), 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_snapshot is not None:
        _restore(model, best_snapshot)
    model.validate()
    return {"epochs_trained": model.meta["epochs"], "holdout_score": float(-best_score if meta["task"] == "classification" else best_score)}


class CEPPool:
    """Up to three small experts fitted on train residual / probability."""

    def __init__(self, task, classes, alpha=0.3, feature_mode="full", base_dim=0):
        self.task = task
        self.classes = list(classes)
        self.alpha = float(alpha)
        self.feature_mode = str(feature_mode)
        self.base_dim = int(base_dim)
        self.experts = []
        self.feature_mean = None
        self.feature_scale = None
        self.views = []
        self.view_weights = []
        self.logit_bias = None
        self.residual_bias = 0.0
        self.asym_weight = 1.0

    def _features(self, model, x_aug):
        if self.feature_mode == "derived" and self.base_dim > 0:
            return np.asarray(x_aug[:, self.base_dim:], dtype=np.float64)
        normalized = model.normalized(x_aug)
        hidden, _ = model.forward(normalized)
        return np.concatenate([normalized, np.tanh(hidden)], axis=1)

    def _view_features(self, raw_x, view_spec):
        augmented = primitives.transform_features(raw_x, view_spec)
        return augmented[:, len(view_spec.get("base_paths", [])):]

    @staticmethod
    def _logistic_proba(coef, intercept, classes, features):
        raw = np.asarray(features) @ np.asarray(coef).T + np.asarray(intercept)
        raw = raw - raw.max(axis=1, keepdims=True)
        probabilities = np.exp(raw)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def _fit_views(self, dataset, views, view_weights, logistic_c, logit_bias, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from .contracts import group_folds
        self.base_dim = len(dataset.paths)
        weights = np.asarray(view_weights or [1.0 / len(views)] * len(views), dtype=float)
        self.view_weights = weights / max(1e-12, weights.sum())
        for position, profile in enumerate(views):
            view_spec = primitives.fit_feature_spec(dataset, profile)
            primitives.attach_paths(view_spec, dataset.paths)
            features = self._view_features(dataset.x, view_spec)
            mean = features.mean(axis=0)
            scale = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
            z = np.clip((features - mean) / scale, -60, 60)
            estimator = LogisticRegression(max_iter=1500, C=float(logistic_c),
                                           class_weight="balanced" if position == 0 else None,
                                           random_state=seed)
            estimator.fit(z, dataset.y.astype(int))
            self.views.append({
                "spec": view_spec, "feature_mean": mean, "feature_scale": scale,
                "coef": np.asarray(estimator.coef_).copy(),
                "intercept": np.asarray(estimator.intercept_).copy(),
                "classes": np.asarray(estimator.classes_, dtype=int).tolist(),
            })
        if logit_bias:
            folds = group_folds(dataset.groups, n_splits=4, seed=seed)
            oof = np.zeros((len(dataset), len(self.classes)))
            for fold in np.unique(folds):
                train_index, holdout = np.where(folds != fold)[0], np.where(folds == fold)[0]
                if len(holdout) == 0 or len(train_index) < 2:
                    continue
                blended = np.zeros((len(holdout), len(self.classes)))
                for view, weight in zip(self.views, self.view_weights):
                    features = self._view_features(dataset.x[train_index], view["spec"])
                    z = np.clip((features - view["feature_mean"]) / view["feature_scale"], -60, 60)
                    estimator = LogisticRegression(max_iter=1000, C=float(logistic_c),
                                                   class_weight="balanced" if view is self.views[0] else None,
                                                   random_state=seed)
                    estimator.fit(z, dataset.y[train_index].astype(int))
                    holdout_z = np.clip((self._view_features(dataset.x[holdout], view["spec"])
                                         - view["feature_mean"]) / view["feature_scale"], -60, 60)
                    probabilities = self._logistic_proba(estimator.coef_, estimator.intercept_,
                                                         estimator.classes_, holdout_z)
                    ordered = np.zeros_like(blended)
                    ordered[:, np.asarray(estimator.classes_, dtype=int)] = probabilities
                    blended += float(weight) * ordered
                oof[holdout] = blended
            bias = np.zeros(len(self.classes))
            target = dataset.y.astype(int)
            counts = np.bincount(target, minlength=len(self.classes)).astype(float)
            minority_weight = 0.6 + 0.4 * (1.0 - counts / max(1.0, counts.max()))

            def objective(labels):
                per_class = f1_score(target, labels, average=None, zero_division=0)
                return float(np.mean(per_class) + np.dot(minority_weight, per_class))

            for _pass in range(3):
                for class_index in range(len(self.classes)):
                    best_value, best_score = bias[class_index], -1.0
                    for candidate in np.linspace(-2.0, 2.0, 17):
                        trial = bias.copy()
                        trial[class_index] = candidate
                        score = objective((oof + trial).argmax(axis=1))
                        if score > best_score:
                            best_score, best_value = score, candidate
                    bias[class_index] = best_value
            # Class-prior correction from train counts only: lift the minority
            # class that OOF macro-F1 optimization tends to suppress.
            prior = 0.3 * np.log(counts.max() / np.maximum(counts, 1.0))
            self.logit_bias = np.clip(bias + prior, -2.5, 2.5)
        return self

    def _view_probabilities(self, raw_x):
        blended = np.zeros((len(raw_x), len(self.classes)))
        for view, weight in zip(self.views, self.view_weights):
            features = self._view_features(raw_x, view["spec"])
            z = np.clip((features - view["feature_mean"]) / view["feature_scale"], -60, 60)
            probabilities = self._logistic_proba(view["coef"], view["intercept"], view["classes"], z)
            ordered = np.zeros((len(raw_x), len(self.classes)))
            ordered[:, np.asarray(view["classes"], dtype=int)] = probabilities
            blended += float(weight) * ordered
        if self.logit_bias is not None:
            blended = np.log(np.clip(blended, 1e-12, 1.0)) + np.asarray(self.logit_bias)
            blended = np.exp(blended - blended.max(axis=1, keepdims=True))
            blended /= blended.sum(axis=1, keepdims=True)
        return blended

    def fit(self, model, dataset, spec, kinds, seed, alpha=None, feature_mode=None, base_dim=None,
            logistic_c=0.5, views=None, view_weights=None, logit_bias=False,
            asym_weight=1.0, conservative_bias=0.0, class_balanced=True, tree_params=None):
        cfg_trees = dict(tree_params or {})
        if feature_mode:
            self.feature_mode = str(feature_mode)
        if base_dim is not None:
            self.base_dim = int(base_dim)
        self.asym_weight = float(asym_weight)
        self.residual_bias = -abs(float(conservative_bias)) * max(1e-9, float(np.std(dataset.y)))
        if views and self.task == "classification":
            return self._fit_views(dataset, views, view_weights, logistic_c, logit_bias, seed)
        from sklearn.linear_model import LogisticRegression, Ridge
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            # LightGBM 的扩展在导入期 dlopen libgomp.so.1；运行库缺失时抛出的是
            # OSError 而非 ImportError。两种都按「不可用」处理，退化为无树模型的
            # 专家组合，避免一个可选依赖让整轮 trial 全部失败。
            lgb = None
        x_aug = primitives.transform_features(dataset.x, spec)
        features = self._features(model, x_aug)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
        features = np.clip((features - self.feature_mean) / self.feature_scale, -60, 60)
        residual = dataset.y.astype(float) - model.predict(x_aug) if self.task == "regression" else None
        for kind in kinds[:3]:
            if kind in ("ridge", "logistic"):
                if self.task == "classification":
                    estimator = LogisticRegression(max_iter=600, C=float(logistic_c),
                                                   class_weight="balanced" if class_balanced else None,
                                                   random_state=seed)
                    estimator.fit(features, dataset.y.astype(int))
                    self.experts.append({"kind": "logistic", "coef": estimator.coef_.copy(), "intercept": estimator.intercept_.copy(),
                                         "classes": estimator.classes_.astype(int).tolist(), "residual": False})
                else:
                    estimator = Ridge(alpha=5.0, random_state=seed)
                    sample_weight = np.where(residual < 0, self.asym_weight, 1.0)
                    estimator.fit(features, residual, sample_weight=sample_weight)
                    self.experts.append({"kind": "ridge", "coef": np.asarray(estimator.coef_).copy(),
                                         "intercept": float(estimator.intercept_), "residual": True})
            elif kind == "lgbm" and lgb is not None:
                if self.task == "classification":
                    estimator = lgb.LGBMClassifier(n_estimators=60, max_depth=4, num_leaves=15, learning_rate=0.06,
                                                   reg_lambda=2.0, verbosity=-1, random_state=seed, n_jobs=2)
                    estimator.fit(features, dataset.y.astype(int))
                    text = estimator.booster_.model_to_string()
                    self.experts.append({"kind": "lgbm_classifier", "model_text": text, "residual": False})
                else:
                    estimator = lgb.LGBMRegressor(n_estimators=int(cfg_trees.get("n_estimators", 80)),
                                                  max_depth=int(cfg_trees.get("max_depth", 4)),
                                                  num_leaves=int(cfg_trees.get("num_leaves", 15)),
                                                  learning_rate=0.06,
                                                  reg_lambda=float(cfg_trees.get("reg_lambda", 2.0)),
                                                  subsample=float(cfg_trees.get("subsample", 1.0)),
                                                  colsample_bytree=float(cfg_trees.get("colsample_bytree", 1.0)),
                                                  min_child_samples=int(cfg_trees.get("min_child_samples", 20)),
                                                  verbosity=-1, random_state=seed, n_jobs=2)
                    estimator.fit(features, residual,
                                  sample_weight=np.where(residual < 0, self.asym_weight, 1.0))
                    text = estimator.booster_.model_to_string()
                    self.experts.append({"kind": "lgbm_regressor", "model_text": text, "residual": True})
        if alpha is not None:
            self.alpha = float(alpha)
        return self

    def predict(self, model, x_aug, base=None):
        if self.views:
            raw_x = np.asarray(x_aug, dtype=np.float64)[:, :self.base_dim]
            view_prediction = self._view_probabilities(raw_x)
            if base is None:
                base = model.predict(x_aug)
            return (1 - self.alpha) * base + self.alpha * view_prediction
        if base is None:
            base = model.predict(x_aug)
        if not self.experts:
            return base + (self.residual_bias if self.task == "regression" else 0.0)
        features = self._features(model, x_aug)
        features = np.clip((features - self.feature_mean) / self.feature_scale, -60, 60)
        if self.task == "regression":
            correction = np.zeros(len(base))
            weight = 0.0
            for expert in self.experts:
                value = self._expert_values(expert, features)
                correction += value
                weight += 1.0
            correction = correction / max(1.0, weight)
            return np.maximum(0.0, base + self.alpha * correction + self.residual_bias)
        votes = np.zeros_like(base)
        for expert in self.experts:
            values = self._expert_values(expert, features)
            votes += values
        votes /= max(1, len(self.experts))
        return (1 - self.alpha) * base + self.alpha * votes

    def _expert_values(self, expert, features):
        kind = expert["kind"]
        if kind == "ridge":
            return features @ np.asarray(expert["coef"]) + float(expert["intercept"])
        if kind == "logistic":
            raw = features @ np.asarray(expert["coef"]).T + np.asarray(expert["intercept"])
            raw = raw - raw.max(axis=1, keepdims=True)
            probabilities = np.exp(raw)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            ordered = np.zeros((len(features), len(self.classes)))
            ordered[:, np.asarray(expert["classes"], dtype=int)] = probabilities
            return ordered
        if kind in ("lgbm_regressor", "lgbm_classifier"):
            try:
                import lightgbm as lgb
            except (ImportError, OSError) as exc:
                # 这里不能像训练那样静默退化：检查点里已有树专家，跳过它们会
                # 算出与训练时不同的预测。直接报出原因（通常是缺 libgomp.so.1）。
                raise RuntimeError(
                    "checkpoint contains a LightGBM expert but lightgbm cannot be "
                    "imported (%s); install libgomp1 in the runtime image" % exc)
            # 流式 predict 逐行调用本方法：Booster 只反序列化一次并缓存于内存，
            # 缓存键以下划线开头，serialize() 会跳过，不落盘。
            booster = expert.get("_booster")
            if booster is None:
                booster = lgb.Booster(model_str=expert["model_text"])
                expert["_booster"] = booster
            values = booster.predict(features)
            if kind == "lgbm_classifier":
                if values.ndim == 1:
                    values = np.stack([1 - values, values], axis=1)
                ordered = np.zeros((len(features), len(self.classes)))
                width = min(ordered.shape[1], values.shape[1])
                ordered[:, :width] = values[:, :width]
                return ordered
            return values.reshape(-1)
        raise ValueError("unknown expert kind")

    def serialize(self, path):
        arrays = {"feature_mean": self.feature_mean, "feature_scale": self.feature_scale,
                  "feature_mode": np.asarray(self.feature_mode), "base_dim": np.asarray(self.base_dim),
                  "view_count": np.asarray(len(self.views)),
                  "view_weights": np.asarray(self.view_weights),
                  "logit_bias": np.asarray(self.logit_bias if self.logit_bias is not None else []),
                  "residual_bias": np.asarray(self.residual_bias),
                  "asym_weight": np.asarray(self.asym_weight)}
        for index, view in enumerate(self.views):
            for key, value in view.items():
                if key == "spec":
                    arrays["view_%d_spec" % index] = np.asarray(json.dumps(value, sort_keys=True, allow_nan=False))
                else:
                    arrays["view_%d_%s" % (index, key)] = np.asarray(value)
        for index, expert in enumerate(self.experts):
            for key, value in expert.items():
                if key.startswith("_"):
                    continue  # 内存缓存（如 _booster）绝不落盘
                arrays["expert_%d_%s" % (index, key)] = np.asarray(value)
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                value = np.asarray(arrays[name])
                if value.dtype.kind == "U" and value.ndim == 0:
                    value = np.asarray(str(value))
                elif value.dtype.kind not in ("f", "i", "U"):
                    value = np.asarray(value, dtype=np.float64)
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, value, allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path, task, classes, alpha, spec=None):
        pool = cls(task, classes, alpha)
        if not path or not Path(path).exists():
            return pool
        with np.load(path, allow_pickle=False) as arrays:
            pool.feature_mean = arrays["feature_mean"].copy()
            pool.feature_scale = arrays["feature_scale"].copy()
            if "feature_mode" in arrays:
                pool.feature_mode = str(arrays["feature_mode"])
            if "base_dim" in arrays:
                pool.base_dim = int(arrays["base_dim"])
            if "view_count" in arrays and int(arrays["view_count"]) > 0:
                pool.view_weights = np.asarray(arrays["view_weights"], dtype=float)
                if "logit_bias" in arrays and arrays["logit_bias"].size:
                    pool.logit_bias = np.asarray(arrays["logit_bias"], dtype=float)
                for index in range(int(arrays["view_count"])):
                    view = {"spec": json.loads(str(arrays["view_%d_spec" % index]))}
                    for key in ("feature_mean", "feature_scale", "coef", "intercept", "classes"):
                        value = arrays["view_%d_%s" % (index, key)]
                        view[key] = (np.asarray(value, dtype=int) if key == "classes"
                                     else np.asarray(value, dtype=float))
                    pool.views.append(view)
            # residual_bias / asym_weight 属于回归池（保守偏置），与视图无关，必须总是恢复。
            if "residual_bias" in arrays:
                pool.residual_bias = float(arrays["residual_bias"])
            if "asym_weight" in arrays:
                pool.asym_weight = float(arrays["asym_weight"])
            groups = {}
            for name in arrays.files:
                if not name.startswith("expert_"):
                    continue
                parts = name[len("expert_"):].split("_", 1)
                index = int(parts[0])
                groups.setdefault(index, {})[parts[1]] = arrays[name].copy()
        for index in sorted(groups):
            item = {}
            for key, value in groups[index].items():
                if value.dtype.kind == "U":
                    item[key] = str(value) if value.ndim == 0 else [str(v) for v in value.tolist()]
                elif key == "classes":
                    item[key] = [int(v) for v in value.tolist()]
                else:
                    item[key] = value.astype(float) if value.ndim else float(value)
            pool.experts.append(item)
        return pool


def parent_checkpoint_path(rnd, work_root=None):
    """Resolve the declared parent checkpoint dynamically; never scan /app."""
    declared = list(getattr(rnd.parent.descriptor, "get", lambda *_: [])("checkpoint_paths", []))
    if not declared:
        declared = list(rnd.parent.descriptor.get("checkpoint_paths", []))
    preferred = [name for name in declared if name.endswith(".npz")]
    if not preferred:
        preferred = declared
    if not preferred:
        raise ValueError("parent artifact declares no checkpoint")
    path = rnd.parent.path(preferred[0])
    return Path(path), preferred[0]


def load_bundle(model_dir, spec=None):
    model_dir = Path(model_dir)
    artifact = os.environ.get("CONTEST_ARTIFACT_PATH", "")
    checkpoint = Path(artifact) if artifact and Path(artifact).exists() else None
    if checkpoint is None:
        candidates = sorted(model_dir.glob("checkpoint*.npz"))
        if not candidates:
            raise ValueError("no checkpoint in model directory")
        checkpoint = candidates[0]
    spec_path = model_dir / "model_spec.json"
    if spec_path.exists():
        meta = json.loads(spec_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    if spec is None:
        spec = meta.get("feature_spec")
    if not isinstance(spec, dict):
        raise ValueError("model_spec.json missing feature_spec")
    backbone = MLPBackbone.load(checkpoint)
    pool_path = model_dir / "cep_bundle.npz"
    pool = CEPPool.load(pool_path, backbone.meta["task"], backbone.meta["classes"],
                        meta.get("alpha", 0.0), spec)
    return backbone, pool, meta


_BUNDLE_CACHE = {}


def predict_one(vector, model_dir):
    key = (str(model_dir), os.environ.get("CONTEST_ARTIFACT_PATH", ""))
    cached = _BUNDLE_CACHE.get(key)
    if cached is None:
        cached = load_bundle(model_dir)
        _BUNDLE_CACHE[key] = cached
    backbone, pool, meta = cached
    x = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    x_aug = primitives.transform_features(x, meta["feature_spec"])
    if x_aug.shape[1] != len(backbone.meta["paths"]):
        raise ValueError("predict feature schema mismatch")
    base = backbone.predict(x_aug)
    final = pool.predict(backbone, x_aug, base)
    if backbone.meta["task"] == "regression" and meta.get("feature_spec", {}).get("target_transform") == "sqrt":
        final = np.square(np.maximum(np.asarray(final, dtype=np.float64), 0.0))
    if backbone.meta["task"] == "classification":
        final = np.clip(final, 0, 1)
        final = final / final.sum(axis=1, keepdims=True)
        classes = backbone.meta["classes"]
        scores = tie_break_scores(final[0].tolist())
        return [{"label": str(label), "score": float(score)} for label, score in zip(classes, scores)]
    return {"value": float(max(0.0, final[0]))}


def tie_break_scores(scores):
    """分类分数平局打破：严格保证 [0,1] 区间且唯一最大值。

    先整体截断到 [0,1]；保持最高分不变（全 0 时抬到 1e-6）；
    所有与最高分接近平局的次高类减去 1e-6 并截断。
    旧逻辑 `max(top0, top1 + 1e-6)` 在 top1≈1.0 时会产生 >1 的非法分数。
    """
    scores = [min(1.0, max(0.0, float(s))) for s in scores]
    if not scores:
        return scores
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = order[0]
    if scores[top] <= 0.0:
        scores[top] = 1e-6
    for index in order[1:]:
        if scores[top] - scores[index] < 1e-6:
            scores[index] = max(0.0, min(1.0, scores[index] - 1e-6))
    return scores


def checkpoint_sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
