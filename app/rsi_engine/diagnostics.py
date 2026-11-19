"""Leakage-free diagnostics: regime drift, frozen baseline, loss attribution."""
import hashlib
import math
from collections import defaultdict

import numpy as np

from . import primitives


def classification_metrics(y_true, y_prob, classes):
    labels = np.asarray(y_prob).argmax(axis=1)
    y_true = np.asarray(y_true, dtype=int)
    f1, support, confusion = [], [], defaultdict(int)
    for index, name in enumerate(classes):
        tp = int(((labels == index) & (y_true == index)).sum())
        fp = int(((labels == index) & (y_true != index)).sum())
        fn = int(((labels != index) & (y_true == index)).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
        support.append(int((y_true == index).sum()))
        for predicted in np.unique(labels[y_true == index]):
            if int(predicted) != index:
                confusion["%s->%s" % (name, classes[int(predicted)])] += int((labels[y_true == index] == predicted).sum())
    order = np.argsort(f1)
    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.mean(labels == y_true)),
        "per_class_f1": {name: round(float(value), 4) for name, value in zip(classes, f1)},
        "weakest_class": classes[int(order[0])] if len(order) else "",
        "support": dict(zip(classes, support)),
        "top_confusions": sorted(confusion.items(), key=lambda item: -item[1])[:3],
    }


def regression_metrics(y_true, y_pred, dataset_key="C"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    diff = y_pred - y_true
    clipped = np.clip(diff, -100.0, 100.0)
    nasa = float(np.sum(np.where(clipped < 0, np.exp(-clipped / 13.0) - 1.0,
                                 np.exp(clipped / 10.0) - 1.0))) if dataset_key == "C" else 0.0
    result = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "overestimation_rate": float(np.mean(diff > 0)),
        "nasa_score": nasa,
    }
    if dataset_key == "B":
        result["early_rmse"] = float(np.sqrt(np.mean(diff[y_true > 300] ** 2))) if np.any(y_true > 300) else 0.0
        result["late_rmse"] = float(np.sqrt(np.mean(diff[y_true <= 100] ** 2))) if np.any(y_true <= 100) else 0.0
    return result


def evaluate(y_true, prediction, task, classes, dataset_key):
    if task == "classification":
        return classification_metrics(y_true, prediction, classes)
    return regression_metrics(y_true, prediction, dataset_key)


def primary(metrics, task):
    return float(metrics["macro_f1"] if task == "classification" else metrics["rmse"])


def _block_macro_f1(y_true, labels, classes):
    per_class = []
    for class_index in range(len(classes)):
        tp = int(((labels == class_index) & (y_true == class_index)).sum())
        fp = int(((labels == class_index) & (y_true != class_index)).sum())
        fn = int(((labels != class_index) & (y_true == class_index)).sum())
        if tp + fp + fn == 0:
            continue
        denominator = 2 * tp + fp + fn
        per_class.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(per_class)) if per_class else 0.0


def block_losses(y_true, prediction, groups, task, classes):
    """Block-level loss on the PRIMARY metric: 1-macro-F1 (A) or MSE (B/C)."""
    losses = {}
    buckets = defaultdict(list)
    for index, group in enumerate(groups):
        buckets[str(group)].append(index)
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    for group, indices in buckets.items():
        indices = np.asarray(indices, dtype=int)
        if task == "classification":
            target = y_true[indices].astype(int)
            probability = np.asarray(prediction)[indices]
            labels = probability.argmax(axis=1) if probability.ndim == 2 else probability.astype(int)
            loss = 1.0 - _block_macro_f1(target, labels, classes)
        else:
            loss = float(np.mean((np.clip(prediction[indices], 0, None) - y_true[indices].astype(float)) ** 2))
        losses[group] = float(loss)
    return losses


def block_weights(groups):
    """Per-block sample share w_b = n_b / N for the weighted block e-process."""
    buckets = defaultdict(int)
    for group in groups:
        buckets[str(group)] += 1
    total = max(1, sum(buckets.values()))
    return {group: count / total for group, count in buckets.items()}


def _prediction(model, pool, spec, dataset):
    x = primitives.features_for_model(dataset.x, spec, model.meta["paths"])
    base = model.predict(x)
    final = pool.predict(model, x, base)
    if spec.get("target_transform") == "sqrt":
        final = np.square(np.maximum(np.asarray(final, dtype=np.float64), 0.0))
    return final


def _block_feature_drift(train, validation, max_blocks=40):
    scale = np.where(train.x.std(axis=0) < 1e-8, 1.0, train.x.std(axis=0))
    train_blocks, val_blocks = defaultdict(list), defaultdict(list)
    for index, group in enumerate(train.groups):
        train_blocks[group].append(index)
    for index, group in enumerate(validation.groups):
        val_blocks[group].append(index)
    deltas = []
    for group in sorted(set(train_blocks) & set(val_blocks))[:max_blocks]:
        left = train.x[train_blocks[group]].mean(axis=0)
        right = validation.x[val_blocks[group]].mean(axis=0)
        deltas.append(float(np.mean(np.abs((right - left) / scale))))
    return float(np.mean(deltas)) if deltas else 0.0


def _channel_shift(train, validation, spec):
    """Max standardized channel shift, A: 16×8 physical channels, C: regime residuals."""
    left, right = train.x, validation.x
    if train.dataset_key == "A" and left.shape[1] >= 128:
        a = left[:, :128].reshape(len(left), 16, 8)
        b = right[:, :128].reshape(len(right), 16, 8)
        scale = np.where(a.std(axis=0) < 1e-8, 1.0, a.std(axis=0))
        shifts = np.abs((b.mean(axis=0) - a.mean(axis=0)) / scale)
    elif train.dataset_key == "C" and spec.get("centers"):
        centers = np.asarray(spec["centers"])
        sensors = np.asarray(spec.get("sensor_indices", []), dtype=int)
        if len(sensors) == 0:
            return {"max_channel_shift": 0.0, "channel_shift_p90": 0.0}
        shifts = []
        for center, mean, scale in zip(centers, spec["cluster_mean"], spec["cluster_scale"]):
            scale = np.where(np.asarray(scale) < 1e-8, 1.0, np.asarray(scale))
            for dataset in (train, validation):
                z = primitives.setting_standardized(dataset.x, spec)
                # Nearest-centre assignment over all rows of this regime, not
                # just the single closest sample (which yields a noise reading).
                assignment = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
                members = dataset.x[:, sensors][assignment == list(centers.tolist()).index(center.tolist())] \
                    if len(centers) else dataset.x[:, sensors][:0]
                if len(members) < 2:
                    distance = ((z - center) ** 2).sum(axis=1)
                    members = dataset.x[:, sensors][np.argsort(distance)[:max(2, len(dataset.x) // 50)]]
                shifts.append(np.abs((members.mean(axis=0) - np.asarray(mean)) / scale))
        shifts = np.stack(shifts, axis=0)
        shifts = np.abs(shifts[::2].mean(axis=0) - shifts[1::2].mean(axis=0))
    else:
        scale = np.where(left.std(axis=0) < 1e-8, 1.0, left.std(axis=0))
        shifts = np.abs((right.mean(axis=0) - left.mean(axis=0)) / scale)
    return {"max_channel_shift": round(float(np.max(shifts)), 4),
            "channel_shift_p90": round(float(np.quantile(shifts, 0.9)), 4)}


def compute_report(train, validation, backbone, pool, spec, round_index, state, parent_sha256=""):
    val_pred = _prediction(backbone, pool, spec, validation)
    train_pred = _prediction(backbone, pool, spec, train)
    metrics = evaluate(validation.y, val_pred, validation.task, validation.classes, validation.dataset_key)
    train_metrics = evaluate(train.y, train_pred, train.task, train.classes, train.dataset_key)
    report = {
        "round_index": int(round_index),
        "task": validation.task,
        "dataset_key": validation.dataset_key,
        "drift": {"covariate_shift_score": round(_block_feature_drift(train, validation), 4),
                  **_channel_shift(train, validation, spec)},
        "min_class_f1": round(float(min(metrics.get("per_class_f1", {}).values())), 4)
        if metrics.get("per_class_f1") else None,
        "val": metrics,
        "train": {"rmse": train_metrics.get("rmse"), "macro_f1": train_metrics.get("macro_f1")},
        "overfit_gap": round(float(train_metrics.get("rmse", 0) - metrics.get("rmse", 0)), 4)
        if validation.task == "regression" else round(float(metrics.get("macro_f1", 0) - train_metrics.get("macro_f1", 0)), 4),
    }
    if validation.dataset_key == "C" and spec.get("centers"):
        centers = np.asarray(spec["centers"])
        def share(dataset):
            z = primitives.setting_standardized(dataset.x, spec)
            distance = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distance.argmin(axis=1)
            counts = np.bincount(labels, minlength=len(centers)).astype(float)
            return (counts / max(1, len(dataset))).tolist()
        report["drift"]["regime_train_share"] = [round(v, 3) for v in share(train)]
        report["drift"]["regime_val_share"] = [round(v, 3) for v in share(validation)]
    if validation.dataset_key == "B":
        report["overfit_gap"] = round(float(metrics.get("rmse", 0) - train_metrics.get("rmse", 0)), 4)
    drift_score = report["drift"]["covariate_shift_score"]
    channel_shift = report["drift"]["max_channel_shift"]
    overfit_limit = 15 if validation.task == "regression" else -0.05
    min_f1 = report.get("min_class_f1")
    if min_f1 is not None and min_f1 < 0.3:
        regime = "severe_class_drift"
    elif validation.dataset_key == "C" and metrics.get("overestimation_rate", 0) > 0.55:
        regime = "asymmetric_overestimation"
    elif report["overfit_gap"] > overfit_limit:
        regime = "overfitting"
    elif drift_score > 1.5 or channel_shift > 4.0:
        regime = "extreme_covariate_shift"
    elif drift_score > 0.35 or channel_shift > 1.5:
        regime = "regime_covariate_shift"
    else:
        regime = "stable"
    report["root_cause"] = {
        "regime": regime,
        "max_channel_shift": channel_shift,
        "min_class_f1": min_f1,
        "overfit_gap": report["overfit_gap"],
        "overestimation_rate": metrics.get("overestimation_rate"),
        "evidence": ("min_class_f1<0.3" if min_f1 is not None and min_f1 < 0.3 else
                     "overestimation_rate>0.55" if metrics.get("overestimation_rate", 0) > 0.55 else
                     "overfit_gap" if report["overfit_gap"] > overfit_limit else
                     "max_channel_shift/covariate_shift"),
    }
    report["drift_regime"] = regime
    report["bottleneck"] = report.get("val", {}).get("weakest_class") if validation.task == "classification" else (
        "overestimation" if metrics.get("overestimation_rate", 0) > 0.5 else "rmse")
    anchor = state.get("frozen_anchor")
    if anchor is None and int(round_index) == 1:
        digest = parent_sha256 or hashlib.sha256(b"unknown").hexdigest()
        block = block_losses(validation.y, val_pred, validation.groups, validation.task, validation.classes)
        anchor = {
            "checkpoint_sha256": digest,
            "metrics": metrics,
            "block_residual_mean": {key: round(value, 4) for key, value in list(block.items())[:40]},
        }
        state["frozen_anchor"] = anchor
        report["frozen_anchor"] = {k: anchor[k] for k in ("checkpoint_sha256", "metrics")}
    elif anchor is not None:
        report["frozen_anchor"] = {"checkpoint_sha256": anchor.get("checkpoint_sha256"),
                                   "metrics": anchor.get("metrics")}
    return report
