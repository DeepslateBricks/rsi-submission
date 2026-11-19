"""Workspace v2 entry: controlled dual-loop RSI over the official CPU MLP."""
import copy
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from contest_harness import Round

from rsi_engine import aide_search, diagnostics, gatekeeper, primitives
from rsi_engine.cep_model import (CEPPool, MLPBackbone, checkpoint_sha256, migrate,
                                  parent_checkpoint_path, predict_one, train_backbone)
from rsi_engine.contracts import Dataset, group_folds, load_rows, sample_vector, validate_no_leak
from rsi_engine.router_skill import PilotRouter, context_vector, evoguard_update
from rsi_engine.state_store import StateStore

TASKS = ["classification", "regression"]
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

# Local-development mounts only. The platform never sets these variables.
if os.environ.get("RSI_MODEL_ROOT"):
    import contest_harness.predict as _predict_module
    _predict_module.model_dir = lambda: Path(os.environ["RSI_MODEL_ROOT"])
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rsi.submission")


def _declared_parent_files(rnd):
    descriptor = rnd.parent.descriptor
    return list(descriptor.get("checkpoint_paths", [])) + list(descriptor.get("code_paths", []))


def _load_parent_spec(rnd, train_dataset):
    declared = _declared_parent_files(rnd)
    if "model_spec.json" in declared:
        try:
            spec = json.loads(rnd.parent.path("model_spec.json").read_text(encoding="utf-8"))
            if isinstance(spec.get("feature_spec"), dict):
                feature_spec = spec["feature_spec"]
                # Carry CEP blend state into the feature spec so diagnostics
                # evaluate the parent with the same alpha as predict does.
                feature_spec["_cep_alpha"] = spec.get("alpha", 0.2)
                feature_spec["_cep_feature_mode"] = spec.get("cep_feature_mode", "full")
                feature_spec["_cep_base_dim"] = spec.get("cep_base_dim", len(train_dataset.paths))
                spec = feature_spec
            spec.setdefault("base_paths", train_dataset.paths)
            return spec
        except Exception as exc:
            log.warning("parent model_spec unreadable: %s", exc)
    spec = primitives.fit_feature_spec(train_dataset)
    return primitives.attach_paths(spec, train_dataset.paths)


def _load_parent_pool(rnd, backbone, spec, state):
    declared = _declared_parent_files(rnd)
    if "cep_bundle.npz" in declared:
        try:
            path = rnd.parent.path("cep_bundle.npz")
            return CEPPool.load(path, backbone.meta["task"], backbone.meta["classes"],
                                spec.get("_cep_alpha", spec.get("alpha", 0.2)), spec)
        except Exception as exc:
            log.warning("parent CEP bundle unreadable: %s", exc)
    return CEPPool(backbone.meta["task"], backbone.meta["classes"], alpha=spec.get("alpha", 0.2))


def _augmented_prediction(model, pool, spec, dataset):
    x = primitives.features_for_model(dataset.x, spec, model.meta["paths"])
    base = model.predict(x)
    final = pool.predict(model, x, base)
    return _invert_prediction(final, spec)


def _invert_prediction(prediction, spec):
    """sqrt(y) 目标空间逆变换：模型头输出 sqrt 空间值，评估/对比统一回到原始空间。"""
    prediction = np.asarray(prediction, dtype=np.float64)
    if spec.get("target_transform") == "sqrt" and model_task_is_regression(spec):
        return np.square(np.maximum(prediction, 0.0))
    return prediction


def model_task_is_regression(spec):
    """sqrt 逆变换仅对回归有意义；spec 本身不带 task，通过 dataset_key 判定。"""
    return spec.get("dataset_key") in ("B", "C") or spec.get("task") == "regression"


def _target_space_y(y, transform):
    y = np.asarray(y, dtype=np.float64)
    if transform == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
    return y


def _subset_dataset(dataset, index):
    return Dataset(task=dataset.task, dataset_key=dataset.dataset_key, paths=dataset.paths,
                   x=dataset.x[index], y=dataset.y[index],
                   ids=[dataset.ids[i] for i in index], groups=[dataset.groups[i] for i in index],
                   metas=[dataset.metas[i] for i in index], classes=dataset.classes)


def _pool_kinds(pool):
    kinds = []
    for expert in pool.experts:
        kind = expert.get("kind", "")
        kinds.append("lgbm" if kind.startswith("lgbm") else kind)
    return kinds or ["ridge"]


def _oof_anchor(dataset, spec, base_model, recipe, task, classes, dataset_key, n_splits=3):
    """Group-CV OOF anchor metrics for a full pipeline recipe (train-only, leakage-free).

    逐折从零初始化重训完整管线（骨干 + CEP 专家），候选与父配方使用完全相同的协议
    （同一折划分、epochs≤4、树≤150、同一 spec 视图），因此锚点对比度量的是配方
    的组内泛化能力：既不带父检查点对全量 train 的 in-sample 泄漏，也不会因
    in-sample 过拟合误差误杀去过拟合候选。返回整体主指标（OOF 预测 vs 原始 y）。
    """
    transform = str(recipe.get("target_transform", "") or "")
    folds = group_folds(dataset.groups, n_splits=n_splits, seed=42)
    is_classification = task == "classification"
    oof = np.full((len(dataset), len(classes)), np.nan) if is_classification else np.full(len(dataset), np.nan)
    budget = {
        "epochs": max(1, min(4, int(recipe.get("epochs", 3)))),
        "lr": float(recipe.get("lr", 0.001)), "l2": float(recipe.get("l2", 1e-4)),
        "loss": recipe.get("loss", "mse"), "asym_weight": float(recipe.get("asym_weight", 2.0)),
    }
    tree_params = dict(recipe.get("tree_params") or {})
    if tree_params.get("n_estimators"):
        tree_params["n_estimators"] = min(int(tree_params["n_estimators"]), 150)
    kinds = list(recipe.get("experts") or ["ridge"])[:3]
    base_dim = len(spec.get("base_paths", []))
    seed = int(recipe.get("seed", 42))
    hidden = int(base_model.meta.get("hidden", 32))
    for fold in np.unique(folds):
        holdout = np.where(folds == fold)[0]
        train_index = np.where(folds != fold)[0]
        if len(holdout) == 0 or len(train_index) < 2:
            continue
        try:
            fold_train_raw = _subset_dataset(dataset, train_index)
            fold_train = Dataset(**{**fold_train_raw.__dict__,
                                    "y": _target_space_y(fold_train_raw.y, transform)})
            fold_model = MLPBackbone(task, dataset.paths, classes if is_classification else [],
                                     fold_train.x, fold_train.y, hidden=hidden,
                                     seed=seed + int(fold))
            # 迁移到 spec 视图（衍生特征零初始化），与真实 trial 管线的列结构一致。
            fold_model = migrate(fold_model, spec, None)
            train_backbone(fold_model, fold_train, spec, budget, seed)
            fold_pool = CEPPool(task, classes, alpha=float(recipe.get("alpha", 0.2)),
                                feature_mode=str(recipe.get("feature_mode", "full")),
                                base_dim=base_dim)
            fold_pool.fit(fold_model, fold_train, spec, kinds, seed,
                          alpha=float(recipe.get("alpha", 0.2)),
                          feature_mode=str(recipe.get("feature_mode", "full")),
                          base_dim=base_dim,
                          logistic_c=float(recipe.get("logistic_c", 0.5)),
                          views=recipe.get("views"), view_weights=recipe.get("view_weights"),
                          logit_bias=bool(recipe.get("logit_bias", False)),
                          asym_weight=float(recipe.get("asym_weight", 1.0)),
                          conservative_bias=float(recipe.get("conservative_bias", 0.0)),
                          class_balanced=bool(recipe.get("class_balanced", True)),
                          tree_params=tree_params or None)
            x_hold = primitives.features_for_model(dataset.x[holdout], spec, fold_model.meta["paths"])
            base = fold_model.predict(x_hold)
            pred = _invert_prediction(fold_pool.predict(fold_model, x_hold, base), spec)
            oof[holdout] = pred
        except Exception:
            log.warning("oof anchor fold failed", exc_info=True)
            continue
    valid = ~np.isnan(oof).any(axis=1) if is_classification else ~np.isnan(oof)
    if int(valid.sum()) < 2:
        return None
    return diagnostics.evaluate(dataset.y[valid], oof[valid], task, classes, dataset_key)


def _score(metrics, task):
    value = metrics.get("macro_f1") if task == "classification" else metrics.get("rmse")
    return float(value) if task == "classification" else -float(value)


def _sandbox_guard_ok(metrics, dataset_key, task):
    """沙盒预筛：回归 trial 须满足失衡/NASA 温和护栏才进入 winner 竞争。

    若全部 trial 都不满足则回退全集（与门控 Track-2 同向，提前避免浪费
    OOF 锚点预算在注定被拦的候选上）。
    """
    if task != "regression":
        return True
    over = metrics.get("overestimation_rate")
    if over is not None and float(over) > 0.70:
        return False
    late, early = metrics.get("late_rmse"), metrics.get("early_rmse")
    if late is not None and early is not None and float(early) > 0 \
            and float(late) > float(early) * 4.0:
        return False
    return True


def train(rnd: Round):
    started = time.monotonic()
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    store = StateStore(rnd, work)
    state = store.data
    state["task"] = rnd.task.get("task")
    state["dataset_key"] = rnd.dataset_key
    state["round"] = int(rnd.round_index)
    state["drift_regime"] = state.get("drift_regime", "stable")

    history = rnd.history()
    train_rows = list(rnd.data.training_rows())
    validation_rows = list(rnd.data.validation_rows())
    task = rnd.task.get("task")
    if task not in ("classification", "regression"):
        raise ValueError("task contract is invalid")

    if os.environ.get("RSI_PARENT_ROOT"):
        rnd.parent.root = Path(os.environ["RSI_PARENT_ROOT"])
    parent_path, parent_filename = parent_checkpoint_path(rnd)
    parent_backbone = MLPBackbone.load(parent_path)
    classes = parent_backbone.meta["classes"]
    train_ds = load_rows(train_rows, task, rnd.dataset_key, classes=classes)
    validate_no_leak(train_ds)
    validation_ds = load_rows(validation_rows, task, rnd.dataset_key, classes=classes,
                              expected_paths=train_ds.paths)
    validate_no_leak(validation_ds)

    spec = _load_parent_spec(rnd, train_ds)
    pool = _load_parent_pool(rnd, parent_backbone, spec, state)
    parent_sha = (rnd.parent.checkpoint_sha256 or "").replace("sha256:", "")
    report = diagnostics.compute_report(train_ds, validation_ds, parent_backbone, pool, spec,
                                        int(rnd.round_index), state, parent_sha)
    state["drift_regime"] = report["drift_regime"]
    parent_metrics = report["val"]
    parent_train_metrics = report["train"]
    parent_prediction = _augmented_prediction(parent_backbone, pool, spec, validation_ds)
    parent_losses = diagnostics.block_losses(validation_ds.y, parent_prediction, validation_ds.groups,
                                             task, classes)
    parent_transform = str(spec.get("target_transform", "") or "")
    log.info("round %d report=%s", rnd.round_index, json.dumps(report, ensure_ascii=False)[:900])

    router = PilotRouter(state.get("router"))
    context = context_vector(report, state)
    action = router.select(context)
    proposals, receipt, action = aide_search.propose(rnd, report, state, action, history)
    log.info("router=%s proposals=%d", action, len(proposals))

    deadline = started + (float(rnd.wall_seconds) * 0.72 if getattr(rnd, "wall_seconds", None) else 480)
    trials = []
    trials_root = work / ("trials_r%d" % int(rnd.round_index))
    trials_root.mkdir(exist_ok=True)
    for position, proposal in enumerate(proposals):
        if time.monotonic() > deadline:
            log.warning("trial budget exhausted after %d trials", position)
            break
        trial_dir = trials_root / ("trial_%d" % position)
        trial_dir.mkdir(exist_ok=True)
        started_trial = time.monotonic()
        try:
            params = proposal["params"]
            requested_profile = params.get("profile", "auto")
            if requested_profile not in ("auto", spec.get("profile")):
                trial_spec = primitives.attach_paths(
                    primitives.fit_feature_spec(train_ds, requested_profile), train_ds.paths)
            else:
                trial_spec = dict(spec)
            # 目标空间：候选默认继承父模型空间，也可显式切换（sqrt / 原始）。
            transform = str(params.get("target_transform", trial_spec.get("target_transform", "")) or "")
            if transform in ("none", "raw"):
                transform = ""
            if transform:
                trial_spec["target_transform"] = transform
            else:
                trial_spec.pop("target_transform", None)
            # 关键：从父模型的独立快照迁移。migrate 在路径一致时返回原对象，
            # 若不深拷贝，多个 trial 会共享并依次变异同一个父对象，
            # 导致「trial 评估的权重」与「序列化落盘的权重」不一致。
            model = migrate(copy.deepcopy(parent_backbone), trial_spec, parent_sha)
            train_ds_space = train_ds
            if task == "regression":
                y_space = _target_space_y(train_ds.y, transform)
                train_ds_space = Dataset(**{**train_ds.__dict__, "y": y_space})
                if transform != parent_transform:
                    # 切换目标空间时同步重建模型头归一化统计量。
                    model.meta["target_mean"] = float(y_space.mean())
                    model.meta["target_scale"] = max(float(y_space.std()), 1.0)
            if params.get("shrinkage"):
                from rsi_engine.cep_model import shrink_backbone
                model = shrink_backbone(model, params["shrinkage"])
            training_info = train_backbone(model, train_ds_space, trial_spec, params, proposal["seed"])
            trial_pool = CEPPool(task, classes, alpha=params.get("alpha", 0.2),
                                 feature_mode=params.get("feature_mode", "full"),
                                 base_dim=len(trial_spec.get("base_paths", [])))
            trial_pool.fit(model, train_ds_space, trial_spec, params.get("experts", ["ridge"]),
                           proposal["seed"], params.get("alpha", 0.2),
                           feature_mode=params.get("feature_mode", "full"),
                           base_dim=len(trial_spec.get("base_paths", [])),
                           logistic_c=float(params.get("logistic_c", 0.5)),
                           views=params.get("views"), view_weights=params.get("view_weights"),
                           logit_bias=bool(params.get("logit_bias", False)),
                           asym_weight=float(params.get("asym_weight", 1.0)),
                           conservative_bias=float(params.get("conservative_bias", 0.0)),
                           class_balanced=bool(params.get("class_balanced", True)),
                           tree_params=params.get("tree_params"))
            val_prediction = _augmented_prediction(model, trial_pool, trial_spec, validation_ds)
            val_metrics = diagnostics.evaluate(validation_ds.y, val_prediction, task, classes,
                                               rnd.dataset_key)
            train_prediction = _augmented_prediction(model, trial_pool, trial_spec, train_ds)
            train_metrics = diagnostics.evaluate(train_ds.y, train_prediction, task, classes,
                                                 rnd.dataset_key)
            losses = diagnostics.block_losses(validation_ds.y, val_prediction, validation_ds.groups,
                                              task, classes)
            result = {
                "directory": trial_dir, "model": model, "pool": trial_pool, "spec": trial_spec,
                "val_metrics": val_metrics, "train_metrics": train_metrics,
                "losses": losses, "proposal": proposal, "training": training_info,
                "cpu_seconds": time.monotonic() - started_trial,
                "score": _score(val_metrics, task),
            }
            trials.append(result)
            log.info("trial %d %s score=%.4f cpu=%.2fs", position, proposal["primitive_id"],
                     result["score"], result["cpu_seconds"])
        except Exception as exc:
            log.exception("trial %d failed", position)
            continue

    if not trials:
        raise RuntimeError("all sandbox trials failed")

    eligible = [item for item in trials
                if _sandbox_guard_ok(item["val_metrics"], rnd.dataset_key, task)]
    winner = max(eligible or trials, key=lambda item: item["score"])
    candidate_dir = work / ("candidate_r%d" % int(rnd.round_index))
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir()
    winner["model"].save(candidate_dir / "checkpoint.npz")
    winner["pool"].serialize(candidate_dir / "cep_bundle.npz")
    model_spec = {
        "format": "rsi.model-spec.v1", "task": task, "dataset_key": rnd.dataset_key,
        "classes": classes, "base_paths": train_ds.paths, "feature_spec": winner["spec"],
        "alpha": winner["pool"].alpha, "experts": [expert["kind"] for expert in winner["pool"].experts],
        "cep_feature_mode": winner["pool"].feature_mode, "cep_base_dim": winner["pool"].base_dim,
        "migration": winner["model"].meta.get("migration", {}),
        "migrated_from_sha256": parent_sha, "round_index": int(rnd.round_index),
    }
    (candidate_dir / "model_spec.json").write_text(
        json.dumps(model_spec, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")

    proposal = winner["proposal"]
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=proposal["hypothesis"],
        change_summary=proposal["change_summary"],
        configuration={"primitive_id": proposal["primitive_id"], "params": proposal["params"],
                       "router_action": action, "diagnostic_regime": report["drift_regime"],
                       "training": winner["training"], "round_index": int(rnd.round_index)},
        seed=int(proposal["seed"]),
        checkpoint_paths=["checkpoint.npz", "cep_bundle.npz", "model_spec.json"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    summary = submitted.get("summary") or {}
    measured = summary.get("metrics") if isinstance(summary, dict) else None
    if not isinstance(measured, dict):
        measured = winner["val_metrics"]
    anchor_metrics = (state.get("frozen_anchor") or {}).get("metrics")
    # Anchor-replay 基准：回归任务使用 Group-CV OOF（逐折重训、同协议），分类沿用 in-sample。
    if task == "regression":
        state["anchor_basis"] = "group_oof"
        # OOF 有预算护栏：剩余时间不足（留 10% wall 或至少 30s）时降级为 in-sample 基准。
        wall = float(rnd.wall_seconds) if getattr(rnd, "wall_seconds", None) else 480.0
        budget_ok = time.monotonic() < deadline - max(30.0, wall * 0.1)
        parent_recipe = {"epochs": 2, "lr": 0.001, "l2": 1e-4, "loss": "mse",
                         "experts": _pool_kinds(pool), "alpha": pool.alpha,
                         "tree_params": {}, "target_transform": parent_transform}
        # 父配方 OOF 缓存：父未变化（回滚轮）直接复用，免去每轮 3 折从零重训。
        cache = state.get("parent_anchor_oof_cache") or {}
        cache_key = parent_sha or str(rnd.parent.artifact_id)
        parent_anchor_metrics = None
        if cache.get("key") == cache_key and cache.get("metrics"):
            parent_anchor_metrics = cache["metrics"]
        elif budget_ok:
            parent_anchor_metrics = _oof_anchor(train_ds, spec, parent_backbone, parent_recipe,
                                                task, classes, rnd.dataset_key)
            if parent_anchor_metrics is not None:
                state["parent_anchor_oof_cache"] = {"key": cache_key, "metrics": parent_anchor_metrics}
        candidate_recipe = dict(winner["proposal"]["params"])
        candidate_recipe["target_transform"] = str(winner["spec"].get("target_transform", "") or "")
        candidate_anchor_metrics = None
        if budget_ok:
            candidate_anchor_metrics = _oof_anchor(train_ds, winner["spec"], winner["model"],
                                                   candidate_recipe, task, classes, rnd.dataset_key)
        if parent_anchor_metrics is None:
            parent_anchor_metrics = parent_train_metrics
            state["anchor_basis"] = "group_oof_fallback_insample"
        if candidate_anchor_metrics is None:
            candidate_anchor_metrics = winner["train_metrics"]
            state["anchor_basis"] = "group_oof_fallback_insample"
        if int(rnd.round_index) == 1:
            state["frozen_anchor_oof"] = {"metrics": parent_anchor_metrics}
    else:
        state["anchor_basis"] = "insample"
        candidate_anchor_metrics = winner["train_metrics"]
        parent_anchor_metrics = parent_train_metrics
    val_weights = diagnostics.block_weights(validation_ds.groups)
    verdict = gatekeeper.decide(measured, parent_metrics, anchor_metrics,
                                candidate_anchor_metrics, parent_anchor_metrics,
                                winner["losses"], parent_losses, task, state,
                                candidate_weights=val_weights, parent_weights=val_weights)
    decision = verdict["decision"]
    selected = submitted["artifact_id"]
    if decision == "ROLLBACK":
        fallback = state.get("best_artifact_id")
        available = {item.get("artifact_id") for item in rnd.artifacts}
        if fallback and fallback in available and fallback != rnd.parent.artifact_id:
            restore = work / ("history_restore_r%d" % int(rnd.round_index))
            if not restore.exists():
                try:
                    rnd.restore_artifact(fallback, str(restore))
                    selected = fallback
                except Exception as exc:
                    log.warning("restore failed: %s", exc)
        if selected == submitted["artifact_id"]:
            selected = rnd.parent.artifact_id
    else:
        state["best_artifact_id"] = selected
        state["best_metrics"] = measured

    parent_primary = gatekeeper.primary_value(parent_metrics, task)
    measured_primary = gatekeeper.primary_value(measured, task)
    gain = 0.0
    if parent_primary and measured_primary:
        gain = (measured_primary - parent_primary) / max(abs(parent_primary), 1e-9)
        if task == "regression":
            gain = -gain
    usage = (receipt.response.get("usage") if receipt is not None and isinstance(receipt.response, dict) else {}) or {}
    used_tokens = float(usage.get("completion_tokens") or usage.get("total_tokens") or 0)
    reward = float(gain) - 0.02 * min(1.0, used_tokens / max(1, int(rnd.max_output_tokens)))
    router.update(action, context, reward)
    state["router"] = router.serialize()
    evoguard_update(state, decision, proposal["primitive_id"], gain,
                    winner["cpu_seconds"], int(rnd.round_index))
    store.event({"round": int(rnd.round_index), "action": action, "decision": decision,
                 "primitive": proposal["primitive_id"], "gain": gain,
                 "metrics": measured, "e_value": verdict["e_value"], "reason": verdict["reason"]})
    state["events"] = state.get("events", [])[-200:]
    state["last_verdict"] = verdict
    log.info("round %d verdict=%s %s", rnd.round_index, decision, verdict["reason"])
    store.save()

    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=proposal["hypothesis"],
        change_summary=proposal["change_summary"],
        reason="诊断漂移=%s；路由=%s；平台指标=%s；父指标=%s；%s" % (
            report["drift_regime"], action, measured, parent_metrics, verdict["reason"]),
        configuration={"primitive_id": proposal["primitive_id"], "params": proposal["params"],
                       "router_action": action, "diagnostic_regime": report["drift_regime"],
                       "gate": verdict},
        seed=int(proposal["seed"]),
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    """Single-sample, stateless, CPU-only forward. Cache is process-local."""
    vector = sample_vector(sample.inputs, _base_paths(model_dir))
    return predict_one(vector, model_dir)


def _base_paths(model_dir):
    spec_path = Path(model_dir) / "model_spec.json"
    if spec_path.exists():
        try:
            meta = json.loads(spec_path.read_text(encoding="utf-8"))
            return meta["base_paths"]
        except Exception:
            pass
    return _BASE_PATHS_CACHE.get(str(model_dir), [])


_BASE_PATHS_CACHE = {}
