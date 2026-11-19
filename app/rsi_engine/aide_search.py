"""Controlled hypothesis generation: structured JSON, strict parsing, safe fallback."""
import json
import logging
import time

from . import primitives
from .router_skill import SkillApt

log = logging.getLogger("rsi.aide")


def inference_options(rnd, action, attempt=0):
    """Token budget for one planner call.

    Reasoning models (Qwen3.5/3.8) emit a ``reasoning`` trace before
    ``content``. Measured on the real gateway with a diagnostic-sized prompt:
    1024 tokens are fully consumed by reasoning and ``content`` comes back
    empty; ~4096 is the first budget that reliably yields parseable JSON.
    A starving budget therefore does not fail loudly -- it silently degrades
    the planner to templates, which is worse than an error.

    ``attempt`` escalates the budget on retry so a truncated reply is not
    retried at the same size.
    """
    ceiling = max(1, int(rnd.max_output_tokens))
    # With the compacted prompt (<=2 KB) a reply needs ~1.4-2 K completion
    # tokens, roughly half of it reasoning. 4096 leaves a comfortable margin
    # without inviting runaway thinking, which an 8-16 K budget does.
    if action == "deep_path":
        budget = min(6144, ceiling)
    else:
        budget = min(4096, ceiling)
    if attempt:
        budget = min(ceiling, budget * (2 ** attempt))
    options = {
        "max_tokens": budget,
        "temperature": 0.2 if action == "deep_path" else 0.1,
        "response_format": {"type": "json_object"},
    }
    # Measured on Qwen3.8-27B with this prompt: enable_thinking=False yields
    # reasoning_tokens=0 and answers in ~15 s, while omitting the flag spends
    # ~2 K tokens on reasoning and takes ~65 s. The fast path therefore asks
    # for thinking to be off explicitly instead of relying on the default.
    options["enable_thinking"] = action == "deep_path"
    return options


def _slim_report(report):
    """Trim the diagnostic report to the fields the planner actually reasons over.

    Prompt size drives reasoning length on Qwen3: a 2 KB prompt answers in
    ~35 s, while a 5 KB one can burn thousands of reasoning tokens and return
    empty content. Only the decision-relevant summary is kept.
    """
    if not isinstance(report, dict):
        return {}
    val = report.get("val") or {}
    return {
        "round_index": report.get("round_index"),
        "dataset_key": report.get("dataset_key"),
        "task": report.get("task"),
        "drift_regime": report.get("drift_regime"),
        "drift": {k: v for k, v in (report.get("drift") or {}).items()
                  if k in ("max_channel_shift", "channel_shift_p90", "covariate_shift_score")},
        "min_class_f1": report.get("min_class_f1"),
        "overfit_gap": report.get("overfit_gap"),
        "root_cause": {k: v for k, v in (report.get("root_cause") or {}).items()
                       if k in ("regime", "evidence", "overestimation_rate")},
        "val": {
            "macro_f1": val.get("macro_f1"),
            "accuracy": val.get("accuracy"),
            "rmse": val.get("rmse"),
            "mae": val.get("mae"),
            "nasa_score": val.get("nasa_score"),
            "overestimation_rate": val.get("overestimation_rate"),
            "weakest_class": val.get("weakest_class"),
            "per_class_f1": val.get("per_class_f1"),
            "top_confusions": (val.get("top_confusions") or [])[:2],
        },
    }


# One-line rationale per primitive, so a small planner model can tell which
# recipe suits the diagnosed drift instead of guessing from hyperparameters.
_PRIMITIVE_NOTES = {
    # A: 128 features = 16 sensors x 8 channels; the public split is batch-1 -> batch-2.
    "group_dual": "dual view (global L2 + sensor cross-group log-ratio); best for batch drift, keep alpha high (~0.8)",
    "group_log_rel": "sensor cross-group log-ratio only; light probe of the drift",
    "l2_domain": "global L2 normalisation; cheap, suits stable batches",
    "class_rebalance": "oversampling + logistic expert; use when a minority class collapses",
    "sensor_l2": "per-sensor L2; keeps channel structure, useful for minority classes",
    # B: bearing vibration, RUL in minutes.
    "shrink_reg": "shrink MLP weights + ridge expert; suits noisy degradation signals",
    "bearing_inv": "scale-invariant bearing features; main recipe for RUL",
    "strong_reg": "stronger regularisation on channel-relative features",
    # C: C-MAPSS, RUL in cycles, overestimation is penalised asymmetrically.
    "regime_big": "regime clustering + larger trees; main recipe for cycle RUL",
    "regime_sqrt": "sqrt target transform; reduces overestimation",
    "regime_xlarge": "largest tree ensemble; use when underfitting persists",
    "asym_huber": "asymmetric loss; penalises overestimation more",
    "regime_norm": "huber loss + regime view; conservative fallback",
}

# Knobs whose defaults matter for the drift being handled. A planner may tune
# them, but not so far that the primitive's view is bypassed (alpha=0 drops the
# expert blend entirely; alpha=1 zeroes the MLP backbone).
_ALPHA_FLOOR = {"group_dual": 0.5, "sensor_dual": 0.4, "class_rebalance": 0.2}


def _slim_candidate(candidate):
    """The planner needs to know *what* each primitive is for, not just its knobs.

    Sending only hyperparameters made the small planner model pick recipes at
    random (and e.g. crush ``group_dual``'s alpha), which is why datasets drifted
    back to baseline. Keep the profile plus a one-line note.
    """
    keep = ("id", "profile", "epochs", "lr", "l2", "alpha", "loss", "experts",
            "class_weight", "shrinkage", "target_transform", "oversample")
    slim = {k: candidate[k] for k in keep if k in candidate}
    note = _PRIMITIVE_NOTES.get(candidate.get("id"))
    if note:
        slim["note"] = note
    return slim


def _slim_baseline(metrics):
    """Primary numbers only: per-class detail is already in the report."""
    if not isinstance(metrics, dict):
        return None
    return {k: metrics.get(k) for k in ("macro_f1", "accuracy", "rmse", "mae", "nasa_score")
            if metrics.get(k) is not None}


def _slim_history(history):
    """Keep the last three rounds, metrics only."""
    out = []
    for item in list(history or [])[-3:]:
        if not isinstance(item, dict):
            continue
        out.append({"id": item.get("id"),
                    "selected_artifact_id": item.get("selected_artifact_id"),
                    "metrics": item.get("metrics")})
    return out


def build_prompt(rnd, report, state, candidates, history):
    task = rnd.task
    skill = SkillApt(state.get("skill"))
    ranks = skill.rank(task.get("task", "regression"), report.get("drift_regime", "stable"),
                       [item["id"] for item in candidates], int(rnd.round_index))
    payload = {
        "dataset_key": rnd.dataset_key,
        "task_contract": {
            "task": task.get("task"), "metric": task.get("metric"),
            "metric_direction": task.get("metric_direction"), "target_unit": task.get("target_unit"),
        },
        "round_index": int(rnd.round_index),
        "diagnostic_report": _slim_report(report),
        "frozen_baseline": _slim_baseline(report.get("frozen_anchor", {}).get("metrics")),
        "recent_history": _slim_history(history),
        "skill_utility_ranking": [
            {"primitive_id": name, "conditional_gain": round(float(score), 4),
             "samples": entry["count_with"], "rollback_rate": round(entry["rollback_rate"], 3)}
            for score, name, entry in ranks
        ],
        "candidates": [_slim_candidate(c) for c in candidates],
    }
    instructions = (
        "你是工业时序漂移实验规划器。只输出 JSON 对象，格式 "
        '{"experiments":[{"hypothesis":"...","change_summary":"...","primitive_id":"...",'
        '"hyperparams":{"epochs":3,"lr":0.001,"l2":0.0001,"alpha":0.3},"seed":42}]}。'
        "给出 2-3 条可证伪实验；primitive_id 必须来自 candidates；"
        "hyperparams 只能使用 candidates 中出现的键；不要写任何代码，不要粘贴任务提示。"
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _extract_json(text):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None


def fallback_proposals(dataset_key, round_index, candidates, state):
    chosen = candidates[:3]
    proposals = []
    for offset, candidate in enumerate(chosen):
        variant = dict(candidate)
        variant["epochs"] = max(2, int(candidate.get("epochs", 3)) + ((round_index + offset) % 2))
        proposals.append({
            "hypothesis": "在%s原语上以受控超参微调，验证本轮漂移诊断下的条件效用。" % candidate["id"],
            "change_summary": "继承父检查点与Adam状态，显式迁移衍生特征，训练并融合CEP专家。",
            "primitive_id": candidate["id"],
            "hyperparams": variant,
            "seed": 42 + 17 * round_index + offset,
        })
    return proposals


def sanitize(proposals, candidates, round_index, deadline=None):
    known = {item["id"]: item for item in candidates}
    result = []
    if not isinstance(proposals, list):
        return result
    for position, item in enumerate(proposals[:3]):
        if not isinstance(item, dict):
            continue
        primitive = item.get("primitive_id")
        if primitive not in known:
            continue
        template = dict(known[primitive])
        hyper = item.get("hyperparams")
        if isinstance(hyper, dict):
            for key, value in hyper.items():
                if key in template and isinstance(value, (int, float)) and not isinstance(value, bool):
                    template[key] = value
        template["epochs"] = max(1, min(12, int(template.get("epochs", 3))))
        template["lr"] = max(1e-5, min(0.01, float(template.get("lr", 0.001))))
        template["l2"] = max(0.0, min(0.05, float(template.get("l2", 1e-4))))
        floor = _ALPHA_FLOOR.get(primitive, 0.0)
        template["alpha"] = max(floor, min(1.0, float(template.get("alpha", 0.3))))
        try:
            seed = int(item.get("seed", 42 + 17 * round_index + position))
        except (TypeError, ValueError):
            seed = 42 + position
        result.append({
            "hypothesis": str(item.get("hypothesis") or "本轮受控假设")[:2000],
            "change_summary": str(item.get("change_summary") or "依诊断结果续训并融合专家")[:2000],
            "primitive_id": primitive,
            "params": template,
            "seed": seed,
        })
        if deadline and time.monotonic() > deadline:
            break
    return result


def _signature(proposal):
    params = proposal.get("params", proposal)
    keys = ("profile", "epochs", "lr", "l2", "alpha", "shrinkage", "feature_mode", "target_transform",
            "logistic_c", "loss", "asym_weight", "conservative_bias", "effective_beta")
    parts = [str(proposal.get("primitive_id", ""))]
    for key in keys:
        if key in params:
            value = params[key]
            parts.append("%s=%s" % (key, round(float(value), 6) if isinstance(value, (int, float)) else value))
    for key in ("views", "view_weights", "experts", "tree_params"):
        if key in params:
            parts.append("%s=%s" % (key, json.dumps(params[key], sort_keys=True, default=str)))
    return "|".join(parts)


def _perturb(proposal, round_index):
    params = dict(proposal.get("params", {}))
    scale = 0.7 + 0.1 * (round_index % 3)
    params["epochs"] = max(1, min(12, int(params.get("epochs", 3)) + 1 + (round_index % 2)))
    params["lr"] = max(1e-5, min(0.01, float(params.get("lr", 0.001)) * scale))
    params["l2"] = min(0.05, float(params.get("l2", 1e-4)) * (1.5 - 0.1 * (round_index % 3)))
    params["alpha"] = max(0.0, min(1.0, float(params.get("alpha", 0.3)) * (0.8 + 0.1 * (round_index % 3))))
    if "view_weights" in params:
        weights = [float(value) for value in params["view_weights"]]
        if len(weights) > 1:
            shift = 0.05 + 0.02 * (round_index % 3)
            weights[0] = max(0.1, weights[0] - shift)
            weights[1] = weights[1] + shift
            total = sum(weights)
            params["view_weights"] = [value / total for value in weights]
    proposal = dict(proposal)
    proposal["params"] = params
    proposal["seed"] = int(proposal.get("seed", 42)) + 101
    return proposal


def _explore(candidates, state, round_index):
    unfiltered = list(candidates)
    tried = set(state.get("tried_signatures", []))
    if int(state.get("rollback_streak", 0)) >= 2:
        cooled = set(state.get("cooldown_primitives", []))
        filtered = [item for item in unfiltered if item["id"] not in cooled]
        if len(filtered) >= 2:
            unfiltered = filtered
    original = list(unfiltered)
    fresh = [item for item in original if _signature({"primitive_id": item["id"], "params": item}) not in tried]
    if len(fresh) >= 2:
        return fresh + [item for item in original if item not in fresh]
    return original


def _make_novel(cleaned, state, round_index):
    tried = set(state.get("tried_signatures", []))
    result = []
    for proposal in cleaned:
        signature = _signature(proposal)
        if signature in tried:
            proposal = _perturb(proposal, round_index)
            signature = _signature(proposal)
        tried.add(signature)
        result.append(proposal)
    state["tried_signatures"] = sorted(tried)[-120:]
    return result


def propose(rnd, report, state, action, history):
    candidates = _explore(primitives.primitive_candidates(rnd.dataset_key), state, int(rnd.round_index))
    messages = build_prompt(rnd, report, state, candidates, history)
    deadline = None
    if getattr(rnd, "wall_seconds", None):
        deadline = time.monotonic() + float(rnd.wall_seconds) * 0.6

    def attempt(attempt_index, retry_messages=None, strip_extras=None):
        """One planner call; returns a receipt or raises."""
        options = inference_options(rnd, action, attempt=attempt_index)
        if strip_extras:
            # Some backends reject response_format / enable_thinking outright;
            # dropping them widens compatibility without changing the content.
            options.pop("response_format", None)
            options.pop("enable_thinking", None)
        return rnd.model(retry_messages or messages, **options)

    receipt = None
    last_error = None
    # Two escalating attempts: a reply truncated by the reasoning trace is
    # retried with a larger budget, not the same one.
    for attempt_index in range(2):
        try:
            receipt = attempt(attempt_index,
                              strip_extras=(attempt_index > 0 and action != "deep_path"))
            parsed = _extract_json(receipt.text)
            experiments = parsed.get("experiments") if isinstance(parsed, dict) else None
            cleaned = sanitize(experiments, candidates, int(rnd.round_index), deadline)
            if len(cleaned) >= 2:
                return _make_novel(cleaned, state, int(rnd.round_index)), receipt, action
            last_error = "insufficient valid experiments (%d)" % len(cleaned)
        except Exception as exc:
            last_error = str(exc)
        log.warning("planner attempt %d failed (%s)", attempt_index + 1, last_error)
        if deadline and time.monotonic() > deadline:
            break

    # Contract: every round must record at least one real model call id.
    if receipt is None:
        try:
            receipt = rnd.model([{"role": "user", "content": "确认收到本轮结构化诊断摘要。"}],
                                max_tokens=min(32, int(rnd.max_output_tokens)))
        except Exception as call_error:
            log.warning("fallback model call failed (%s)", call_error)
    fallback = sanitize(fallback_proposals(rnd.dataset_key, int(rnd.round_index), candidates, state),
                        candidates, int(rnd.round_index))
    return _make_novel(fallback, state, int(rnd.round_index)), receipt, action
