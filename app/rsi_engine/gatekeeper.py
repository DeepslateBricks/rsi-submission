"""Block-level e-process gate with mathematically consistent dual-track decision.

e-process 语义（v3，样本加权）:
  log(e) = clip( Σ_b w_b (L_parent,b - L_candidate,b) / (Σ_b w_b L_parent,b + 1e-9), -4, 4 )
  其中 w_b = 块 b 的样本数占比。块损失均为逐样本均值损失（回归=MSE，分类=1-块内宏F1），
  块按样本划分互不重叠，因此加权损失差严格对应整体主指标损失之差：
  回归下 Δ 加权 = MSE_parent - MSE_candidate，e>1 ⟺ RMSE 改善；
  分类下块损失为分组宏 F1（近似），另加一致性钳制兜底（见 decide）。
"""
import math


def primary_value(metrics, task):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("macro_f1" if task == "classification" else "rmse")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def better(candidate, parent, task, tolerance=0.0):
    a, b = primary_value(candidate, task), primary_value(parent, task)
    if a is None or b is None:
        return False
    return a >= b + tolerance if task == "classification" else a <= b - tolerance


def paired_evalue(candidate_losses, parent_losses, weights=None, cap=1e12):
    """Sample-weighted block e-value on the PRIMARY metric loss.

    Blocks are weighted by their sample share w_b, so the weighted aggregate
    equals the overall primary-loss difference; per-block clipping is dropped
    in favour of a single aggregate clip [-4, 4] to keep the correspondence
    exact. e > 1 iff the overall primary metric improved; e <= 1 whenever it
    did not — short/long blocks can no longer inflate the e-value.
    """
    shared = [group for group in parent_losses if group in candidate_losses]
    if not shared:
        return 1.0, 0
    if weights is None:
        weights = {group: 1.0 / len(shared) for group in shared}
    delta, parent_total = 0.0, 0.0
    for group in shared:
        weight = float(weights.get(group, 0.0))
        delta += weight * (float(parent_losses[group]) - float(candidate_losses[group]))
        parent_total += weight * float(parent_losses[group])
    relative = delta / (abs(parent_total) + 1e-9)
    log_value = max(-4.0, min(4.0, relative))
    return min(cap, math.exp(log_value)), len(shared)


def _anchor_replay_ok(candidate_anchor, parent_anchor, frozen_anchor, task, state):
    """Anchor-replay red line on a comparable basis (Group-CV OOF for regression).

    - 常规模式：候选锚点误差不得劣于父锚点误差 1.5%。
    - overfitting 治理模式（regime=overfitting 或数据集 B）：红线放宽为
      「不劣于父锚点与冻结基线锚点中更宽松者」的 1.5%，即泛化能力不弱于
      Frozen Baseline 的 Group-CV OOF 误差即放行，避免用 in-sample 过拟合
      误差误杀去过拟合候选。
    """
    cand = primary_value(candidate_anchor, task)
    parent = primary_value(parent_anchor, task)
    if cand is None or parent is None:
        return True, "basis=unavailable(skip)"
    governance = state.get("drift_regime") == "overfitting" or state.get("dataset_key") == "B"
    limit_parent = parent
    basis = "group_oof" if state.get("anchor_basis") == "group_oof" else "insample"
    if governance and frozen_anchor is not None:
        frozen = primary_value(frozen_anchor, task)
        if frozen is not None:
            # regression: looser (larger) line; classification: looser (smaller) line.
            limit_parent = max(parent, frozen) if task == "regression" else min(parent, frozen)
            basis += "+frozen_floor"
    if task == "regression":
        ok = cand <= limit_parent * 1.015
        detail = "cand=%.4f line=%.4f" % (cand, limit_parent * 1.015)
    else:
        ok = cand >= limit_parent * 0.985
        detail = "cand=%.4f line=%.4f" % (cand, limit_parent * 0.985)
    return bool(ok), "basis=%s %s" % (basis, detail)


def decide(candidate_metrics, parent_metrics, anchor_metrics,
           candidate_anchor_metrics, parent_anchor_metrics, candidate_losses, parent_losses,
           task, state, e_threshold=1.01, candidate_weights=None, parent_weights=None):
    """Dual-track verdict.

    ``e_threshold``：样本加权 e-process 下，e = exp(整体主指标损失的相对改善率)，
    规范要求「e > 1.0 才可 ACCEPT」；默认取 1.01（在 e>1.0 之上保留 1% 证据缓冲，
    对齐验证集噪声底，过滤 <0.5% RMSE 量级的随机抖动）。分类块损失（分组宏F1）
    噪声更大，调用方可按需调高。

    ``candidate_anchor_metrics`` / ``parent_anchor_metrics``: anchor-replay
    metrics on a comparable basis — Group-CV OOF metrics for regression
    (computed fold-wise by the caller), in-sample train metrics otherwise.
    ``candidate_weights`` / ``parent_weights``: per-block sample shares used by
    the weighted e-process (validation-side and parent-side may differ only if
    the parent was evaluated on a different projection; normally identical).
    """
    state = state or {}
    candidate_primary = primary_value(candidate_metrics, task)
    parent_primary = primary_value(parent_metrics, task)
    anchor_primary = primary_value(anchor_metrics, task)
    weights = candidate_weights or parent_weights
    evalue, blocks = paired_evalue(candidate_losses, parent_losses, weights)
    strict_primary = better(candidate_metrics, parent_metrics, task, 1e-9)
    clamped = False
    if not strict_primary and evalue > 1.0:
        # 一致性钳制：主指标未改善时，块级证据不允许给出 e>1。
        # 分类块损失（分组宏F1）与整体宏F1 的加权均值仅近似对应，此处兜底归一。
        evalue = 1.0
        clamped = True
    track1 = bool(strict_primary and evalue >= e_threshold and blocks >= 1)
    reasons = [
        "Track1=%s (candidate=%s parent=%s strict=%s e=%.3f threshold=%.2f blocks=%d%s)" % (
            "PASS" if track1 else "FAIL", candidate_primary, parent_primary,
            strict_primary, evalue, e_threshold, blocks, " e-clamp" if clamped else "")
    ]

    anchor_ok = better(candidate_metrics, anchor_metrics, task, -1e-9) if anchor_primary is not None else True
    reasons.append("Track2-baseline=%s (frozen=%s)" % ("PASS" if anchor_ok else "FAIL", anchor_primary))

    replay_ok, replay_detail = _anchor_replay_ok(
        candidate_anchor_metrics, parent_anchor_metrics,
        (state.get("frozen_anchor_oof") or {}).get("metrics"), task, state)
    reasons.append("Track2-anchor-replay=%s (%s)" % ("PASS" if replay_ok else "FAIL", replay_detail))

    nasa_ok = True
    if state.get("dataset_key") == "C":
        candidate_nasa = (candidate_metrics or {}).get("nasa_score")
        parent_nasa = (parent_metrics or {}).get("nasa_score")
        if candidate_nasa is not None and parent_nasa:
            growth = (float(candidate_nasa) - float(parent_nasa)) / abs(float(parent_nasa))
            improvement = abs((candidate_primary - parent_primary) / max(abs(parent_primary), 1e-9)) \
                if candidate_primary is not None and parent_primary is not None else 0.0
            nasa_ok = not (growth > 0.10 and improvement < 0.01)
        reasons.append("Track2-nasa-guard=%s" % ("PASS" if nasa_ok else "FAIL"))

    # 温和失衡约束（仅回归）：防止接受高估率 / 后段误差极端恶化的候选。
    imbalance_ok = True
    if task == "regression":
        cand_over = (candidate_metrics or {}).get("overestimation_rate")
        parent_over = (parent_metrics or {}).get("overestimation_rate")
        if cand_over is not None:
            ceiling = min(float(parent_over) + 0.15, 0.70) if parent_over is not None else 0.70
            if float(cand_over) > ceiling:
                imbalance_ok = False
                reasons.append("Track2-imbalance=FAIL (overestimation %.3f > %.3f)" % (float(cand_over), ceiling))
        for late_key, early_key in (("late_rmse", "early_rmse"),):
            cand_late = (candidate_metrics or {}).get(late_key)
            parent_late = (parent_metrics or {}).get(late_key)
            cand_early = (candidate_metrics or {}).get(early_key)
            parent_early = (parent_metrics or {}).get(early_key)
            if cand_late is not None and parent_late and float(parent_late) > 0:
                if float(cand_late) > float(parent_late) * 1.25:
                    imbalance_ok = False
                    reasons.append("Track2-imbalance=FAIL (%s %.1f > %.1f)" % (
                        late_key, float(cand_late), float(parent_late) * 1.25))
            if cand_early is not None and parent_early and float(parent_early) > 0:
                if float(cand_early) > float(parent_early) * 1.25:
                    imbalance_ok = False
                    reasons.append("Track2-imbalance=FAIL (%s %.1f > %.1f)" % (
                        early_key, float(cand_early), float(parent_early) * 1.25))
        if imbalance_ok:
            reasons.append("Track2-imbalance=PASS")

    accepted = bool(track1 and anchor_ok and replay_ok and nasa_ok and imbalance_ok)
    return {
        "decision": "ACCEPT" if accepted else "ROLLBACK",
        "e_value": evalue,
        "e_threshold": e_threshold,
        "blocks": blocks,
        "e_clamped": clamped,
        "reason": "; ".join(reasons),
        "track1": track1,
        "track2": bool(anchor_ok and replay_ok and nasa_ok and imbalance_ok),
        "strict_primary": strict_primary,
    }
