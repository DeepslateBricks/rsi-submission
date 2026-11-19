"""PILOT budget router (LinUCB), SkillApt utility table, EvoGuard lock."""
import math
from collections import defaultdict

import numpy as np


def context_vector(report, state, total_rounds=10):
    drift = float(report.get("drift", {}).get("covariate_shift_score", 0.0))
    overfit = abs(float(report.get("overfit_gap", 0.0)))
    rollbacks = int(state.get("rollback_streak", 0))
    remaining = max(0.0, 1.0 - float(state.get("round", 1)) / total_rounds)
    ranked = state.get("skill", {}).get("__confidence__", 0.5)
    return np.asarray([
        min(drift / 2.0, 1.0),
        min(overfit / (200.0 if report.get("task") == "regression" else 0.5), 1.0),
        min(rollbacks / 3.0, 1.0),
        remaining,
        float(ranked),
    ], dtype=float)


class PilotRouter:
    ARMS = ("fast_path", "deep_path")

    def __init__(self, saved=None):
        saved = saved or {}
        arms = saved.get("arms", {})
        self.arms = {}
        for arm in self.ARMS:
            entry = arms.get(arm, {})
            matrix = np.asarray(entry.get("A", np.eye(5).tolist()), dtype=float)
            vector = np.asarray(entry.get("b", np.zeros(5).tolist()), dtype=float)
            if matrix.shape != (5, 5) or vector.shape != (5,):
                matrix, vector = np.eye(5), np.zeros(5)
            self.arms[arm] = {"A": matrix, "b": vector}
        self.last_action = saved.get("last_action")

    def select(self, context, exploration=0.25):
        context = np.asarray(context, dtype=float).reshape(5)
        if np.random.default_rng().random() < exploration:
            action = self.ARMS[int(np.random.default_rng().random() * 2)]
        else:
            scores = {}
            for arm, values in self.arms.items():
                inverse = np.linalg.pinv(values["A"])
                theta = inverse @ values["b"]
                bonus = 0.25 * math.sqrt(max(0.0, float(context @ inverse @ context)))
                scores[arm] = float(context @ theta) + bonus
            action = max(scores, key=scores.get)
        self.last_action = action
        return action

    def update(self, action, context, reward):
        context = np.asarray(context, dtype=float).reshape(5)
        arm = self.arms[action]
        arm["A"] += np.outer(context, context)
        arm["b"] += reward * context

    def serialize(self):
        return {
            "arms": {arm: {"A": values["A"].tolist(), "b": values["b"].tolist()}
                     for arm, values in self.arms.items()},
            "last_action": self.last_action,
        }


class SkillApt:
    def __init__(self, saved=None):
        self.table = defaultdict(lambda: {
            "count_with": 0, "count_without": 0, "mean_gain_with": 0.0, "mean_gain_without": 0.0,
            "avg_cpu_sec": 0.0, "rollback_rate": 0.0, "cooldown_until": 0,
        })
        for key, value in (saved or {}).items():
            if isinstance(key, str) and "|" in key:
                parts = key.split("|")
                if len(parts) == 3:
                    self.table[(parts[0], parts[1], parts[2])] = dict(value)

    def rank(self, task, regime, primitive_ids, round_index=1, limit=3):
        scored = []
        for primitive in primitive_ids:
            entry = self.table[(task, regime, primitive)]
            if int(entry.get("cooldown_until", 0)) > round_index:
                continue
            delta = entry["mean_gain_with"] - entry["mean_gain_without"]
            bonus = 0.05 * math.sqrt(1 + entry["count_with"])
            scored.append((delta + bonus, primitive, entry))
        scored.sort(key=lambda item: -item[0])
        return scored[:limit]

    def record(self, task, regime, primitive, gain, with_primitive, cpu_sec, rollback, round_index):
        entry = self.table[(task, regime, primitive)]
        target = "with" if with_primitive else "without"
        count_key = "count_" + target
        mean_key = "mean_gain_" + target
        count = entry[count_key]
        entry[count_key] = count + 1
        entry[mean_key] = (entry[mean_key] * count + gain) / (count + 1)
        entry["avg_cpu_sec"] = (entry["avg_cpu_sec"] * count + cpu_sec) / (count + 1)
        entry["rollback_rate"] = (entry["rollback_rate"] * count + (1 if rollback else 0)) / (count + 1)
        if rollback:
            entry["cooldown_until"] = round_index + 2
        return entry

    def serialize(self):
        return {"|".join(key): dict(value) for key, value in self.table.items()}


def evoguard_update(state, decision, primitive, gain, cpu_sec, round_index):
    """Only positive, accepted marginal utility is reinforced."""
    skill = SkillApt(state.get("skill"))
    task = state.get("task", "regression")
    regime = state.get("drift_regime", "stable")
    skill.record(task, regime, primitive, gain, True, cpu_sec, decision == "ROLLBACK", round_index)
    state["skill"] = skill.serialize()
    if decision == "ROLLBACK":
        state["rollback_streak"] = int(state.get("rollback_streak", 0)) + 1
        if state["rollback_streak"] >= 2:
            cooldown = set(state.get("cooldown_primitives", []))
            cooldown.add(primitive)
            state["cooldown_primitives"] = sorted(cooldown)[-12:]
    else:
        state["rollback_streak"] = 0
    return state
