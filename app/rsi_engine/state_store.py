"""Native state lifecycle: /state (read-only) -> /workspace/state (writable)."""
import json
import logging
from pathlib import Path

log = logging.getLogger("rsi.state")

DEFAULT_STATE = {
    "version": 1,
    "round": 0,
    "best_artifact_id": None,
    "best_metrics": None,
    "frozen_anchor": None,
    "router": {
        "arms": {
            "fast_path": {"A": [[], []], "b": []},
            "deep_path": {"A": [[], []], "b": []},
        },
        "last_action": None,
    },
    "skill": {},
    "trials": [],
    "cooldown": {},
    "rollback_streak": 0,
}


class StateStore:
    def __init__(self, rnd, work: Path):
        self.rnd = rnd
        base = work / "state"
        target = base
        counter = 1
        while target.exists() or target.is_symlink():
            target = work / ("state_%d" % counter)
            counter += 1
        self.dir = Path(rnd.state_workspace(str(target)))
        self.data = self._load()

    def _load(self):
        path = self.dir / "state.json"
        if not path.exists():
            return json.loads(json.dumps(DEFAULT_STATE))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("state root must be an object")
            merged = json.loads(json.dumps(DEFAULT_STATE))
            merged.update(value)
            return merged
        except Exception as exc:
            log.warning("state unreadable (%s); using defaults", exc)
            return json.loads(json.dumps(DEFAULT_STATE))

    def save(self):
        payload = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        (self.dir / "state.json").write_text(payload, encoding="utf-8")
        digest = self.rnd.save_state(str(self.dir))
        self.data["round"] = int(getattr(self.rnd, "round_index", 0))
        return digest

    def event(self, event: dict):
        events = self.data.setdefault("events", [])
        events.append(event)
        if len(events) > 200:
            del events[: len(events) - 200]
