"""Unit tests for protocol, migration, contracts, gate and streaming predict."""
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

APP = ROOT / "app"
SDK = ROOT / "sdk_src"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(SDK))
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.pathsep.join([str(APP), str(SDK)])


class ProtocolTests(unittest.TestCase):
    def test_describe_declares_both_tasks(self):
        result = subprocess.run([sys.executable, str(APP / "harness.py"), "describe"],
                                capture_output=True, text=True, env=ENV, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        frame = json.loads(result.stdout.strip().splitlines()[0])
        self.assertEqual(frame["protocol"], "contest.harness.workspace.v2")
        self.assertEqual(set(frame["tasks"]), {"classification", "regression"})

    def test_forbidden_input_rejected(self):
        from rsi_engine.contracts import flatten_inputs
        with self.assertRaises(ValueError):
            flatten_inputs({"features": [1.0], "rul": 2.0})


class ModelTests(unittest.TestCase):
    def test_migration_preserves_parent_output(self):
        from rsi_engine.cep_model import MLPBackbone, migrate
        paths = [json.dumps([["key", "f"], ["index", i]]) for i in range(4)]
        x = np.random.default_rng(0).normal(size=(20, 4))
        y = np.arange(20, dtype=float)
        parent = MLPBackbone("regression", paths, [], x, y, hidden=6, seed=1)
        derived = [json.dumps([["key", "__derived__"], ["index", i]]) for i in range(2)]
        spec = {"derived_paths": derived, "derived_mean": [0.5, -0.5], "derived_scale": [1.0, 2.0],
                "base_paths": paths}
        migrated = migrate(parent, spec, "abc")
        probe = np.zeros((1, 4))
        migrated_input = np.concatenate([probe, np.asarray([[0.5, -0.5]])], axis=1)
        self.assertTrue(np.allclose(parent.predict(probe), migrated.predict(migrated_input), atol=1e-6))

    def test_rollback_gate(self):
        from rsi_engine.gatekeeper import decide
        parent = {"rmse": 100.0}
        anchor = {"rmse": 110.0}
        degraded = {"rmse": 150.0}
        improved = {"rmse": 80.0}
        bad = decide(degraded, parent, anchor, degraded, parent, {"a": 4.0}, {"a": 1.0},
                     "regression", {})
        good = decide(improved, parent, anchor, improved, parent, {"a": 0.5}, {"a": 1.0},
                      "regression", {})
        self.assertEqual(bad["decision"], "ROLLBACK")
        self.assertEqual(good["decision"], "ACCEPT")


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="rsi_stream_"))
        self.artifact = self.temp / "model"
        self.artifact.mkdir()
        from rsi_engine.cep_model import CEPPool, MLPBackbone
        paths = [json.dumps([["key", "features"], ["index", i]], separators=(",", ":")) for i in range(3)]
        x = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        y = np.asarray([0.0, 1.0, 2.0, 1.0])
        model = MLPBackbone("regression", paths, [], x, y, hidden=4, seed=3)
        model.save(self.artifact / "checkpoint.npz")
        CEPPool("regression", [], 0.0).serialize(self.artifact / "cep_bundle.npz")
        spec = {"dataset_key": "C", "profile": "none", "base_paths": paths, "derived_paths": [],
                "derived_mean": [], "derived_scale": [], "derived_dim": 0, "base_dim": 3}
        (self.artifact / "model_spec.json").write_text(json.dumps({
            "format": "rsi.model-spec.v1", "task": "regression", "classes": [],
            "base_paths": paths, "feature_spec": spec, "alpha": 0.0,
        }), encoding="utf-8")
        context = {"protocol": "contest.predict-context.v1", "dataset_key": "C", "task": "regression",
                   "artifact": {}, "artifact_id": "unit"}
        self.env = ENV.copy()
        self.env.update({
            "RSI_MODEL_ROOT": str(self.artifact), "CONTEST_DATASET_KEY": "C", "CONTEST_TASK": "regression",
            "CONTEST_PREDICTION_CONTEXT": json.dumps(context),
            "CONTEST_ARTIFACT_PATH": str(self.artifact / "checkpoint.npz"),
            "CONTEST_WORK": str(self.temp / "work"),
        })

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_reply_before_next_line(self):
        process = subprocess.Popen([sys.executable, str(APP / "harness.py"), "predict"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, bufsize=1, env=self.env)
        try:
            first = json.dumps({"task": "regression", "inputs": {"features": [1.0, 0.0, 0.0]}})
            second = json.dumps({"task": "regression", "inputs": {"features": [0.0, 1.0, 0.0]}})
            process.stdin.write(first + "\n")
            process.stdin.flush()
            self.assertTrue(select.select([process.stdout], [], [], 10)[0], "no immediate first reply")
            reply = json.loads(process.stdout.readline())
            self.assertIn("value", reply["outputs"])
            process.stdin.write(second + "\n")
            process.stdin.flush()
            self.assertTrue(select.select([process.stdout], [], [], 10)[0], "no immediate second reply")
            process.stdout.readline()
        finally:
            process.stdin.close()
            process.kill()


class PhaseTwoTests(unittest.TestCase):
    def test_gas_group_log_rel_shape_and_finite(self):
        from rsi_engine.primitives import _gas_group_log_rel
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 1e4, size=(12, 128))
        result = _gas_group_log_rel(x)
        self.assertEqual(result.shape, (12, 128))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.allclose(np.linalg.norm(result.reshape(12, 16, 8), axis=1), 1.0, atol=1e-6))

    def test_bearing_shrinkage_scales_weights_and_moments(self):
        from rsi_engine.cep_model import MLPBackbone, shrink_backbone
        paths = [json.dumps([["key", "f"], ["index", i]], separators=(",", ":")) for i in range(5)]
        x = np.random.default_rng(1).normal(size=(20, 5))
        y = np.arange(20, dtype=float)
        model = MLPBackbone("regression", paths, [], x, y, hidden=4, seed=1)
        before = {key: model.params[key].copy() for key in ("w1", "w2")}
        first_before = model.first["w1"].copy()
        shrink_backbone(model, 0.5)
        self.assertTrue(np.allclose(model.params["w1"], before["w1"] * 0.5))
        self.assertTrue(np.allclose(model.params["w2"], before["w2"] * 0.5))
        self.assertTrue(np.allclose(model.first["w1"], first_before * 0.5))

    def test_regime_spec_uses_six_train_only_clusters(self):
        from rsi_engine.contracts import load_rows
        from rsi_engine.primitives import fit_feature_spec, transform_features
        rows = [json.loads(line) for line in Path(
            DATA_ROOT / "cpu-c-guide-20261110-v1" / "public" / "train.jsonl"
        ).read_text(encoding="utf-8").splitlines()[:3000] if line.strip()]
        dataset = load_rows(rows, "regression", "C")
        spec = fit_feature_spec(dataset, "regime")
        self.assertEqual(len(spec["centers"]), 6)
        self.assertIn("constant_sensor_mask", spec)
        augmented = transform_features(dataset.x[:5], spec)
        self.assertEqual(augmented.shape[1], dataset.x.shape[1] + spec["derived_dim"])

    def test_gate_evalue_direction_consistency(self):
        from rsi_engine.gatekeeper import decide
        parent_losses = {"u1": 10.0, "u2": 10.0, "u3": 10.0}
        better_losses = {"u1": 9.0, "u2": 9.0, "u3": 9.0}
        worse_losses = {"u1": 11.0, "u2": 11.0, "u3": 11.0}
        good = decide({"rmse": 80.0}, {"rmse": 100.0}, {"rmse": 110.0},
                      {"rmse": 80.0}, {"rmse": 100.0}, better_losses, parent_losses,
                      "regression", {"dataset_key": "B"})
        bad = decide({"rmse": 120.0}, {"rmse": 100.0}, {"rmse": 110.0},
                     {"rmse": 120.0}, {"rmse": 100.0}, worse_losses, parent_losses,
                     "regression", {"dataset_key": "B"})
        self.assertGreater(good["e_value"], 1.05)
        self.assertLess(bad["e_value"], 1.0)
        self.assertEqual(good["decision"], "ACCEPT")
        self.assertEqual(bad["decision"], "ROLLBACK")

    def test_gate_requires_e_metric_even_when_primary_improves(self):
        from rsi_engine.gatekeeper import decide
        parent_losses = {"u1": 10.0, "u2": 10.0}
        tiny_gain = {"u1": 9.99, "u2": 9.99}
        verdict = decide({"rmse": 99.9}, {"rmse": 100.0}, {"rmse": 110.0},
                         {"rmse": 99.9}, {"rmse": 100.0}, tiny_gain, parent_losses,
                         "regression", {"dataset_key": "B"})
        self.assertTrue(verdict["strict_primary"])
        self.assertEqual(verdict["decision"], "ROLLBACK")

    def test_weighted_evalue_resists_short_block_inflation(self):
        """样本加权后，小块上的巨大相对改善无法淹没大块上的恶化。"""
        from rsi_engine.gatekeeper import paired_evalue
        parent = {"big": 1.0, "s1": 1.0, "s2": 1.0, "s3": 1.0}
        candidate = {"big": 2.0, "s1": 0.5, "s2": 0.5, "s3": 0.5}
        weights = {"big": 0.7, "s1": 0.1, "s2": 0.1, "s3": 0.1}
        value, blocks = paired_evalue(candidate, parent, weights)
        self.assertEqual(blocks, 4)
        # 整体 MSE: parent 1.0 < candidate 1.55 → e 必须 < 1
        self.assertLess(value, 1.0)
        # 无权重时旧实现会给出 e>1（相对改善求和 -1+0.5*3=0.5）
        unweighted, _ = paired_evalue(candidate, parent, None)
        self.assertGreater(unweighted, 1.0)

    def test_weighted_evalue_matches_overall_mse_difference(self):
        from rsi_engine.gatekeeper import paired_evalue
        parent = {"u1": 4.0, "u2": 16.0}
        candidate = {"u1": 9.0, "u2": 4.0}
        weights = {"u1": 0.75, "u2": 0.25}
        value, _ = paired_evalue(candidate, parent, weights)
        # overall MSE: parent 7.0, candidate 7.75 → 变差 → e<1
        self.assertLess(value, 1.0)
        candidate2 = {"u1": 2.0, "u2": 8.0}
        value2, _ = paired_evalue(candidate2, parent, weights)
        # overall MSE: candidate 3.5 < 7.0 → 改善 → e>1
        self.assertGreater(value2, 1.05)

    def test_anchor_replay_governance_uses_frozen_oof_floor(self):
        """B 治理模式：候选 OOF 锚点略劣于父但优于冻结基线 → 放行。"""
        from rsi_engine.gatekeeper import decide
        state = {"dataset_key": "B", "anchor_basis": "group_oof",
                 "drift_regime": "overfitting",
                 "frozen_anchor_oof": {"metrics": {"rmse": 500.0}}}
        losses = {"b1": 400.0}
        verdict = decide({"rmse": 400.0}, {"rmse": 450.0}, {"rmse": 600.0},
                         {"rmse": 500.5}, {"rmse": 490.0}, losses, {"b1": 500.0},
                         "regression", state)
        self.assertIn("Track2-anchor-replay=PASS", verdict["reason"])
        # 非 B 数据集（无治理 + 父锚点 490）: 候选 500.5 > 490*1.015 → 拦截
        state_c = {"dataset_key": "C", "anchor_basis": "group_oof"}
        verdict_c = decide({"rmse": 400.0}, {"rmse": 450.0}, {"rmse": 600.0},
                           {"rmse": 500.5}, {"rmse": 490.0}, losses, {"b1": 500.0},
                           "regression", state_c)
        self.assertIn("Track2-anchor-replay=FAIL", verdict_c["reason"])

    def test_gate_blocks_extreme_overestimation_shift(self):
        from rsi_engine.gatekeeper import decide
        parent_losses = {"u1": 10.0}
        losses = {"u1": 5.0}
        candidate = {"rmse": 80.0, "overestimation_rate": 0.62}
        parent = {"rmse": 100.0, "overestimation_rate": 0.27}
        verdict = decide(candidate, parent, {"rmse": 110.0},
                         {"rmse": 80.0}, {"rmse": 100.0}, losses, parent_losses,
                         "regression", {"dataset_key": "C"})
        self.assertEqual(verdict["decision"], "ROLLBACK")
        self.assertIn("Track2-imbalance=FAIL", verdict["reason"])

    def test_inference_options_fast_path_floor(self):
        from types import SimpleNamespace
        from rsi_engine.aide_search import inference_options
        rnd = SimpleNamespace(max_output_tokens=2048)
        fast = inference_options(rnd, "fast_path")
        # Reasoning models spend the first tokens on a ``reasoning`` trace; a
        # 1024-token cap returns empty content. The floor must stay above that.
        self.assertEqual(fast["max_tokens"], 2048)
        # Thinking must be switched off explicitly on the fast path: omitting
        # the flag makes Qwen3 spend ~2K tokens reasoning (~65 s vs ~15 s).
        self.assertIs(fast["enable_thinking"], False)
        # Never exceed the platform-granted ceiling.
        small = inference_options(SimpleNamespace(max_output_tokens=600), "fast_path")
        self.assertEqual(small["max_tokens"], 600)
        deep = inference_options(rnd, "deep_path")
        self.assertEqual(deep["max_tokens"], 2048)
        self.assertIs(deep["enable_thinking"], True)
        # A retry escalates the budget instead of repeating the same size.
        retry = inference_options(SimpleNamespace(max_output_tokens=16384), "fast_path", attempt=1)
        self.assertGreater(retry["max_tokens"], 4096)

    def test_trials_migrate_from_independent_snapshots(self):
        """路径一致时 migrate 返回原对象；trial 必须用独立快照，防止相互污染。"""
        import copy as copy_module
        from rsi_engine.cep_model import MLPBackbone, migrate, train_backbone
        paths = [json.dumps([["key", "f"], ["index", i]], separators=(",", ":")) for i in range(4)]
        x = np.random.default_rng(5).normal(size=(40, 4))
        y = np.abs(np.random.default_rng(6).normal(size=40)) * 10
        parent = MLPBackbone("regression", paths, [], x, y, hidden=6, seed=2)
        spec = {"dataset_key": "C", "profile": "none", "base_paths": paths,
                "derived_paths": [], "derived_mean": [], "derived_scale": [],
                "derived_dim": 0, "base_dim": 4}
        frozen = copy_module.deepcopy(parent.params)
        # trial A：migrate + 训练（模拟真实 trial 流程）
        model_a = migrate(copy_module.deepcopy(parent), spec, "s")
        train_backbone(model_a, type("D", (), {"x": x, "y": y, "groups": ["g"] * len(y),
                                               "paths": paths, "task": "regression",
                                               "classes": [], "dataset_key": "C"})(),
                       spec, {"epochs": 2, "lr": 0.003}, 7)
        # 父模型权重必须保持不变（未被 trial 污染）
        for key in frozen:
            self.assertTrue(np.allclose(parent.params[key], frozen[key]),
                            "parent weights mutated by trial training")

    def test_lgbm_booster_cached_and_not_serialized(self):
        """Booster 只反序列化一次（缓存于内存），且缓存绝不写入落盘文件。"""
        import tempfile
        import zipfile
        from rsi_engine.cep_model import CEPPool, MLPBackbone
        from types import SimpleNamespace
        paths = [json.dumps([["key", "f"], ["index", i]], separators=(",", ":")) for i in range(3)]
        x = np.random.default_rng(2).normal(size=(80, 3))
        y = np.abs(x[:, 0]) * 10 + 1
        model = MLPBackbone("regression", paths, [], x, y, hidden=4, seed=1)
        spec = {"dataset_key": "C", "profile": "none", "base_paths": paths, "derived_paths": [],
                "derived_mean": [], "derived_scale": [], "derived_dim": 0, "base_dim": 3}
        dataset = SimpleNamespace(x=x, y=y, groups=["g"] * len(y), paths=paths,
                                  task="regression", classes=[], dataset_key="C")
        pool = CEPPool("regression", [], alpha=0.3)
        pool.fit(model, dataset, spec, ["ridge", "lgbm"], 1, alpha=0.3)
        before = pool.predict(model, x)
        lgbm_expert = next(e for e in pool.experts if e["kind"].startswith("lgbm"))
        self.assertIn("_booster", lgbm_expert)
        # 缓存后二次预测结果一致
        again = pool.predict(model, x)
        self.assertTrue(np.allclose(before, again, atol=0.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.npz"
            pool.serialize(path)
            with zipfile.ZipFile(path) as archive:
                self.assertFalse(any("_booster" in name for name in archive.namelist()))
            restored = CEPPool.load(path, "regression", [], 0.3)
            after = restored.predict(model, x)
        self.assertTrue(np.allclose(before, after, atol=1e-10))

    def test_tie_break_scores_bounds_and_uniqueness(self):
        """平局打破极端用例：分数必须落在 [0,1] 且唯一最大。"""
        from rsi_engine.cep_model import tie_break_scores
        cases = ([0.5, 0.5], [1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.9999999],
                 [0.99999999999, 1.0, 0.3], [1e-6, 1e-6])
        for scores in cases:
            out = tie_break_scores(list(scores))
            self.assertTrue(all(0.0 <= s <= 1.0 for s in out), (scores, out))
            self.assertEqual(len(out), len(scores))
            top = max(out)
            self.assertEqual(sum(1 for s in out if s >= top - 0.0 and s == top), 1, (scores, out))
            self.assertGreaterEqual(top, 1e-6)

    def test_cep_serialize_round_trip_keeps_residual_bias(self):
        """回归池的保守偏置必须跨序列化保留，否则落盘重载后预测整体上移。"""
        import tempfile
        from rsi_engine.cep_model import CEPPool
        pool = CEPPool("regression", [], alpha=0.4)
        pool.residual_bias = -1.25
        pool.asym_weight = 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.npz"
            pool.serialize(path)
            restored = CEPPool.load(path, "regression", [], 0.4)
        self.assertAlmostEqual(restored.residual_bias, -1.25)
        self.assertAlmostEqual(restored.asym_weight, 2.0)

    def test_target_space_sqrt_round_trip(self):
        from contest_submission import _invert_prediction, _target_space_y
        y = np.asarray([0.0, 4.0, 25.0, 100.0])
        space = _target_space_y(y, "sqrt")
        self.assertTrue(np.allclose(space, [0.0, 2.0, 5.0, 10.0]))
        raw = _invert_prediction(space, {"dataset_key": "C", "target_transform": "sqrt"})
        self.assertTrue(np.allclose(raw, y))
        # 未带变换标记的 spec 不做逆变换
        same = _invert_prediction(space, {"dataset_key": "C"})
        self.assertTrue(np.allclose(same, space))

    def test_regime_settings_standardized(self):
        from rsi_engine.contracts import load_rows
        from rsi_engine.primitives import fit_feature_spec, setting_standardized, transform_features
        rows = [json.loads(line) for line in Path(
            DATA_ROOT / "cpu-c-guide-20261110-v1" / "public" / "train.jsonl"
        ).read_text(encoding="utf-8").splitlines()[:3000] if line.strip()]
        dataset = load_rows(rows, "regression", "C")
        spec = fit_feature_spec(dataset, "regime")
        self.assertIn("setting_mean", spec)
        z = setting_standardized(dataset.x[:50], spec)
        self.assertTrue(np.all(np.isfinite(z)))
        # 标准化后各工况维度量级一致（不会 [60,100] 淹没 [0,0.84]）
        self.assertLess(float(z.std(axis=0).max()), 10.0)
        # 单样本变换与批量变换一致（无状态）
        batch = transform_features(dataset.x[:8], spec)
        single = np.stack([transform_features(dataset.x[i:i + 1], spec)[0] for i in range(8)])
        self.assertTrue(np.allclose(batch, single, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
