"""Domain primitives. Every fitted statistic comes from train.jsonl only."""
import json
import math

import numpy as np
from sklearn.cluster import KMeans

EPS = 1e-8


def _path_has(path, token):
    return token in str(path)


def _column(paths, token, channel=None):
    for index, path in enumerate(paths):
        if _path_has(path, token) and (channel is None or _path_has(path, channel)):
            return index
    return None


def _safe_ratio(a, b):
    return a / (np.abs(b) + EPS)


def _standardize_fit(values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _standardize_apply(values, mean, scale):
    return np.clip((values - mean) / scale, -60.0, 60.0)


def _gas_group_log_rel(x):
    """A 专属：保号 log1p 后，在 16×8 结构的每个通道内跨传感器去均值+L2。

    乘性传感器增益漂移在对数域变为加性偏移，通道内去均值直接消掉它；
    8 个物理通道（ΔR / |ΔR|/R0 / EMA_pos / EMA_neg）被完整保留。
    """
    head = x[:, :128].reshape(len(x), 16, 8)
    logged = np.sign(head) * np.log1p(np.abs(head))
    centered = logged - logged.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True) + EPS
    return (centered / norm).reshape(len(x), 128)


def _bearing_scale_invariant(x, paths):
    """B 专属：无量纲频带占比、双通道幅值/冲击指标，抑制跨轴承共振频点过拟合。"""
    channels = []
    for name in ("horizontal", "vertical"):
        channels.append({key: _column(paths, '"%s"' % key, name) for key in (
            "rms", "peak", "peak_to_peak", "kurtosis", "crest_factor", "impulse_factor",
            "margin_factor", "spectral_rms", "spectral_kurtosis",
            "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")})

    def value(channel, key, default=0.0):
        index = channel.get(key)
        return x[:, index] if index is not None else np.full(len(x), default)

    cols = []
    for channel in channels:
        bands = np.stack([value(channel, key) for key in (
            "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")], axis=1)
        total = bands.sum(axis=1, keepdims=True) + EPS
        cols.extend([
            np.log1p(np.abs(value(channel, "rms"))),
            np.log1p(np.abs(value(channel, "peak"))),
            np.log1p(np.abs(value(channel, "spectral_rms"))),
            np.log1p(np.abs(value(channel, "spectral_kurtosis"))),
            bands[:, 0] / total[:, 0], bands[:, 1] / total[:, 0], bands[:, 2] / total[:, 0],
            np.log1p(np.abs(value(channel, "kurtosis"))),
            np.log1p(np.abs(value(channel, "crest_factor"))),
            np.log1p(np.abs(value(channel, "margin_factor"))),
        ])
    h, v = value(channels[0], "rms", 1.0), value(channels[1], "rms", 1.0)
    cols.append(h / (h + v + EPS))
    for key in ("kurtosis", "crest_factor", "margin_factor"):
        left, right = value(channels[0], key), value(channels[1], key)
        cols.append((left + right) / 2.0)
        cols.append(np.abs(left - right))
    return np.stack(cols, axis=1)


def _gas_sensor_l2(x):
    """Per-sensor L2 normalization: 128 = 16 sensors x 8 channels.

    The public A split is a pure batch-1 -> batch-2 transfer. Global L2 removes
    the dominant common-mode scale (PC1 carries ~98% of the variance) but also
    erases the channel structure the minority classes rely on. Normalizing each
    sensor block separately keeps that structure while still cancelling the
    per-batch gain. On the public validation split this is the only transform
    that lifts Acetaldehyde F1 off zero (0.00 -> 0.25).
    """
    if x.shape[1] < 128:
        return np.zeros((len(x), 0))
    blocks = x[:, :128].reshape(len(x), 16, 8)
    norm = np.linalg.norm(blocks, axis=2, keepdims=True)
    return (blocks / np.maximum(norm, EPS)).reshape(len(x), -1)


def _gas_derived(x, paths):
    """128 = 16 sensors x 8 features: single-sample cross-channel ratios."""
    if x.shape[1] < 128:
        return np.zeros((len(x), 0))
    head = x[:, :128].reshape(len(x), 16, 8)
    magnitude = head[:, :, 0]
    log_magnitude = np.log1p(np.abs(magnitude))
    relative = log_magnitude - log_magnitude.mean(axis=1, keepdims=True)
    # Every subsequent子特征除以同一样本该传感器的稳态幅值，消除批次全局增益漂移。
    ratios = head[:, :, 1:] / (1.0 + np.abs(magnitude)[:, :, None])
    ratios = np.sign(ratios) * np.log1p(np.abs(ratios))
    ratios = ratios.reshape(len(x), -1)
    return np.concatenate([relative, ratios], axis=1)


def _bearing_derived(x, paths):
    """34 pre-extracted vibration features, two channels x 17."""
    channels = []
    for name in ("horizontal", "vertical"):
        channels.append({key: _column(paths, '"%s"' % key, name) for key in (
            "rms", "peak", "peak_to_peak", "kurtosis", "skewness", "crest_factor",
            "impulse_factor", "margin_factor", "spectral_rms", "spectral_kurtosis",
            "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")})
    cols = []
    for channel in channels:
        def get(key, default=0.0):
            index = channel[key]
            return x[:, index] if index is not None else np.full(len(x), default)

        rms = get("rms", 1.0)
        peak = get("peak")
        cols.extend([
            np.log1p(np.abs(rms)),
            np.log1p(np.abs(peak)),
            _safe_ratio(get("band2_energy_2k_6k"), get("band1_energy_0_2k") + 1e-6),
            _safe_ratio(get("band3_energy_6k_12k"), get("band2_energy_2k_6k") + 1e-6),
            np.log1p(np.abs(get("kurtosis") * get("crest_factor"))),
            _safe_ratio(get("spectral_rms"), rms),
            get("impulse_factor") - get("margin_factor"),
            np.log1p(np.abs(get("peak_to_peak")) / (rms + 1e-8)),
        ])
    if all(channel["rms"] is not None for channel in channels):
        h, v = x[:, channels[0]["rms"]], x[:, channels[1]["rms"]]
        cols.append(np.log((h + 1e-8) / (v + 1e-8)))
        cols.append(np.log1p(np.sqrt(h * h + v * v)))
    else:
        cols.extend([np.zeros(len(x)), np.zeros(len(x))])
    return np.stack(cols, axis=1)


def setting_standardized(x, spec):
    """C 专属：工况 settings 的标准化视图（统计量仅来自 train）。

    没有该统计量的旧 spec 退回原始值，保证向后兼容。
    """
    settings = np.asarray(spec.get("setting_indices", [0, 1, 2]), dtype=np.int64)
    z = np.asarray(x, dtype=np.float64)[:, settings]
    mean = np.asarray(spec.get("setting_mean", []), dtype=np.float64)
    scale = np.asarray(spec.get("setting_scale", []), dtype=np.float64)
    if len(mean) == z.shape[1] and len(scale) == z.shape[1]:
        z = (z - mean) / scale
    return z


def _engine_derived(x, paths, spec):
    settings = np.asarray(spec.get("setting_indices") or [
        index for index, path in enumerate(paths) if _path_has(path, '"settings"')], dtype=np.int64)
    sensors = np.asarray(spec.get("sensor_indices") or [
        index for index, path in enumerate(paths) if _path_has(path, '"sensors"')], dtype=np.int64)
    if len(settings) != 3 or len(sensors) < 1:
        return np.zeros((len(x), 0))
    centers = np.asarray(spec["centers"], dtype=np.float64)
    # 工况聚类在标准化 settings 空间中进行，避免 [60, 100] 淹没 [0, 0.84]。
    z = setting_standardized(x, spec)
    distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    means = np.asarray(spec["cluster_mean"], dtype=np.float64)
    scales = np.asarray(spec["cluster_scale"], dtype=np.float64)
    return np.clip((x[:, sensors] - means[nearest]) / scales[nearest], -12.0, 12.0)


def fit_feature_spec(dataset, profile="auto"):
    """Fit transform on train rows only. profile: auto|rel_norm|channel_rel|regime."""
    key = dataset.dataset_key
    if profile == "auto":
        profile = {"A": "rel_norm", "B": "channel_rel", "C": "regime"}[key]
    x, paths = dataset.x, dataset.paths
    spec = {"dataset_key": key, "profile": profile, "base_dim": int(x.shape[1])}
    if key == "C" and profile == "regime":
        settings = np.asarray([i for i, p in enumerate(paths) if _path_has(p, '"settings"')], dtype=np.int64)
        sensors = np.asarray([i for i, p in enumerate(paths) if _path_has(p, '"sensors"')], dtype=np.int64)
        if len(settings) == 3 and len(sensors) > 0:
            setting_values = x[:, settings]
            setting_mean, setting_scale = _standardize_fit(setting_values)
            km = KMeans(n_clusters=6 if len(x) >= 600 else max(2, len(x) // 500),
                        n_init=10, random_state=42)
            # 先标准化再聚类：仅用 train 拟合统计量，防止大数值维度淹没小数值维度。
            km.fit(np.clip((setting_values - setting_mean) / setting_scale, -60.0, 60.0))
            centers = km.cluster_centers_
            setting_z = np.clip((setting_values - setting_mean) / setting_scale, -60.0, 60.0)
            # Use the fitted cluster assignment, not "the single nearest point":
            # members of a regime are all rows that belong to it, and fitting the
            # statistics on one row would give sd=0 and a noise-only mean.
            labels = km.labels_
            means, scales = [], []
            for index in range(len(centers)):
                members = x[:, sensors][labels == index]
                if len(members) < 2:
                    # Degenerate cluster: fall back to its centre's neighbourhood
                    # rather than a single sample.
                    distance = ((setting_z - centers[index]) ** 2).sum(axis=1)
                    members = x[:, sensors][np.argsort(distance)[:max(2, len(x) // 50)]]
                mu, sd = _standardize_fit(members)
                means.append(mu)
                scales.append(sd)
            constant_mask = (np.asarray(scales).min(axis=0) >= 1.0).tolist()
            spec.update(centers=centers.tolist(), cluster_mean=np.asarray(means).tolist(),
                        cluster_scale=np.asarray(scales).tolist(), sensor_indices=sensors.tolist(),
                        setting_indices=settings.tolist(), constant_sensor_mask=constant_mask,
                        setting_mean=setting_mean.tolist(), setting_scale=setting_scale.tolist())
            raw = _engine_derived(x, paths, spec)
        else:
            raw = np.zeros((len(x), 0))
    elif key == "B" and profile in ("channel_rel", "scale_invariant"):
        raw = _bearing_scale_invariant(x, paths) if profile == "scale_invariant" else _bearing_derived(x, paths)
        if profile == "scale_invariant":
            spec["zero_columns"] = [i for i, path in enumerate(paths) if _path_has(path, '"dominant_freq"')]
    elif key == "A" and profile == "sensor_group_log_rel":
        raw = _gas_group_log_rel(x)
    elif key == "A" and profile == "l2_norm":
        raw = x[:, :128] / (np.linalg.norm(x[:, :128], axis=1, keepdims=True) + EPS)
    elif key == "A" and profile == "rel_norm":
        raw = _gas_derived(x, paths)
    elif key == "A" and profile == "sensor_l2":
        raw = _gas_sensor_l2(x)
    else:
        raw = np.zeros((len(x), 0))
    if raw.shape[1]:
        mean, scale = _standardize_fit(raw)
        spec["derived_mean"] = mean.tolist()
        spec["derived_scale"] = scale.tolist()
        spec["derived_dim"] = int(raw.shape[1])
    else:
        spec["derived_mean"] = []
        spec["derived_scale"] = []
        spec["derived_dim"] = 0
    # The derived column name carries the profile. Different profiles produce the
    # same number of derived columns (128 for A), so a bare ``__derived__`` key
    # would make ``migrate()`` treat a view switch as "paths unchanged" and keep
    # the old view's mean/scale and w1 rows.
    derived_key = "__derived__:" + str(spec["profile"])
    spec["derived_paths"] = [json.dumps([["key", derived_key], ["index", i]], separators=(",", ":"))
                             for i in range(spec["derived_dim"])]
    return spec


def transform_features(x, spec):
    """Deterministic single/multi-sample transform; uses only persisted spec."""
    key = spec["dataset_key"]
    profile = spec["profile"]
    if key == "C" and profile == "regime" and "centers" in spec:
        raw = _engine_derived(x, [], spec)
    elif key == "B" and profile in ("channel_rel", "scale_invariant"):
        # B derivation uses column names; rebuild from the base paths stored in spec.
        base_paths = spec.get("base_paths", [])
        raw = (_bearing_scale_invariant(x, base_paths) if profile == "scale_invariant"
               else _bearing_derived(x, base_paths))
    elif key == "A" and profile == "sensor_group_log_rel":
        raw = _gas_group_log_rel(x)
    elif key == "A" and profile == "l2_norm":
        raw = x[:, :128] / (np.linalg.norm(x[:, :128], axis=1, keepdims=True) + EPS)
    elif key == "A" and profile == "sensor_l2":
        raw = _gas_sensor_l2(x)
    elif key == "A" and profile == "rel_norm":
        raw = _gas_derived(x, [])
    else:
        raw = np.zeros((len(x), 0))
    if raw.shape[1] and spec.get("derived_mean"):
        raw = _standardize_apply(raw - np.asarray(spec["derived_mean"]),
                                 np.zeros(raw.shape[1]), np.asarray(spec["derived_scale"]))
    base = np.asarray(x, dtype=np.float64).copy()
    for index in spec.get("zero_columns", []):
        if 0 <= int(index) < base.shape[1]:
            base[:, int(index)] = 0.0
    return np.concatenate([base, raw], axis=1)


def attach_paths(spec, base_paths):
    spec["base_paths"] = list(base_paths)
    return spec


def features_for_model(x_raw, spec, model_paths):
    """Unmigrated parent consumes raw columns; migrated models consume augmented columns."""
    if list(spec.get("base_paths", [])) == list(model_paths):
        return np.asarray(x_raw, dtype=np.float64)
    return transform_features(x_raw, spec)


def primitive_candidates(dataset_key):
    if dataset_key == "A":
        return [
            # 骨干必须真实参与：alpha=1.0 会把 MLP 权重乘 0 完全旁路；
            # 0.8 保留 20% 骨干投票，并以 sensor_group_log_rel 视图真实续训。
            {"id": "group_dual", "profile": "sensor_group_log_rel", "epochs": 4, "lr": 0.002, "l2": 1e-4,
             "class_weight": 1.0, "experts": [], "alpha": 0.8, "logistic_c": 10.0,
             "views": ["l2_norm", "sensor_group_log_rel"], "view_weights": [0.65, 0.35],
             "logit_bias": True, "effective_beta": 0.99},
            {"id": "group_log_rel", "profile": "sensor_group_log_rel", "epochs": 1, "lr": 1e-5,
             "l2": 0.0, "class_weight": 1.0, "experts": [], "alpha": 1.0, "logistic_c": 10.0,
             "views": ["sensor_group_log_rel"], "view_weights": [1.0], "logit_bias": True,
             "effective_beta": 0.99},
            {"id": "l2_domain", "profile": "l2_norm", "epochs": 1, "lr": 1e-5, "l2": 0.0,
             "class_weight": 1.0, "experts": ["logistic"], "alpha": 1.0,
             "feature_mode": "derived", "logistic_c": 10.0, "effective_beta": 0.99},
            {"id": "class_rebalance", "profile": "rel_norm", "epochs": 8, "lr": 0.0020, "l2": 3e-4,
             "class_weight": 1.0, "oversample": True, "experts": ["logistic"], "alpha": 0.25,
             "effective_beta": 0.99},
        ]
    if dataset_key == "B":
        return [
            {"id": "shrink_reg", "profile": "scale_invariant", "epochs": 5, "lr": 0.0005,
             "l2": 5e-3, "loss": "huber", "shrinkage": 0.55, "experts": ["ridge"], "alpha": 0.6,
             "tree_params": {"n_estimators": 60, "max_depth": 2, "min_child_samples": 120,
                             "colsample_bytree": 0.5, "reg_lambda": 50}},
            {"id": "bearing_inv", "profile": "scale_invariant", "epochs": 4, "lr": 0.0008,
             "l2": 2e-3, "loss": "mse", "shrinkage": 0.7, "experts": ["ridge", "lgbm"], "alpha": 0.7,
             "tree_params": {"n_estimators": 100, "max_depth": 3, "min_child_samples": 100,
                             "colsample_bytree": 0.6, "reg_lambda": 20}},
            {"id": "strong_reg", "profile": "channel_rel", "epochs": 4, "lr": 0.0006, "l2": 5e-3,
             "loss": "huber", "shrinkage": 0.8, "experts": ["ridge"], "alpha": 0.45},
        ]
    return [
        {"id": "regime_big", "profile": "regime", "epochs": 4, "lr": 0.0008, "l2": 1e-4,
         "loss": "asym", "asym_weight": 2.0, "conservative_bias": 0.03,
         "experts": ["ridge", "lgbm"], "alpha": 0.5,
         "tree_params": {"n_estimators": 250, "max_depth": 5, "num_leaves": 31,
                         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10}},
        {"id": "regime_sqrt", "profile": "regime", "epochs": 4, "lr": 0.0008, "l2": 1e-4,
         "loss": "asym", "asym_weight": 2.0, "conservative_bias": 0.02,
         "target_transform": "sqrt", "experts": ["ridge", "lgbm"], "alpha": 0.5,
         "tree_params": {"n_estimators": 250, "max_depth": 5, "num_leaves": 31,
                         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10}},
        {"id": "regime_xlarge", "profile": "regime", "epochs": 4, "lr": 0.0008, "l2": 1e-4,
         "loss": "asym", "asym_weight": 2.0, "conservative_bias": 0.03,
         "experts": ["ridge", "lgbm"], "alpha": 0.5,
         "tree_params": {"n_estimators": 300, "max_depth": 6, "num_leaves": 63,
                         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 8}},
        {"id": "asym_huber", "profile": "regime", "epochs": 5, "lr": 0.0010, "l2": 3e-4,
         "loss": "asym", "asym_weight": 2.2, "conservative_bias": 0.02,
         "experts": ["ridge"], "alpha": 0.3},
        {"id": "regime_norm", "profile": "regime", "epochs": 3, "lr": 0.0006, "l2": 1e-4,
         "loss": "huber", "asym_weight": 1.5, "experts": ["ridge", "lgbm"], "alpha": 0.45,
         "tree_params": {"n_estimators": 150, "max_depth": 4, "num_leaves": 15, "reg_lambda": 5}},
    ]
