"""
配置指纹：防止不同配置的结果混在一起（README_v3 第 6 节）。

- config_fingerprint(cfg)：影响结果的全部设置的哈希。不包括只决定"跑哪些"或"放在哪里"的键
  （实验矩阵、设备、路径、worker 数等），所以增加 seed 或方法不会改变指纹。
  run_all.py 把它写进 runs/protocol.json，并在每个 detector 结果中记录；
  aggregate_results.py 发现同一目录里有不同指纹的结果时报错。
- pool_settings(cfg)：决定 SSL 划分、extractor 和伪标签池的设置。tools/import_from_v2.py
  用它确认 v2 的伪标签池在 v3 中仍然有效（两边必须完全一致）。
"""

from __future__ import annotations

import copy
import hashlib
import json

# (section, key) pairs that never change results.
_EXCLUDED_KEYS = {
    ("experiment", "label_ratios"),
    ("experiment", "seeds"),
    ("experiment", "methods"),
    ("experiment", "device"),
    ("experiment", "cleanup_checkpoints"),
    ("training", "num_workers"),
    ("training", "seed"),          # 每个 run 由 run_all.py 的 --seed 覆盖
}
_EXCLUDED_SECTIONS = {"paths", "use_local_models"}

# Settings read by make_ssl_split.py, train_extractor.py and generate_pseudolabels.py.
_POOL_TRAINING_KEYS = (
    "batch_size", "max_length", "learning_rate", "max_epochs", "extractor_max_epochs",
    "gradient_accumulation", "extractor_gradient_accumulation", "extractor_text_augment",
    "early_stopping_patience", "warmup_ratio", "warmup_steps", "freeze_layers", "init_from_nli",
    "use_bf16", "use_fp16", "use_tf32",
)
_POOL_IMBALANCE_KEYS = (
    "use_balanced_extractor_sampler", "extractor_sampler_power", "class_weight_power",
    "normalize_class_weights",
)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_fingerprint(cfg: dict) -> str:
    trimmed = copy.deepcopy(cfg)
    for section in _EXCLUDED_SECTIONS:
        trimmed.pop(section, None)
    for section, key in _EXCLUDED_KEYS:
        if isinstance(trimmed.get(section), dict):
            trimmed[section].pop(key, None)
    return hashlib.sha256(_canonical(trimmed).encode("utf-8")).hexdigest()[:12]


def pseudolabel_prior_tau(cfg: dict) -> float | None:
    """Effective class-prior correction used when generating pseudo labels (None = disabled).

    v3 reads imbalance.pseudolabel_prior_tau; v2 configs used algorithm.logit_adjust_tau for this,
    which also drove the detector loss. Both are understood here so v2 and v3 pools can be compared.
    """
    imb = cfg.get("imbalance", {}) or {}
    if not bool(imb.get("apply_prior_adjust_in_pseudolabels", True)):
        return None
    if "pseudolabel_prior_tau" in imb:
        tau = float(imb["pseudolabel_prior_tau"])
    else:
        tau = float((cfg.get("algorithm", {}) or {}).get("logit_adjust_tau", 0.0))
    return tau if tau > 0 else None


def pool_settings(cfg: dict) -> dict:
    training = cfg.get("training", {}) or {}
    imbalance = cfg.get("imbalance", {}) or {}
    hyper = cfg.get("hyperparameters", {}) or {}
    return {
        "models": {k: (cfg.get("models", {}) or {}).get(k) for k in ("deberta_base", "nli_model")},
        "data": cfg.get("data", {}),
        "training": {k: training.get(k) for k in _POOL_TRAINING_KEYS},
        "imbalance": {k: imbalance.get(k) for k in _POOL_IMBALANCE_KEYS},
        "pseudolabel_prior_tau": pseudolabel_prior_tau(cfg),
        "composite_weight": {k: hyper.get(k) for k in ("beta1", "beta2", "beta3")},
        "weight_threshold": (cfg.get("experiment", {}) or {}).get("weight_threshold"),
    }


def diff_settings(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Human-readable list of differences between two nested dicts."""
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        name = f"{prefix}{key}"
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_settings(va, vb, prefix=name + ".")
        elif va != vb:
            out.append(f"{name}: {va!r} != {vb!r}")
    return out
