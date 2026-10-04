"""
Path resolution for v3.

Every relative path in configs/config.yaml is resolved against the v3/ root,
so scripts behave the same no matter which directory they are launched from.

Per-run layout (one run = one (label_ratio, seed) pair):

    runs/r0.10_s42/
    ├── data/
    │   ├── labeled_train.jsonl      有标签部分
    │   ├── unlabeled_pool.jsonl     无标签池：claim + evidence，不含标签
    │   └── unlabeled_gold.jsonl     无标签池的金标签，仅供评估/Oracle 使用
    ├── pseudo/                      伪标签池、NLI 三类概率（方法 F）与各方法的伪标签集合
    ├── checkpoints/                 extractor / RL 权重（默认跑完即删；detector 的最佳权重只保存在内存中）
    └── outputs/
        ├── extractor/               extractor 训练记录
        ├── detector/<tag>/          tag = 方法名（如 B）；换训练 seed 重复时为 B_t1042
        ├── pseudo_label_quality.json
        └── sanity_check.json        每个 run 结束后的自动检查

runs/protocol.json 记录这个 runs 目录使用的配置指纹（见 common/fingerprint.py），
防止不同配置的结果混在同一个目录里。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

# Ablation methods (see training/build_baseline_sets.py and README_v3.md).
MAIN_METHODS = ["A", "B", "Q", "K", "O"]
LEGACY_METHODS = ["W", "R", "C"]          # v2 设计，尚未按 v3 重新设计（README_v3 第 8 节）
ALL_METHODS = ["A", "B", "Q", "K", "F", "W", "R", "C", "O"]
FUSION_METHODS = ["F"]                    # 方向一：extractor + NLI 融合的 teacher（README_v3 10.6），需要 teacher.nli_fusion_alpha

_TAG_PATTERN = re.compile(r"^(?P<method>[A-Z])(?:_t(?P<train_seed>\d+))?$")


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute() and not config_path.exists():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path_str: str | Path) -> Path:
    """Resolve a config path relative to the v3 root."""
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def cfg_path(cfg: dict, key: str) -> Path:
    return resolve(cfg["paths"][key])


def labeled_split_path(cfg: dict, split: str) -> Path:
    return cfg_path(cfg, "processed_dir") / "labeled" / f"{split}.jsonl"


def nli_split_path(cfg: dict, split: str) -> Path:
    """NLI probabilities of a labeled split (training/compute_nli_probs.py; used by the A⊕NLI control)."""
    return cfg_path(cfg, "processed_dir") / "nli" / f"{split}.jsonl"


def nli_fusion_alpha(cfg: dict) -> float:
    """Weight of the NLI probabilities in the fused teacher of method F: p = (1-α)·p_extractor + α·p_NLI."""
    teacher = cfg.get("teacher") or {}
    if "nli_fusion_alpha" not in teacher:
        raise SystemExit("方法 F 需要配置 teacher.nli_fusion_alpha（见 configs/config_f.yaml）")
    alpha = float(teacher["nli_fusion_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"teacher.nli_fusion_alpha must be in [0, 1], got {alpha}")
    return alpha


def run_name(ratio: float, seed: int) -> str:
    return f"r{float(ratio):.2f}_s{int(seed)}"


def run_dir(cfg: dict, ratio: float, seed: int) -> Path:
    return cfg_path(cfg, "runs_dir") / run_name(ratio, seed)


def detector_tag(method: str, split_seed: int, train_seed: int | None = None) -> str:
    """Output name of one detector run: the method, plus _t<seed> when the training seed differs from the split seed."""
    if train_seed is None or int(train_seed) == int(split_seed):
        return method
    return f"{method}_t{int(train_seed)}"


def parse_detector_tag(tag: str) -> tuple[str, int | None] | None:
    """'B' -> ('B', None); 'B_t1042' -> ('B', 1042); anything else -> None."""
    m = _TAG_PATTERN.match(tag)
    if not m:
        return None
    return m["method"], (int(m["train_seed"]) if m["train_seed"] else None)


class RunPaths:
    """All file locations inside one run directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = resolve(root)
        self.data = self.root / "data"
        self.pseudo = self.root / "pseudo"
        self.checkpoints = self.root / "checkpoints"
        self.outputs = self.root / "outputs"

        self.labeled_train = self.data / "labeled_train.jsonl"
        self.unlabeled_pool = self.data / "unlabeled_pool.jsonl"
        self.unlabeled_gold = self.data / "unlabeled_gold.jsonl"
        self.split_stats = self.data / "split_stats.json"

        self.extractor_ckpt = self.checkpoints / "extractor" / "best_model.pt"
        self.extractor_outputs = self.outputs / "extractor"
        self.extractor_metrics = self.extractor_outputs / "extractor_train_metrics.json"

        self.pseudo_pool = self.pseudo / "pseudo_pool.jsonl"            # 全部无标签样本的伪标签
        self.pseudo_filtered = self.pseudo / "pseudo_filtered.jsonl"    # weight >= threshold（RL 与 W 的输入）
        self.pseudo_stats = self.pseudo / "pseudo_stats.json"
        self.nli_probs = self.pseudo / "nli_probs.jsonl"                # 池中每条的 NLI 三类概率（方法 F）
        self.baseline_stats = self.pseudo / "baseline_sets_stats.json"
        self.rl_selected = self.pseudo / "set_C_rl.jsonl"
        self.rl_outputs = self.outputs / "rl_selector"
        self.rl_ckpt = self.checkpoints / "rl_selector" / "ppo_model.pt"

        self.quality_report = self.outputs / "pseudo_label_quality.json"
        self.sanity_report = self.outputs / "sanity_check.json"
        self.import_marker = self.root / "imported_from.json"

    def pseudo_set(self, method: str) -> Path | None:
        """Pseudo-label file consumed by the detector for each ablation method."""
        mapping = {
            "A": None,
            "B": self.pseudo / "set_B_conf_top.jsonl",
            "Q": self.pseudo / "set_Q_conf_c_quantile.jsonl",
            "K": self.pseudo / "set_K_conf_topk.jsonl",
            "F": self.pseudo / "set_F_fused_top.jsonl",
            "W": self.pseudo / "set_W_weighted.jsonl",
            "R": self.pseudo / "set_R_random.jsonl",
            "C": self.rl_selected,
            "O": self.pseudo / "set_O_oracle.jsonl",
        }
        if method not in mapping:
            raise ValueError(f"Unknown method '{method}'. Expected one of {sorted(mapping)}")
        return mapping[method]

    def detector_outputs(self, tag: str) -> Path:
        return self.outputs / "detector" / tag

    def detector_ckpt(self, tag: str) -> Path:
        return self.checkpoints / f"detector_{tag}" / "best_model.pt"

    def detector_results(self) -> dict[str, dict]:
        """All finished detector runs in this run directory: {tag: test_results.json content}."""
        import json

        out = {}
        det_dir = self.outputs / "detector"
        if not det_dir.exists():
            return out
        for res_path in sorted(det_dir.glob("*/test_results.json")):
            if parse_detector_tag(res_path.parent.name) is None:
                continue
            with open(res_path, encoding="utf-8") as fh:
                out[res_path.parent.name] = json.load(fh)
        return out
