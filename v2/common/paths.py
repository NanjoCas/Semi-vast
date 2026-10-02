"""
Path resolution for v2.

Every relative path in configs/config.yaml is resolved against the v2/ root,
so scripts behave the same no matter which directory they are launched from.

Per-run layout (one run = one (label_ratio, seed) pair):

    runs/r0.10_s42/
    ├── data/
    │   ├── labeled_train.jsonl      有标签部分（含 Climate-FEVER 过采样）
    │   ├── unlabeled_pool.jsonl     无标签池：claim + evidence，不含标签
    │   └── unlabeled_gold.jsonl     无标签池的金标签，仅供评估/Oracle 使用
    ├── pseudo/                      伪标签与各方法的伪标签集合
    ├── checkpoints/                 extractor / detector 权重（默认跑完即删）
    └── outputs/                     各阶段指标、图、测试集预测
"""

from __future__ import annotations

from pathlib import Path

import yaml

V2_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = V2_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute() and not config_path.exists():
        config_path = V2_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path_str: str | Path) -> Path:
    """Resolve a config path relative to the v2 root."""
    p = Path(path_str)
    return p if p.is_absolute() else (V2_ROOT / p).resolve()


def cfg_path(cfg: dict, key: str) -> Path:
    return resolve(cfg["paths"][key])


def labeled_split_path(cfg: dict, split: str) -> Path:
    return cfg_path(cfg, "processed_dir") / "labeled" / f"{split}.jsonl"


def run_name(ratio: float, seed: int) -> str:
    return f"r{float(ratio):.2f}_s{int(seed)}"


def run_dir(cfg: dict, ratio: float, seed: int) -> Path:
    return cfg_path(cfg, "runs_dir") / run_name(ratio, seed)


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

        self.pseudo_pool = self.pseudo / "pseudo_pool.jsonl"            # 全部无标签样本的伪标签
        self.pseudo_filtered = self.pseudo / "pseudo_filtered.jsonl"    # weight >= threshold（RL 与 W 的输入）
        self.pseudo_stats = self.pseudo / "pseudo_stats.json"
        self.rl_selected = self.pseudo / "set_C_rl.jsonl"
        self.rl_outputs = self.outputs / "rl_selector"
        self.rl_ckpt = self.checkpoints / "rl_selector" / "ppo_model.pt"

    def pseudo_set(self, method: str) -> Path | None:
        """Pseudo-label file consumed by the detector for each ablation method."""
        mapping = {
            "A": None,
            "B": self.pseudo / "set_B_confidence.jsonl",
            "W": self.pseudo / "set_W_weighted.jsonl",
            "R": self.pseudo / "set_R_random.jsonl",
            "C": self.rl_selected,
            "O": self.pseudo / "set_O_oracle.jsonl",
        }
        if method not in mapping:
            raise ValueError(f"Unknown method '{method}'. Expected one of {sorted(mapping)}")
        return mapping[method]

    def detector_ckpt_dir(self, method: str) -> Path:
        return self.checkpoints / f"detector_{method}"

    def detector_outputs(self, method: str) -> Path:
        return self.outputs / "detector" / method
