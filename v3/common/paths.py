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
    ├── pseudo/                      伪标签池、NLI 三类概率（方法 F；方法 E 的另外两个模型为 nli_probs__<模型>.jsonl）与各方法的伪标签集合
    ├── checkpoints/                 extractor / RL / TAPT 后的 encoder 权重（默认跑完即删；detector 的最佳权重只保存在内存中）
    └── outputs/
        ├── extractor/               extractor 训练记录
        ├── tapt/、tapt_probe/       TAPT 的训练记录；可行性测试（只有验证集结果，README_v3 13.5）
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
ALL_METHODS = ["A", "AC", "AT", "B", "Q", "K", "F", "F25", "F75", "F100", "E", "FIX", "FREE", "W", "R", "C", "O"]
FUSION_METHODS = ["F"]                    # 方向一：extractor + NLI 融合的 teacher（README_v3 10.6），需要 teacher.nli_fusion_alpha
# README_v3 第 13 节。α 与集成模型写在这里而不进配置：配置里新增任何键都会改变指纹，使已有结果失效。
ALPHA_VARIANTS = {"F25": 0.25, "F75": 0.75, "F100": 1.0}   # 与 F 相同，只是 α 不同（α = 1.0 即只用 NLI）
CLAIM_ONLY_METHODS = ["AC"]               # 与 A 相同，但 detector 只输入 claim
ENSEMBLE_NLI_MODELS = ["facebook/bart-large-mnli", "FacebookAI/roberta-large-mnli"]  # 方法 E：加上 models.nli_model 共 3 个
NLI_POOL_METHODS = [*FUSION_METHODS, *ALPHA_VARIANTS, "E"]  # 需要无标签池的 NLI 概率
TAPT_METHODS = ["AT"]                     # 与 A 相同，但 encoder 先在无标签池上做 MLM 继续预训练（training/tapt.py，README_v3 13.5）
SSL_METHODS = ["FIX", "FREE"]             # 标准半监督基线 FixMatch / FreeMatch（README_v3 第 14 节）：一致性训练，不用伪标签集合
NO_PSEUDO_METHODS = ["A", *CLAIM_ONLY_METHODS, *TAPT_METHODS, *SSL_METHODS]   # 不需要 extractor 的伪标签池
# TAPT 的设置（README_v3 13.5）：在 seed 42 上做完可行性测试、补进 13.5 并提交 git 之后才填。
# 和 α 一样写在代码里，不进配置（否则指纹改变，runs_fc 的结果全部失效）。为 None 时 run_all.py 拒绝跑 AT。
TAPT_SETTINGS: dict | None = {"lr": 3e-6, "mask_prob": 0.15, "epochs": 1}   # 2026-11-07 可行性测试选定（README_v3 13.5.1）

# FixMatch / FreeMatch 的设置（README_v3 14.2，按 USB 的实现）。和 α、TAPT 一样写在代码里，不进配置。
# bt_variant 由 tools/backtranslate.py 在验证集上按 14.4 第 1 项的规则选出、补进 14.4 之后才填；为 None 时 run_all.py 拒绝跑。
SSL_SETTINGS: dict = {
    "p_cutoff": 0.95,          # FIX：固定阈值
    "ema_p": 0.999,            # FREE：自适应阈值（SAT）的 EMA 动量
    "ent_loss_ratio": 0.001,   # FREE：公平性损失（SAF）的权重（USB NLP 配置）
    "bt_langs": ["de", "ru"],  # 强增强：从这两份回译中每次随机取一份；弱增强为原文
    "bt_variant": "V1",        # V1 claim 和证据都回译 / V2 只回译证据 / V3 只回译 claim；2026-11-14 按 14.4 第 1 项选定
}
# 回译模型（en → 中间语言，中间语言 → en），用模型自带的解码设置（README_v3 14.3）
BT_MODELS = {
    "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
}
BT_MAX_WORDS = 450   # 每段证据只回译前 450 个词：detector 的输入截断在 512 个 token，之后的部分本来就看不到

_TAG_PATTERN = re.compile(r"^(?P<method>[A-Z][A-Za-z0-9]*)(?:_t(?P<train_seed>\d+))?$")


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


def nli_model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def nli_split_path(cfg: dict, split: str, model_name: str | None = None) -> Path:
    """NLI probabilities of a labeled split (training/compute_nli_probs.py; used by the A⊕NLI control).
    model_name: None or models.nli_model for the LogicScore model, otherwise one of the method-E models."""
    suffix = "" if model_name in (None, cfg["models"]["nli_model"]) else f"__{nli_model_slug(model_name)}"
    return cfg_path(cfg, "processed_dir") / "nli" / f"{split}{suffix}.jsonl"


def nli_fusion_alpha(cfg: dict) -> float:
    """Weight of the NLI probabilities in the fused teacher of method F: p = (1-α)·p_extractor + α·p_NLI."""
    teacher = cfg.get("teacher") or {}
    if "nli_fusion_alpha" not in teacher:
        raise SystemExit("方法 F 需要配置 teacher.nli_fusion_alpha（见 configs/config_f.yaml）")
    alpha = float(teacher["nli_fusion_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"teacher.nli_fusion_alpha must be in [0, 1], got {alpha}")
    return alpha


def backtranslation_path(cfg: dict, lang: str) -> Path:
    """Back-translation cache (tools/backtranslate.py): one JSON line per unit, {"src": original, "bt": back-translated}."""
    return cfg_path(cfg, "processed_dir") / "backtranslation" / f"{lang}.jsonl"


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

        self.tapt_encoder = self.checkpoints / "tapt" / "encoder"         # 方法 AT 的初始权重（跑完随 checkpoints/ 删除）
        self.tapt_outputs = self.outputs / "tapt"                          # TAPT 的训练记录
        self.tapt_probe = self.outputs / "tapt_probe"                      # 可行性测试（只有验证集结果）

        self.quality_report = self.outputs / "pseudo_label_quality.json"
        self.sanity_report = self.outputs / "sanity_check.json"
        self.import_marker = self.root / "imported_from.json"

    def pseudo_set(self, method: str) -> Path | None:
        """Pseudo-label file consumed by the detector for each ablation method."""
        mapping = {
            "A": None,
            "AC": None,
            "AT": None,
            "B": self.pseudo / "set_B_conf_top.jsonl",
            "Q": self.pseudo / "set_Q_conf_c_quantile.jsonl",
            "K": self.pseudo / "set_K_conf_topk.jsonl",
            "F": self.pseudo / "set_F_fused_top.jsonl",
            **{m: self.pseudo / f"set_{m}_fused_top.jsonl" for m in ALPHA_VARIANTS},
            "E": self.pseudo / "set_E_nli_ensemble.jsonl",
            **{m: None for m in SSL_METHODS},
            "W": self.pseudo / "set_W_weighted.jsonl",
            "R": self.pseudo / "set_R_random.jsonl",
            "C": self.rl_selected,
            "O": self.pseudo / "set_O_oracle.jsonl",
        }
        if method not in mapping:
            raise ValueError(f"Unknown method '{method}'. Expected one of {sorted(mapping)}")
        return mapping[method]

    def pool_nli_probs(self, model_name: str | None = None) -> Path:
        """NLI probabilities of the pool: nli_probs.jsonl for the LogicScore model (None), nli_probs__<model>.jsonl for the method-E models."""
        if model_name is None:
            return self.nli_probs
        return self.pseudo / f"nli_probs__{nli_model_slug(model_name)}.jsonl"

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
