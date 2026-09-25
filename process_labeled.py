"""
process_labeled.py
==================
处理有标签数据集：Climate-FEVER + PUBHEALTH
输出统一格式的 JSONL 文件，供 DeBERTa Textual Feature Extractor 训练使用。

统一输出格式（每行一个 JSON）：
{
    "id":       "cf_75",              # 数据集前缀 + 原始ID
    "claim":    "Arctic ice is melting faster...",
    "evidence": ["Sea ice extent has declined...", "..."],
    "label":    "SUPPORTS",           # 统一三分类：SUPPORTS / REFUTES / NOT_ENOUGH_INFO
    "source":   "climate_fever"
}
"""

import json
import re
import random
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# 路径配置（根据实际下载路径修改）
# ─────────────────────────────────────────────────────────────
CLIMATE_FEVER_JSONL  = "./Data/Climate Fever Dataset/archive/climate-fever.jsonl"
PUBHEALTH_DIR        = "./Data/PUBHEALTH-DATASET/archive"
OUTPUT_DIR           = "./processed"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════
# 1. Climate-FEVER 处理
# ═══════════════════════════════════════════════════════════════

# Climate-FEVER 有4个标签，需要映射到3分类
CLIMATE_FEVER_LABEL_MAP = {
    "SUPPORTS":        "SUPPORTS",
    "REFUTES":         "REFUTES",
    "DISPUTED":        "REFUTES",       # 存在反驳证据 → 归为 REFUTES
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
}


def _select_evidence_for_claim(evidences: list, claim_label: str) -> list:
    """
    从5条候选证据中选取最相关的，优先选与 claim_label 一致的证据。

    策略：
    1. 优先选与 claim 标签匹配的 evidence（如 claim=SUPPORTS → 选 evidence_label=SUPPORTS）
    2. 若无匹配，则取全部有效证据
    3. 最多保留 3 条，避免噪声过多
    """
    mapped_label = CLIMATE_FEVER_LABEL_MAP.get(claim_label, "NOT_ENOUGH_INFO")

    # 与 claim 标签一致的证据
    matching = [
        ev["evidence"] for ev in evidences
        if ev.get("evidence_label") == claim_label
        and ev.get("evidence", "").strip()
    ]

    # 其他有效证据（补充用）
    others = [
        ev["evidence"] for ev in evidences
        if ev.get("evidence_label") != claim_label
        and ev.get("evidence", "").strip()
    ]

    selected = matching if matching else others
    return selected[:3]


def process_climate_fever(
    input_path: str = CLIMATE_FEVER_JSONL,
    test_size: float = 0.15,
    dev_size: float = 0.15,
) -> dict:
    """
    加载并处理 Climate-FEVER，自行切分 train/dev/test（官方无预设划分）。

    Returns:
        dict: {"train": [...], "dev": [...], "test": [...], "stats": {...}}
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Climate-FEVER 文件未找到：{input_path}\n"
            "请先运行 download_datasets.sh 下载数据集。"
        )

    raw_records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            claim  = item.get("claim", "").strip()
            c_label = item.get("claim_label", "NOT_ENOUGH_INFO")
            label  = CLIMATE_FEVER_LABEL_MAP.get(c_label, "NOT_ENOUGH_INFO")
            evs    = item.get("evidences", [])
            evidence_texts = _select_evidence_for_claim(evs, c_label)

            if not claim or not evidence_texts:
                continue  # 跳过无效条目

            raw_records.append({
                "id":       f"cf_{item['claim_id']}",
                "claim":    claim,
                "evidence": evidence_texts,
                "label":    label,
                "source":   "climate_fever",
            })

    # 按标签分层切分，保证各 split 标签分布均衡
    labels = [r["label"] for r in raw_records]
    train_data, temp_data, _, temp_labels = train_test_split(
        raw_records, labels,
        test_size=(test_size + dev_size),
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    dev_ratio_of_temp = dev_size / (test_size + dev_size)
    dev_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - dev_ratio_of_temp),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    stats = {
        "total": len(raw_records),
        "train": len(train_data),
        "dev":   len(dev_data),
        "test":  len(test_data),
        "label_dist": dict(Counter(labels)),
    }

    return {"train": train_data, "dev": dev_data, "test": test_data, "stats": stats}


# ═══════════════════════════════════════════════════════════════
# 2. PUBHEALTH 处理
# ═══════════════════════════════════════════════════════════════

# PUBHEALTH 4分类 → 3分类映射
PUBHEALTH_LABEL_MAP = {
    "true":     "SUPPORTS",
    "false":    "REFUTES",
    "unproven": "NOT_ENOUGH_INFO",
    "mixture":  "NOT_ENOUGH_INFO",   # 部分真实，信息不足以确定 → NEI
}

# COVID 相关关键词正则（用于过滤）
COVID_PATTERN = re.compile(
    r"\b(covid|coronavirus|sars[\-\s]?cov[\-\s]?2?|pandemic|"
    r"lockdown|quarantine|mask mandate|pcr test|contact tracing|"
    r"mRNA vaccine|pfizer|moderna|astrazeneca|johnson.*johnson)\b",
    re.IGNORECASE,
)


def _is_covid_related(text: str) -> bool:
    """判断文本是否涉及 COVID-19。"""
    return bool(COVID_PATTERN.search(str(text)))


def _clean_text(text: str) -> str:
    """基础文本清洗：去除多余空白、HTML残留等。"""
    text = str(text).strip()
    text = re.sub(r"<[^>]+>", " ", text)      # 去HTML标签
    text = re.sub(r"\s{2,}", " ", text)        # 压缩多余空格
    return text


def process_pubhealth(
    data_dir: str = PUBHEALTH_DIR,
) -> dict:
    """
    加载并处理 PUBHEALTH（train/dev/test 已预设），过滤 COVID 条目并统一标签。

    PUBHEALTH 特殊性：
    - evidence 来源是记者撰写的"解释文本"（explanation），而非独立证据句
    - 这与 Climate-FEVER 的 Wikipedia 证据风格不同，需在预训练时注意
    - 解释文本作为 evidence[0] 传入，长度通常较长（100~500词）

    Returns:
        dict: {"train": [...], "dev": [...], "test": [...], "stats": {...}}
    """
    data_dir = Path(data_dir)
    result = {"stats": {}}
    total_raw, total_kept = 0, 0

    # Handle missing dev.tsv by splitting from train.tsv
    available_splits = [s for s in ["train", "dev", "test"]
                        if (data_dir / f"{s}.tsv").exists()]
    if "dev" not in available_splits and "train" in available_splits:
        print("  [PUBHEALTH] dev.tsv 未找到，将从 train.tsv 中切分 15% 作为验证集")

    # Pre-load train data and split if dev is missing
    train_df_for_split = None
    dev_df_from_split = None
    if "dev" not in available_splits and "train" in available_splits:
        _full_train = pd.read_csv(data_dir / "train.tsv", sep="\t", dtype=str)
        total_raw += len(_full_train)
        _full_train = _full_train[_full_train["label"].isin(PUBHEALTH_LABEL_MAP.keys())]
        labels_for_split = _full_train["label"].tolist()
        train_df_for_split, dev_df_from_split = train_test_split(
            _full_train, test_size=0.15, stratify=labels_for_split, random_state=RANDOM_SEED
        )

    for split in ["train", "dev", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists() and split == "dev" and dev_df_from_split is not None:
            df = dev_df_from_split
        elif not tsv_path.exists() and split == "train" and train_df_for_split is not None:
            df = train_df_for_split
        elif not tsv_path.exists():
            print(f"  [跳过] PUBHEALTH {split}.tsv 未找到：{tsv_path}")
            result[split] = []
            continue
        else:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str)
            # Only count toward total_raw if we didn't already count this file
            # (train.tsv is pre-counted when doing the dev split)
            if not (train_df_for_split is not None and split == "train"):
                total_raw += len(df)

        # ── 过滤步骤 ──────────────────────────────────────────
        # 1. 仅保留有效标签
        df = df[df["label"].isin(PUBHEALTH_LABEL_MAP.keys())]

        # 2. 过滤 COVID 相关条目（同时检查 claim 和 subjects）
        subjects_col = "subjects" if "subjects" in df.columns else None
        def is_covid_row(row):
            if _is_covid_related(row.get("claim", "")):
                return True
            if subjects_col and _is_covid_related(row.get(subjects_col, "")):
                return True
            return False
        df = df[~df.apply(is_covid_row, axis=1)]

        # 3. 过滤空 claim 或空 explanation
        df = df.dropna(subset=["claim"])
        if "explanation" in df.columns:
            df = df[df["explanation"].notna() & (df["explanation"].str.strip() != "")]

        # ── 构建记录 ─────────────────────────────────────────
        records = []
        for _, row in df.iterrows():
            claim = _clean_text(row.get("claim", ""))
            explanation = _clean_text(row.get("explanation", ""))

            # explanation 截断：最多保留 512 个词（避免过长）
            words = explanation.split()
            if len(words) > 512:
                explanation = " ".join(words[:512]) + "..."

            if not claim or not explanation:
                continue

            records.append({
                "id":       f"ph_{row.get('claim_id', hash(claim))}",
                "claim":    claim,
                "evidence": [explanation],
                "label":    PUBHEALTH_LABEL_MAP[row["label"]],
                "source":   "pubhealth",
            })

        result[split] = records
        total_kept += len(records)

    result["stats"] = {
        "total_raw":  total_raw,
        "total_kept": total_kept,
        "removed_covid_or_invalid": total_raw - total_kept,
        "split_sizes": {
            s: len(result.get(s, [])) for s in ["train", "dev", "test"]
        },
        "label_dist": dict(Counter(
            r["label"]
            for s in ["train", "dev", "test"]
            for r in result.get(s, [])
        )),
    }

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 合并两个数据集
# ═══════════════════════════════════════════════════════════════

def merge_labeled_datasets(
    cf_result: dict,
    ph_result: dict,
    output_dir: str = OUTPUT_DIR,
    cf_weight: float = 1.0,
    ph_weight: float = 1.0,
) -> dict:
    """
    合并 Climate-FEVER 与 PUBHEALTH，按 split 输出 JSONL 文件。

    Args:
        cf_weight: Climate-FEVER 过采样倍数（若要强化气候域，可设为 2.0）
        ph_weight: PUBHEALTH 过采样倍数

    说明：
    - Climate-FEVER 体量小（~1500条），PUBHEALTH 较大（~8000+条）
    - 默认等权重合并；若气候域是核心，可设 cf_weight=2.0 进行过采样
    """
    output_dir = Path(output_dir) / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ["train", "dev", "test"]:
        cf_split = cf_result.get(split, [])
        ph_split = ph_result.get(split, [])

        # 过采样（整数倍重复 + shuffle）
        cf_samples = cf_split * int(cf_weight) if cf_weight >= 1 else \
                     random.sample(cf_split, int(len(cf_split) * cf_weight))
        ph_samples = ph_split * int(ph_weight) if ph_weight >= 1 else \
                     random.sample(ph_split, int(len(ph_split) * ph_weight))

        combined = cf_samples + ph_samples
        random.shuffle(combined)

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in combined:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        label_dist = dict(Counter(r["label"] for r in combined))
        stats[split] = {
            "total":      len(combined),
            "from_cf":    len(cf_samples),
            "from_ph":    len(ph_samples),
            "label_dist": label_dist,
        }
        print(f"  [{split}] 共 {len(combined)} 条 → {out_path}")
        print(f"          Climate-FEVER: {len(cf_samples)} | PUBHEALTH: {len(ph_samples)}")
        print(f"          标签分布: {label_dist}")

    return stats


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("有标签数据处理管道")
    print("=" * 60)

    # ── Climate-FEVER ─────────────────────────────────────────
    print("\n[1/3] 处理 Climate-FEVER...")
    cf = process_climate_fever()
    print(f"  总计: {cf['stats']['total']} 条")
    print(f"  划分: train={cf['stats']['train']}, dev={cf['stats']['dev']}, test={cf['stats']['test']}")
    print(f"  标签: {cf['stats']['label_dist']}")

    # ── PUBHEALTH ─────────────────────────────────────────────
    print("\n[2/3] 处理 PUBHEALTH（过滤 COVID）...")
    ph = process_pubhealth()
    print(f"  原始: {ph['stats']['total_raw']} 条")
    print(f"  保留: {ph['stats']['total_kept']} 条（过滤 {ph['stats']['removed_covid_or_invalid']} 条）")
    print(f"  划分: {ph['stats']['split_sizes']}")
    print(f"  标签: {ph['stats']['label_dist']}")

    # ── 合并输出 ──────────────────────────────────────────────
    print("\n[3/3] 合并输出（Climate-FEVER 2x 过采样，弥补体量差异）...")
    merge_stats = merge_labeled_datasets(cf, ph, cf_weight=2.0, ph_weight=1.0)

    # 保存统计信息
    stats_path = Path(OUTPUT_DIR) / "labeled" / "processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "climate_fever": cf["stats"],
            "pubhealth": ph["stats"],
            "merged": merge_stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 统计信息已保存至 {stats_path}")
    print("\n输出文件：")
    print(f"  {OUTPUT_DIR}/labeled/train.jsonl")
    print(f"  {OUTPUT_DIR}/labeled/dev.jsonl")
    print(f"  {OUTPUT_DIR}/labeled/test.jsonl")


if __name__ == "__main__":
    main()
