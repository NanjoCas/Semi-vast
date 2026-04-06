"""
run_pipeline.py
===============
一键运行全部数据处理流程，并输出对接 DeBERTa 框架的 Dataset 类。

使用方法：
    python run_pipeline.py

输出目录结构：
    processed/
    ├── labeled/
    │   ├── train.jsonl          ← 有标签训练集（Climate-FEVER x2 + PUBHEALTH）
    │   ├── dev.jsonl            ← 有标签验证集
    │   ├── test.jsonl           ← 有标签测试集
    │   └── processing_stats.json
    └── unlabeled/
        ├── unlabeled_pool.jsonl ← 无标签声明池（Twitter + ClimateMiSt）
        ├── unlabeled_twitter.jsonl
        ├── unlabeled_climatemist.jsonl
        ├── climatemist_weak_labeled.jsonl（若ClimateMiSt已下载）
        └── processing_stats.json

依赖安装：
    pip install pandas scikit-learn torch transformers
"""

import json
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 导入处理模块
from process_labeled   import process_climate_fever, process_pubhealth, merge_labeled_datasets
from process_unlabeled import process_twitter_climate, process_climatemist, merge_unlabeled


OUTPUT_DIR  = "./processed"
LABEL2ID    = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}


# ═══════════════════════════════════════════════════════════════
# PyTorch Dataset 类（直接对接 DeBERTa 训练循环）
# ═══════════════════════════════════════════════════════════════

class ClaimEvidenceDataset(Dataset):
    """
    有标签数据集的 Dataset 类。

    输入格式：
        claim + [SEP] + evidence_1 + [SEP] + evidence_2 + ...
        （多条 evidence 用 [SEP] 拼接，DeBERTa 最大支持 512 token）

    使用示例：
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        ds = ClaimEvidenceDataset("processed/labeled/train.jsonl", tokenizer)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
    """

    def __init__(
        self,
        jsonl_path:    str,
        tokenizer,
        max_length:    int = 512,
        evidence_sep:  str = " [SEP] ",  # 多条 evidence 之间的分隔
        max_evidences: int = 3,          # 最多拼接几条 evidence
    ):
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.evidence_sep = evidence_sep
        self.max_evidences = max_evidences
        self.records = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"  加载 {len(self.records)} 条有标签记录 from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        claim    = record["claim"]
        evidences = record.get("evidence", [])[:self.max_evidences]
        label_str = record.get("label", "NOT_ENOUGH_INFO")

        # 拼接 evidence
        evidence_text = self.evidence_sep.join(evidences) if evidences else ""

        # Tokenize：claim 作为 sentence A，evidence 作为 sentence B
        encoding = self.tokenizer(
            claim,
            evidence_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(LABEL2ID[label_str], dtype=torch.long),
            "id":             record["id"],
            "source":         record.get("source", "unknown"),
        }


class UnlabeledClaimDataset(Dataset):
    """
    无标签数据集的 Dataset 类（用于伪标签生成）。

    使用示例：
        unlabeled_ds = UnlabeledClaimDataset(
            "processed/unlabeled/unlabeled_pool.jsonl", tokenizer
        )
        # 用 Textual Feature Extractor 推断伪标签
        loader = DataLoader(unlabeled_ds, batch_size=64)
    """

    def __init__(
        self,
        jsonl_path:  str,
        tokenizer,
        max_length:  int = 256,   # 无标签仅含 claim，可以用较短长度提高吞吐
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"  加载 {len(self.records)} 条无标签记录 from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        claim  = record["claim"]

        encoding = self.tokenizer(
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "id":             record["id"],
            "source":         record.get("source", "unknown"),
        }


# ═══════════════════════════════════════════════════════════════
# 数据统计报告
# ═══════════════════════════════════════════════════════════════

def print_dataset_report(labeled_dir: str, unlabeled_dir: str):
    """打印最终数据集统计报告。"""
    print("\n" + "=" * 60)
    print("数据集统计报告")
    print("=" * 60)

    # 有标签数据
    print("\n【有标签数据】")
    for split in ["train", "dev", "test"]:
        path = Path(labeled_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        label_dist = {}
        source_dist = {}
        for r in records:
            label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
            source_dist[r["source"]] = source_dist.get(r["source"], 0) + 1

        print(f"\n  {split}.jsonl：{len(records)} 条")
        print(f"    标签分布：")
        for label, count in sorted(label_dist.items()):
            print(f"      {label}: {count} ({count/len(records)*100:.1f}%)")
        print(f"    来源分布：{source_dist}")

    # 无标签数据
    print("\n【无标签数据池】")
    pool_path = Path(unlabeled_dir) / "unlabeled_pool.jsonl"
    if pool_path.exists():
        records = [json.loads(l) for l in open(pool_path) if l.strip()]
        source_dist = {}
        for r in records:
            source_dist[r["source"]] = source_dist.get(r["source"], 0) + 1
        print(f"\n  unlabeled_pool.jsonl：{len(records)} 条")
        print(f"    来源分布：{source_dist}")

    weak_path = Path(unlabeled_dir) / "climatemist_weak_labeled.jsonl"
    if weak_path.exists():
        records = [json.loads(l) for l in open(weak_path) if l.strip()]
        print(f"\n  climatemist_weak_labeled.jsonl：{len(records)} 条（弱标签验证集）")

    print("\n【数据规模建议】")
    print("  有标签：用于 Textual Feature Extractor 监督预训练")
    print("  无标签池：送入 Extractor 生成伪标签 → 再由 Reinforced Selector 筛选")
    print("  弱标签集：可作为环境域的 out-of-distribution 测试集")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("完整数据处理管道 — 一键运行")
    print("=" * 60)
    print("\n[阶段 1] 处理有标签数据集...\n")

    # Climate-FEVER
    try:
        print("  处理 Climate-FEVER...")
        cf = process_climate_fever()
        print(f"  ✓ Climate-FEVER：{cf['stats']['total']} 条")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        cf = {"train": [], "dev": [], "test": [], "stats": {}}

    # PUBHEALTH
    try:
        print("  处理 PUBHEALTH（过滤 COVID）...")
        ph = process_pubhealth()
        print(f"  ✓ PUBHEALTH：保留 {ph['stats']['total_kept']} 条（共 {ph['stats']['total_raw']} 条）")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        ph = {"train": [], "dev": [], "test": [], "stats": {}}

    # 合并有标签数据
    print("\n  合并并输出（Climate-FEVER 2x 过采样）...")
    merge_labeled_datasets(cf, ph, output_dir=OUTPUT_DIR, cf_weight=2.0)

    # ──────────────────────────────────────────────────────────
    print("\n[阶段 2] 处理无标签数据集...\n")

    # Twitter
    print("  处理 Climate Change Twitter Dataset...")
    tw_records, tw_stats = process_twitter_climate()
    if not tw_stats.get("skipped"):
        print(f"  ✓ Twitter：保留 {tw_stats['kept']} 条")

    # ClimateMiSt
    print("  处理 ClimateMiSt...")
    cm_unlabeled, cm_weak, cm_stats = process_climatemist()
    if not cm_stats.get("skipped"):
        print(f"  ✓ ClimateMiSt：无标签 {cm_stats['unlabeled_kept']} + 弱标签 {cm_stats['weak_labeled']} 条")
        if cm_weak:
            from process_unlabeled import OUTPUT_DIR as UL_DIR
            weak_path = Path(UL_DIR) / "unlabeled" / "climatemist_weak_labeled.jsonl"
            weak_path.parent.mkdir(parents=True, exist_ok=True)
            with open(weak_path, "w", encoding="utf-8") as f:
                for r in cm_weak:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 合并无标签池
    print("\n  合并输出无标签池...")
    merge_unlabeled(tw_records, cm_unlabeled, output_dir=OUTPUT_DIR)

    # ──────────────────────────────────────────────────────────
    print_dataset_report(
        labeled_dir  =f"{OUTPUT_DIR}/labeled",
        unlabeled_dir=f"{OUTPUT_DIR}/unlabeled",
    )

    # ──────────────────────────────────────────────────────────
    print("\n[阶段 3] 验证 Dataset 类（快速 sanity check）...\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        train_path = f"{OUTPUT_DIR}/labeled/train.jsonl"
        pool_path  = f"{OUTPUT_DIR}/unlabeled/unlabeled_pool.jsonl"

        if Path(train_path).exists():
            ds = ClaimEvidenceDataset(train_path, tokenizer, max_length=512)
            sample = ds[0]
            print(f"  ✓ ClaimEvidenceDataset：input_ids shape={sample['input_ids'].shape}, label={sample['label']}")

        if Path(pool_path).exists():
            uds = UnlabeledClaimDataset(pool_path, tokenizer, max_length=256)
            usample = uds[0]
            print(f"  ✓ UnlabeledClaimDataset：input_ids shape={usample['input_ids'].shape}, id={usample['id']}")

    except Exception as e:
        print(f"  [跳过验证] transformers 未安装或模型未缓存：{e}")
        print("  → 安装：pip install transformers torch")

    print("\n" + "=" * 60)
    print("✓ 全部处理完成！")
    print(f"  有标签数据：{OUTPUT_DIR}/labeled/")
    print(f"  无标签数据：{OUTPUT_DIR}/unlabeled/")
    print("=" * 60)


if __name__ == "__main__":
    main()
