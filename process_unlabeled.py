"""
process_unlabeled.py
====================
处理无标签数据集：Guardian Environment News Dataset
输出供 Reinforced Selector 生成伪标签的无标签声明池。

统一输出格式（每行一个 JSON）：
{
    "id":     "env_00123",
    "claim":  "A new environmental policy aims to reduce plastic waste",
    "source": "guardian_environment_news"
}

注意：
- 只处理当前项目已有的环境新闻数据集
"""

import json
import re
import hashlib
import pandas as pd
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────
# 路径配置（根据实际文件位置修改）
# ─────────────────────────────────────────────────────────────



# Environment News Dataset（Guardian 环境新闻）
ENVIRONMENT_NEWS_CSV = "./Data/Environment News Dataset/guardian_environment_news.csv"

# 输出目录
OUTPUT_DIR          = "./processed"

# 处理参数
MIN_CLAIM_LENGTH    = 20     # 最短声明字符数（太短缺乏语义）
MAX_CLAIM_LENGTH    = 512    # 最长声明字符数（超长截断）


# ─────────────────────────────────────────────────────────────
# 通用工具函数
# ─────────────────────────────────────────────────────────────

# 需要过滤的无效推文模式
NOISE_PATTERNS = [
    re.compile(r"^RT @\w+:", re.IGNORECASE),          # 转推（原文可能已处理）
    re.compile(r"^@\w+", re.IGNORECASE),               # @回复
    re.compile(r"https?://\S+", re.IGNORECASE),        # 纯链接
    re.compile(r"#\w+(\s+#\w+)+", re.IGNORECASE),     # 连续 hashtag（缺乏实质内容）
]

URL_PATTERN     = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")  # 保留词，去#号


def _gen_id(prefix: str, text: str) -> str:
    """基于文本内容生成稳定 ID（防止重复）。"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"


def _clean_tweet(text: str) -> str:
    """
    清洗推文文本：
    1. 去除 URL
    2. 去除 @提及
    3. 将 #hashtag 转为普通词（保留词义）
    4. 压缩空白
    """
    text = str(text).strip()
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)      # #ClimateChange → ClimateChange
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _clean_news(text: str) -> str:
    """清洗新闻标题/正文：去除 HTML、多余空白等。"""
    text = str(text).strip()
    text = re.sub(r"<[^>]+>", " ", text)        # 去 HTML 标签
    text = re.sub(r"&[a-z]+;", " ", text)        # 去 HTML 实体
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _is_valid_claim(text: str) -> bool:
    """判断文本是否适合作为声明（基本质量过滤）。"""
    if not text or len(text) < MIN_CLAIM_LENGTH:
        return False
    # 检查是否匹配噪声模式
    for pattern in NOISE_PATTERNS:
        # 如果整个文本主要是噪声
        cleaned = pattern.sub("", text).strip()
        if len(cleaned) < MIN_CLAIM_LENGTH:
            return False
    # 过滤非英文（简单启发：英文字母占比需>60%）
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / max(len(text), 1) < 0.4:
        return False
    return True


def _truncate_claim(text: str, max_len: int = MAX_CLAIM_LENGTH) -> str:
    """截断过长文本，在句子边界处截断。"""
    if len(text) <= max_len:
        return text
    # 在句子末尾截断
    truncated = text[:max_len]
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > max_len * 0.7:
        return truncated[:last_period + 1]
    return truncated + "..."


# ═══════════════════════════════════════════════════════════════
# 3. Environment News Dataset 处理
# ═══════════════════════════════════════════════════════════════════════

def process_environment_news(
    csv_path: str = ENVIRONMENT_NEWS_CSV,
    sample_size: int = 100_000,
) -> tuple[list, dict]:
    """
    处理 Guardian 环境新闻数据集。

    数据集结构（主要列）：
    - Title         : 新闻标题
    - Intro Text    : 摘要或导语
    - Article Text  : 正文
    - Date Published: 发布日期

    策略：
    - 优先使用标题作为声明
    - 标题无效时回退到导语，再回退到正文前几句
    - 为避免无关长文本，正文中只取前 1-2 句
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [跳过] Environment News CSV 未找到：{csv_path}")
        return [], {"skipped": True}

    print(f"  读取 {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str, engine='python', on_bad_lines='skip')
    if df.empty:
        print(f"  [跳过] Environment News CSV 内容为空：{csv_path}")
        return [], {"skipped": True}

    records = []
    seen_texts = set()

    def _pick_text(row):
        for key in ["Title", "Intro Text", "Article Text"]:
            if key not in row or pd.isna(row[key]):
                continue
            text = str(row[key]).strip()
            if not text:
                continue

            if key == "Article Text":
                # 正文可能很长，取前 1-2 句以保持声明性质
                sentences = re.split(r"(?<=[。.!?])\s+", text)
                text = " ".join(sentences[:2])

            return _clean_news(text)
        return ""

    for _, row in df.sample(n=min(sample_size, len(df)), random_state=42).iterrows():
        cleaned = _pick_text(row)
        if not cleaned or not _is_valid_claim(cleaned) or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)

        records.append({
            "id":     _gen_id("env", cleaned),
            "claim":  _truncate_claim(cleaned),
            "source": "guardian_environment_news",
        })

    stats = {
        "raw_rows": len(df),
        "kept":     len(records),
        "dedup_ratio": f"{len(records)/max(len(df),1)*100:.1f}%",
    }
    return records, stats


# ═══════════════════════════════════════════════════════════════════════
# 4. 合并输出无标签池
# ═══════════════════════════════════════════════════════════════

def merge_unlabeled(
    source_records: dict,
    output_dir:        str = OUTPUT_DIR,
) -> dict:
    """
    合并所有无标签数据，输出 JSONL。

    输出文件：
    - unlabeled_pool.jsonl      : 全量无标签声明（用于伪标签生成）
    - unlabeled_<source>.jsonl : 各来源的单独输出
    """
    out_dir = Path(output_dir) / "unlabeled"
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    combined = []
    for records in source_records.values():
        combined.extend(records)
    random.shuffle(combined)

    pool_path = out_dir / "unlabeled_pool.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for source, records in source_records.items():
        file_name = f"unlabeled_{source}.jsonl"
        out_path = out_dir / file_name
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    source_dist = dict(Counter(r["source"] for r in combined))
    stats = {"total": len(combined), "source_dist": source_dist}
    stats.update({source: len(records) for source, records in source_records.items()})

    print(f"  无标签池总量：{len(combined)} 条 → {pool_path}")
    print(f"  来源分布：{source_dist}")
    return stats


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("无标签数据处理管道")
    print("=" * 60)

    # ── Environment News Dataset ─────────────────────────────────
    print("\n[1/1] 处理 Environment News Dataset（Guardian 环境新闻）...")
    env_records, env_stats = process_environment_news()
    if not env_stats.get("skipped"):
        print(f"  原始行数：{env_stats['raw_rows']} | 保留：{env_stats['kept']} | 去重率：{env_stats['dedup_ratio']}")

    # ── 合并输出 ──────────────────────────────────────────────
    print("\n[2/2] 合并输出无标签池...")
    source_records = {
        "environment_news": env_records,
    }
    merge_stats = merge_unlabeled(source_records)

    # 保存统计
    stats_path = Path(OUTPUT_DIR) / "unlabeled" / "processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "environment_news": env_stats,
            "merged": merge_stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 统计信息已保存至 {stats_path}")
    print("\n输出文件：")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_pool.jsonl   ← 主要训练输入")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_environment_news.jsonl")


if __name__ == "__main__":
    main()
