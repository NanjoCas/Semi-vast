"""
process_unlabeled.py
====================
处理无标签数据集：Climate Change Twitter Dataset + ClimateMiSt
输出供 Reinforced Selector 生成伪标签的无标签声明池。

统一输出格式（每行一个 JSON）：
{
    "id":     "tw_00123",
    "claim":  "New research shows Arctic ice melting at unprecedented rate",
    "source": "twitter_climate"
}

注意：
- Twitter 数据集因版权限制，Kaggle 版仅含推文ID，需先 rehydrate 获取文本
  → 若已有完整文本（Mendeley版），直接读取 CSV 即可
- ClimateMiSt 有标注子集（2008条）→ 可同时输出为"弱标签验证集"
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

# Twitter 气候数据集（Mendeley 完整版 CSV）
TWITTER_CSV         = "./Data/the-climate-change-twitter-dataset/the-climate-change-twitter-dataset/The Climate Change Twitter Dataset.csv"

# ClimateMiSt 推文文件（JSON 或 CSV）
CLIMATEMIST_TWEETS  = "./datasets/climate_twitter/climatemist_tweets.json"
CLIMATEMIST_NEWS    = "./datasets/climate_twitter/climatemist_news.json"

# Environment News Dataset（Guardian 环境新闻）
ENVIRONMENT_NEWS_CSV = "./Data/Environment News Dataset/guardian_environment_news.csv"

# 输出目录
OUTPUT_DIR          = "./processed"

# 处理参数
MIN_CLAIM_LENGTH    = 20     # 最短声明字符数（太短缺乏语义）
MAX_CLAIM_LENGTH    = 512    # 最长声明字符数（超长截断）
TWITTER_SAMPLE_SIZE = 200000 # 从 1500 万推文中采样数量（避免过大）


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
# 1. Climate Change Twitter Dataset 处理
# ═══════════════════════════════════════════════════════════════

def process_twitter_climate(
    csv_path: str = TWITTER_CSV,
    sample_size: int = TWITTER_SAMPLE_SIZE,
) -> tuple[list, dict]:
    """
    处理 Climate Change Twitter Dataset（Mendeley 完整版 CSV）。

    数据集结构（主要列）：
    - id          : 推文 ID
    - message     : 推文文本
    - sentiment   : 立场（-1=否认, 0=中立, 1=支持, 2=新闻转发）
    - tweetid     : Twitter 原始 ID

    策略：
    - 将推文文本作为无标签声明
    - 忽略 sentiment 标签（全部视为待伪标注）
    - 优先保留 sentiment=2（新闻类，更接近事实声明）和 sentiment=1（支持气候变化，可能含有效信息）
    - 过滤 sentiment=-1（否认类，存在更多噪声）→ 可选

    Returns:
        (records_list, stats_dict)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [跳过] Twitter CSV 未找到：{csv_path}")
        print("  → Mendeley 版下载地址：https://data.mendeley.com/datasets/mw8yd7z9wc/2")
        print("  → 或使用 Kaggle 版（仅 ID）后通过 rehydrate_tweets.py 获取文本")
        return [], {"skipped": True}

    print(f"  读取 {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)

    # 确定文本列名（不同版本可能不同）
    text_col = None
    for col in ["message", "text", "tweet", "full_text", "body"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        print(f"  [跳过] 未找到文本列，现有列：{list(df.columns)}")
        print("  → 当前 Twitter CSV 仅包含元数据（无推文文本），需要 Mendeley 完整版")
        return [], {"skipped": True}

    # 可选：按立场过滤（保留中立和新闻类，排除强烈否认）
    if "sentiment" in df.columns:
        # sentiment=2: 新闻类 | sentiment=1: 相信气候变化 | sentiment=0: 中立
        # sentiment=-1: 否认（保留，但体量控制）
        priority_df = df[df["sentiment"].isin(["2", "1", "0"])]
        deny_df     = df[df["sentiment"] == "-1"]

        # 优先取高质量，其余采样填充
        n_deny = min(len(deny_df), sample_size // 5)  # 否认类最多占 20%
        n_priority = sample_size - n_deny

        sample_df = pd.concat([
            priority_df.sample(n=min(n_priority, len(priority_df)), random_state=42),
            deny_df.sample(n=min(n_deny, len(deny_df)), random_state=42)
        ])
    else:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # 清洗与过滤
    records = []
    seen_texts = set()
    for _, row in sample_df.iterrows():
        raw_text = str(row.get(text_col, ""))
        cleaned  = _clean_tweet(raw_text)

        if not _is_valid_claim(cleaned):
            continue
        if cleaned in seen_texts:   # 去重
            continue

        claim = _truncate_claim(cleaned)
        seen_texts.add(cleaned)

        records.append({
            "id":     _gen_id("tw", claim),
            "claim":  claim,
            "source": "twitter_climate",
        })

    stats = {
        "raw_sample":  len(sample_df),
        "kept":        len(records),
        "dedup_ratio": f"{len(records)/max(len(sample_df),1)*100:.1f}%",
    }
    return records, stats


# ═══════════════════════════════════════════════════════════════
# 2. ClimateMiSt 处理（推文 + 新闻）
# ═══════════════════════════════════════════════════════════════

# ClimateMiSt 标签映射（用于有标注子集 → 弱标签验证集）
CLIMATEMIST_LABEL_MAP = {
    "misinformation":     "REFUTES",      # 虚假信息 → REFUTES
    "non-misinformation": "SUPPORTS",     # 非虚假 → SUPPORTS
    "true":               "SUPPORTS",
    "false":              "REFUTES",
    "unknown":            "NOT_ENOUGH_INFO",
}


def _load_climatemist_json(json_path: str) -> list:
    """加载 ClimateMiSt JSON 文件（支持 JSON array 和 JSONL 格式）。"""
    path = Path(json_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    else:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def process_climatemist(
    tweets_path: str = CLIMATEMIST_TWEETS,
    news_path:   str = CLIMATEMIST_NEWS,
) -> tuple[list, list, dict]:
    """
    处理 ClimateMiSt 数据集。

    数据集结构（参考 IEEE DataPort 文档）：
    推文文件包含：tweet_id, text, label (部分有标注), stance
    新闻文件包含：article_id, title, body, source

    Returns:
        (unlabeled_records, weak_labeled_records, stats)
        - unlabeled_records   : 无标签声明（用于半监督学习）
        - weak_labeled_records: 有标注子集（2008条，可作弱标签验证集）
    """
    tweets_raw = _load_climatemist_json(tweets_path)
    news_raw   = _load_climatemist_json(news_path)

    if not tweets_raw and not news_raw:
        print(f"  [跳过] ClimateMiSt 文件未找到")
        print("  → 下载地址（需注册 IEEE DataPort）：")
        print("    https://ieee-dataport.org/documents/climatemist-climate-change-misinformation-and-stance-detection-dataset")
        print("    DOI: 10.21227/cdaz-jh77")
        return [], [], {"skipped": True}

    unlabeled  = []
    weak_labeled = []
    seen_texts = set()

    # ── 处理推文 ────────────────────────────────────────────
    text_keys = ["text", "tweet_text", "full_text", "content"]

    for item in tweets_raw:
        raw_text = ""
        for k in text_keys:
            if k in item and item[k]:
                raw_text = str(item[k])
                break
        if not raw_text:
            continue

        cleaned = _clean_tweet(raw_text)
        if not _is_valid_claim(cleaned) or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)

        claim = _truncate_claim(cleaned)
        item_id = _gen_id("cm_tw", claim)

        # 检查是否有标注（有标注子集 → 弱标签验证集）
        raw_label = str(item.get("label", item.get("veracity", ""))).lower().strip()
        mapped_label = CLIMATEMIST_LABEL_MAP.get(raw_label, "")

        if mapped_label:
            weak_labeled.append({
                "id":     item_id,
                "claim":  claim,
                "evidence": [],   # 无 evidence，标签来自人工标注
                "label":  mapped_label,
                "source": "climatemist_tweet",
            })
        else:
            unlabeled.append({
                "id":     item_id,
                "claim":  claim,
                "source": "climatemist_tweet",
            })

    # ── 处理新闻文章 ─────────────────────────────────────────
    title_keys = ["title", "headline"]
    body_keys  = ["body", "content", "article", "text"]

    for item in news_raw:
        # 优先使用标题（更接近声明形式），正文作为补充
        title = ""
        for k in title_keys:
            if k in item and item[k]:
                title = _clean_news(str(item[k]))
                break

        if not title or not _is_valid_claim(title) or title in seen_texts:
            continue
        seen_texts.add(title)

        unlabeled.append({
            "id":     _gen_id("cm_news", title),
            "claim":  _truncate_claim(title),
            "source": "climatemist_news",
        })

    stats = {
        "tweets_raw":      len(tweets_raw),
        "news_raw":        len(news_raw),
        "unlabeled_kept":  len(unlabeled),
        "weak_labeled":    len(weak_labeled),
    }
    return unlabeled, weak_labeled, stats


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
    df = pd.read_csv(csv_path, dtype=str)
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

    # ── Twitter 气候数据集 ────────────────────────────────────
    print("\n[1/4] 处理 Climate Change Twitter Dataset...")
    tw_records, tw_stats = process_twitter_climate(sample_size=200_000)
    if not tw_stats.get("skipped"):
        print(f"  原始采样：{tw_stats['raw_sample']} | 保留：{tw_stats['kept']} | 去重率：{tw_stats['dedup_ratio']}")

    # ── ClimateMiSt ───────────────────────────────────────────
    print("\n[2/4] 处理 ClimateMiSt（推文 + 新闻）...")
    cm_unlabeled, cm_weak, cm_stats = process_climatemist()
    if not cm_stats.get("skipped"):
        print(f"  推文原始：{cm_stats['tweets_raw']} | 新闻原始：{cm_stats['news_raw']}")
        print(f"  无标签保留：{cm_stats['unlabeled_kept']} | 弱标签集：{cm_stats['weak_labeled']}")

        # 保存弱标签集（ClimateMiSt 有标注子集）
        if cm_weak:
            weak_path = Path(OUTPUT_DIR) / "unlabeled" / "climatemist_weak_labeled.jsonl"
            weak_path.parent.mkdir(parents=True, exist_ok=True)
            with open(weak_path, "w", encoding="utf-8") as f:
                for r in cm_weak:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  弱标签集已保存：{weak_path}（可用于域外验证）")

    # ── Environment News Dataset ─────────────────────────────────
    print("\n[3/4] 处理 Environment News Dataset（Guardian 环境新闻）...")
    env_records, env_stats = process_environment_news()
    if not env_stats.get("skipped"):
        print(f"  原始行数：{env_stats['raw_rows']} | 保留：{env_stats['kept']} | 去重率：{env_stats['dedup_ratio']}")

    # ── 合并输出 ──────────────────────────────────────────────
    print("\n[4/4] 合并输出无标签池...")
    source_records = {
        "twitter": tw_records,
        "climatemist": cm_unlabeled,
        "environment_news": env_records,
    }
    merge_stats = merge_unlabeled(source_records)

    # 保存统计
    stats_path = Path(OUTPUT_DIR) / "unlabeled" / "processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "twitter": tw_stats,
            "climatemist": cm_stats,
            "environment_news": env_stats,
            "merged": merge_stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 统计信息已保存至 {stats_path}")
    print("\n输出文件：")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_pool.jsonl   ← 主要训练输入")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_twitter.jsonl")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_climatemist.jsonl")
    print(f"  {OUTPUT_DIR}/unlabeled/unlabeled_environment_news.jsonl")
    if not cm_stats.get("skipped") and cm_weak:
        print(f"  {OUTPUT_DIR}/unlabeled/climatemist_weak_labeled.jsonl ← 弱标签验证集")


if __name__ == "__main__":
    main()
