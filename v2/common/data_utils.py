"""
Shared datasets and JSONL helpers for v2.

v1 的问题：无标签样本只有 claim（UnlabeledClaimDataset / PseudoClaimDataset），
而验证/测试是 claim + evidence 句对，伪标签数据与评估任务不一致。
v2 中所有样本（有标签、无标签、伪标签）都使用同一种句对编码：ClaimEvidenceDataset。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def label_to_id(raw) -> int:
    """Accept int ids or string labels; unknown values map to NOT_ENOUGH_INFO."""
    if isinstance(raw, bool):
        return LABEL2ID["NOT_ENOUGH_INFO"]
    if isinstance(raw, int) and 0 <= raw < NUM_LABELS:
        return raw
    if isinstance(raw, str):
        return LABEL2ID.get(raw, LABEL2ID["NOT_ENOUGH_INFO"])
    return LABEL2ID["NOT_ENOUGH_INFO"]


def record_label_id(rec: dict) -> int:
    """Resolve a record's training label: pseudo_label first, then label."""
    if rec.get("pseudo_label") is not None:
        return label_to_id(rec["pseudo_label"])
    return label_to_id(rec.get("label", "NOT_ENOUGH_INFO"))


class ClaimEvidenceDataset(Dataset):
    """
    Claim + evidence pair dataset used for every channel in v2.

    Works for three kinds of records:
      - labeled:     {"claim", "evidence", "label"}
      - unlabeled:   {"claim", "evidence"}                (label returned as -1)
      - pseudo:      {"claim", "evidence", "pseudo_label", "weight"}

    Encoding is identical to v1's ClaimEvidenceDataset in run_pipeline.py:
    claim as sentence A, evidences joined by " [SEP] " as sentence B.
    """

    def __init__(
        self,
        source: str | Path | list[dict],
        tokenizer,
        max_length: int = 512,
        evidence_sep: str = " [SEP] ",
        max_evidences: int = 3,
        require_label: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.evidence_sep = evidence_sep
        self.max_evidences = max_evidences
        self.require_label = require_label
        self.records: list[dict] = load_jsonl(source) if isinstance(source, (str, Path)) else list(source)

        self.label_ids: list[int] = []
        for rec in self.records:
            has_label = rec.get("pseudo_label") is not None or rec.get("label") is not None
            if require_label and not has_label:
                raise ValueError(f"Record {rec.get('id')} has no label/pseudo_label")
            self.label_ids.append(record_label_id(rec) if has_label else -1)

    def __len__(self) -> int:
        return len(self.records)

    def evidence_text(self, rec: dict) -> str:
        evidences = rec.get("evidence", []) or []
        if isinstance(evidences, str):
            evidences = [evidences]
        return self.evidence_sep.join(evidences[: self.max_evidences])

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        encoding = self.tokenizer(
            rec.get("claim", ""),
            self.evidence_text(rec),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "label": torch.tensor(self.label_ids[idx], dtype=torch.long),
            "weight": torch.tensor(float(rec.get("weight", 1.0)), dtype=torch.float),
            "id": str(rec.get("id", idx)),
            "source": rec.get("source", "unknown"),
        }
        return item
