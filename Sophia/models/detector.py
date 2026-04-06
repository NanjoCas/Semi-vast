"""
DualChannelDetector: Dual-Channel DeBERTa Fake News Detector.

Architecture:
  - Shared microsoft/deberta-v3-base encoder
  - Reasoning Channel: [CLS] claim [SEP] evidence [SEP]  (labeled data)
  - Content Channel:   [CLS] claim [SEP]                 (pseudo-labeled data)
  - Shared classification head: Linear(hidden,256) -> ReLU -> Dropout(0.1) -> Linear(256, 3)
  - Joint loss: L = L_sup + lambda * sum_i(w_i * L_pseudo_i)
  - Lambda scheduler: linear anneal from lambda_init=0.1 to lambda_final=0.3

Label mapping:
  SUPPORTS=0, REFUTES=1, NOT_ENOUGH_INFO=2
"""

import json
import pathlib
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DualChannelDetector(nn.Module):
    """
    Dual-Channel Fake News Detector with shared DeBERTa encoder.

    Both the Reasoning Channel (claim + evidence) and the Content Channel
    (claim only) pass through the same encoder and classification head.

    Args:
        model_name (str): HuggingFace model identifier.
        num_labels (int): Number of output classes (default 3).
        dropout (float): Dropout rate in the classification head.
        lambda_init (float): Initial lambda weight for pseudo loss.
        lambda_final (float): Final lambda weight after annealing.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        lambda_init: float = 0.1,
        lambda_final: float = 0.3,
    ):
        super().__init__()

        # Shared DeBERTa encoder
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size  # 768 for deberta-v3-base

        # Shared classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        # Lambda schedule parameters
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final

        self.num_labels = num_labels

    # ------------------------------------------------------------------
    # Internal forward pass
    # ------------------------------------------------------------------

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the shared DeBERTa encoder and return the [CLS] representation.

        DeBERTa-v3 variants do not use token_type_ids; this method silently
        drops them when the underlying config does not support them.
        """
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Only pass token_type_ids if the model config declares type vocab
        if (
            token_type_ids is not None
            and getattr(self.deberta.config, "type_vocab_size", 0) > 0
        ):
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.deberta(**kwargs)
        # outputs.last_hidden_state: (batch, seq_len, hidden)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_repr

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process claim+evidence pairs through the Reasoning Channel.

        Input format (handled by the caller's tokenizer):
            [CLS] claim [SEP] evidence [SEP]

        Args:
            input_ids (Tensor): Shape (batch, seq_len).
            attention_mask (Tensor): Shape (batch, seq_len).
            token_type_ids (Tensor, optional): Shape (batch, seq_len).

        Returns:
            Tensor: Logits of shape (batch, num_labels).
        """
        cls_repr = self._encode(input_ids, attention_mask, token_type_ids)
        return self.classifier(cls_repr)

    def forward_content(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process claim-only inputs through the Content Channel.

        Input format (handled by the caller's tokenizer):
            [CLS] claim [SEP]

        Args:
            input_ids (Tensor): Shape (batch, seq_len).
            attention_mask (Tensor): Shape (batch, seq_len).
            token_type_ids (Tensor, optional): Shape (batch, seq_len).

        Returns:
            Tensor: Logits of shape (batch, num_labels).
        """
        cls_repr = self._encode(input_ids, attention_mask, token_type_ids)
        return self.classifier(cls_repr)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_joint_loss(
        self,
        labeled_logits: torch.Tensor,
        labeled_labels: torch.Tensor,
        pseudo_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        pseudo_weights: torch.Tensor,
        lambda_val: float,
        class_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute the joint supervised + pseudo-labeled loss.

        L = L_sup + lambda_val * sum_i(w_i * L_pseudo_i)

        For pseudo loss, CrossEntropy is computed per-sample (reduction='none'),
        then multiplied element-wise by pseudo_weights and averaged. This
        ensures that low-confidence pseudo-labels contribute less to the gradient.

        Args:
            labeled_logits (Tensor): Shape (N_lab, num_labels) — from forward_reasoning.
            labeled_labels (Tensor): Shape (N_lab,) — integer class indices.
            pseudo_logits (Tensor): Shape (N_pse, num_labels) — from forward_content.
            pseudo_labels (Tensor): Shape (N_pse,) — integer pseudo class indices.
            pseudo_weights (Tensor): Shape (N_pse,) — per-sample confidence weights in [0,1].
            lambda_val (float): Global weighting for the pseudo loss term.
            class_weights (Tensor, optional): Shape (num_labels,) — inverse-frequency
                weights passed to the supervised CrossEntropyLoss.

        Returns:
            dict with keys:
                "total"      (Tensor): L_sup + lambda_val * L_pseudo  (scalar)
                "supervised" (Tensor): L_sup                           (scalar)
                "pseudo"     (Tensor): weighted pseudo loss            (scalar)
        """
        # Supervised loss (with optional class-level balancing)
        sup_criterion = nn.CrossEntropyLoss(weight=class_weights)
        l_sup = sup_criterion(labeled_logits, labeled_labels)

        # Pseudo loss: per-sample CE, then weighted mean
        # reduction='none' returns shape (N_pse,)
        pseudo_criterion = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = pseudo_criterion(pseudo_logits, pseudo_labels)

        # Normalise weights to sum to 1 to keep gradient scale stable
        weight_sum = pseudo_weights.sum().clamp(min=1e-8)
        l_pseudo = (pseudo_weights * per_sample_loss).sum() / weight_sum

        l_total = l_sup + lambda_val * l_pseudo

        return {
            "total": l_total,
            "supervised": l_sup,
            "pseudo": l_pseudo,
        }

    # ------------------------------------------------------------------
    # Lambda scheduler
    # ------------------------------------------------------------------

    def get_lambda(self, current_step: int, total_steps: int) -> float:
        """
        Compute linearly annealed lambda for the current training step.

        Lambda increases from lambda_init to lambda_final over total_steps,
        ensuring labeled data dominates in early training and pseudo-labeled
        data contributes more as training progresses.

        Args:
            current_step (int): Current global training step (0-indexed).
            total_steps (int): Total number of training steps.

        Returns:
            float: Lambda value for this step.
        """
        if total_steps <= 0:
            return self.lambda_final
        progress = min(current_step / total_steps, 1.0)
        return self.lambda_init + progress * (self.lambda_final - self.lambda_init)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        channel: str = "reasoning",
    ) -> tuple:
        """
        Run inference and return prediction details.

        Args:
            input_ids (Tensor): Shape (batch, seq_len).
            attention_mask (Tensor): Shape (batch, seq_len).
            token_type_ids (Tensor, optional): Shape (batch, seq_len).
            channel (str): "reasoning" uses forward_reasoning (claim+evidence);
                           "content"   uses forward_content   (claim only).

        Returns:
            tuple of four elements:
                logits          (Tensor): Shape (batch, num_labels) — raw logits.
                probs           (Tensor): Shape (batch, num_labels) — softmax probabilities.
                predicted_class (Tensor): Shape (batch,) — argmax class index.
                confidence      (Tensor): Shape (batch,) — max softmax probability.
        """
        self.eval()
        with torch.no_grad():
            if channel == "reasoning":
                logits = self.forward_reasoning(input_ids, attention_mask, token_type_ids)
            elif channel == "content":
                logits = self.forward_content(input_ids, attention_mask, token_type_ids)
            else:
                raise ValueError(f"Unknown channel: '{channel}'. Choose 'reasoning' or 'content'.")

        probs = F.softmax(logits, dim=-1)
        predicted_class = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        return logits, probs, predicted_class, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights and configuration to a directory.

        Saves:
          - pytorch_model.bin  — state dict
          - config.json        — model hyperparameters

        Args:
            path (str): Directory path (created if it does not exist).
        """
        save_dir = pathlib.Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

        config = {
            "model_name": self.deberta.config.name_or_path,
            "num_labels": self.num_labels,
            "lambda_init": self.lambda_init,
            "lambda_final": self.lambda_final,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    def load(self, path: str, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Load model weights from a directory produced by :meth:`save`.

        Args:
            path (str): Directory path containing ``pytorch_model.bin``.
            device (str or torch.device, optional): Target device. If None,
                weights are loaded onto the device they were saved from.
        """
        save_dir = pathlib.Path(path)
        weights_path = save_dir / "pytorch_model.bin"

        map_location: Optional[Union[str, torch.device]] = device
        state_dict = torch.load(weights_path, map_location=map_location)
        self.load_state_dict(state_dict)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Memory optimisation
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing on the DeBERTa encoder to reduce GPU
        memory at the cost of slightly slower backward passes.

        Recommended when running with batch_size > 8 and max_length=512.
        """
        self.deberta.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DualChannelDataset(Dataset):
    """
    PyTorch Dataset that wraps both labeled and pseudo-labeled data for
    joint training of the DualChannelDetector.

    Labeled records (from ``labeled_path``) are assigned to the
    ``"reasoning"`` channel (claim + evidence).  Pseudo-labeled records
    (from ``pseudo_path``) are assigned to the ``"content"`` channel
    (claim only).

    Each ``__getitem__`` call returns a single sample dict; a custom
    collate function is recommended to group by channel before passing to
    the model.  The recommended batch composition is controlled by
    ``labeled_ratio``, which determines the fraction of labeled samples
    returned via index mapping.

    Expected JSONL schemas
    ----------------------
    Labeled record::

        {
            "id": "cf_75",
            "claim": "...",
            "evidence": ["sentence1", "sentence2"],
            "label": "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
            "source": "climate_fever | pubhealth"
        }

    Pseudo-labeled record::

        {
            "id": "tw_a1b2c3d4",
            "claim": "...",
            "pseudo_label": 0,
            "weight": 0.73
        }

    Args:
        labeled_path (str): Path to labeled ``train.jsonl``.
        pseudo_path (str): Path to pseudo-labeled JSONL (RL-selector output).
        tokenizer: HuggingFace tokenizer (must be pre-loaded by the caller).
        max_length (int): Maximum token sequence length.
        labeled_ratio (float): Fraction of labeled samples in the combined
            index space (0 < labeled_ratio <= 1).  The virtual dataset length
            is set so that each epoch sees all labeled samples and a
            proportional number of pseudo samples.
    """

    def __init__(
        self,
        labeled_path: str,
        pseudo_path: str,
        tokenizer,
        max_length: int = 512,
        labeled_ratio: float = 0.25,
    ):
        if not 0 < labeled_ratio <= 1:
            raise ValueError("labeled_ratio must be in (0, 1].")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labeled_ratio = labeled_ratio

        # Load labeled records
        self.labeled: list[dict] = []
        with open(labeled_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.labeled.append(json.loads(line))

        # Load pseudo-labeled records
        self.pseudo: list[dict] = []
        with open(pseudo_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.pseudo.append(json.loads(line))

        # Build a flat index: first N_lab entries → labeled, remainder → pseudo.
        # Total length is set so that labeled_ratio holds exactly.
        n_lab = len(self.labeled)
        if labeled_ratio < 1.0 and self.pseudo:
            n_pseudo_per_epoch = int(n_lab * (1 - labeled_ratio) / labeled_ratio)
            n_pseudo_per_epoch = min(n_pseudo_per_epoch, len(self.pseudo))
        else:
            n_pseudo_per_epoch = len(self.pseudo)

        self._n_labeled_in_epoch = n_lab
        self._n_pseudo_in_epoch = n_pseudo_per_epoch
        self._total_len = n_lab + n_pseudo_per_epoch

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> dict:
        if idx < self._n_labeled_in_epoch:
            return self._get_labeled(idx)
        else:
            pseudo_idx = (idx - self._n_labeled_in_epoch) % len(self.pseudo)
            return self._get_pseudo(pseudo_idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_labeled(self, idx: int) -> dict:
        """Tokenize a labeled (claim + evidence) record for the Reasoning Channel."""
        record = self.labeled[idx]
        claim = record["claim"]

        # Concatenate evidence sentences with spaces; fall back to empty string
        evidence_parts = record.get("evidence", [])
        if isinstance(evidence_parts, list):
            evidence = " ".join(evidence_parts)
        else:
            evidence = str(evidence_parts)

        # Tokenize: [CLS] claim [SEP] evidence [SEP]
        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_str = record.get("label", "NOT_ENOUGH_INFO")
        label_id = LABEL2ID.get(label_str, LABEL2ID["NOT_ENOUGH_INFO"])

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
            "weight": torch.tensor(1.0, dtype=torch.float),
            "channel": "reasoning",
            "id": record.get("id", ""),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item

    def _get_pseudo(self, idx: int) -> dict:
        """Tokenize a pseudo-labeled (claim-only) record for the Content Channel."""
        record = self.pseudo[idx]
        claim = record["claim"]

        # Tokenize: [CLS] claim [SEP]
        encoding = self.tokenizer(
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        pseudo_label = int(record.get("pseudo_label", 2))  # default NOT_ENOUGH_INFO
        weight = float(record.get("weight", 1.0))

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(pseudo_label, dtype=torch.long),
            "weight": torch.tensor(weight, dtype=torch.float),
            "channel": "content",
            "id": record.get("id", ""),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


# ---------------------------------------------------------------------------
# Utility: class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(train_jsonl_path: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the labeled training file.

    Reads ``train.jsonl``, counts occurrences of each label, and returns a
    weight tensor where rare classes receive higher weight to counteract
    class imbalance during supervised training.

    Formula::

        weight_c = total_samples / (num_classes * count_c)

    If a class has zero samples, its weight is set to 1.0 to avoid division
    by zero (this would only occur with a severely truncated training set).

    Args:
        train_jsonl_path (str): Path to ``processed/labeled/train.jsonl``.

    Returns:
        torch.Tensor: Shape (3,) — weights for [SUPPORTS, REFUTES, NOT_ENOUGH_INFO].
    """
    num_classes = len(LABEL2ID)
    counts = [0] * num_classes

    with open(train_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label_str = record.get("label", "NOT_ENOUGH_INFO")
            label_id = LABEL2ID.get(label_str, LABEL2ID["NOT_ENOUGH_INFO"])
            counts[label_id] += 1

    total = sum(counts)
    if total == 0:
        raise ValueError(f"No records found in {train_jsonl_path}")

    weights = []
    for c in range(num_classes):
        if counts[c] > 0:
            weights.append(total / (num_classes * counts[c]))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float)
