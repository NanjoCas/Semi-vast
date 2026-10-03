"""
DualChannelDetector: Dual-Channel DeBERTa Fake News Detector.

Architecture:
  - Shared microsoft/deberta-v3-base encoder
  - Reasoning Channel: [CLS] claim [SEP] evidence [SEP]  (labeled data)
  - Content Channel:   [CLS] claim [SEP]                 (pseudo-labeled data)
  - Shared classification head: Linear(hidden,256) -> ReLU -> Dropout(0.1) -> Linear(256, 3)
  - Joint loss: L = L_sup + lambda * mean_i(w_i * L_pseudo_i)
  - Lambda scheduler: linear anneal from lambda_init to lambda_final (config)

v2: pseudo-labeled samples are claim+evidence pairs and are trained through
forward_reasoning (see training/train_detector.py); forward_content is kept for
claim-only experiments. compute_supervised_loss is shared by all methods.

v3 (README_v3 section 3):
  - the encoder can be loaded from the NLI cross-encoder (encoder_name) and its
    first `freeze_layers` layers frozen, exactly like the extractor, so the
    supervised-only baseline A starts from the same point as the extractor;
  - _adjust_logits now implements the training-time logit adjustment of
    Menon et al. (2021), logits + tau * log(prior); v1/v2 subtracted the prior
    during training, which pushes predictions *towards* frequent classes.
    The v3 config sets tau = 0 (class-weighted CE only);
  - the pseudo-label loss no longer applies any logit adjustment (the balanced
    pseudo sampler already handles class imbalance).

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
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        encoder_name: Optional[str] = None,
        freeze_layers: int = 0,
    ):
        super().__init__()

        cache_kwargs: dict = {}
        if cache_dir is not None:
            cache_kwargs["cache_dir"] = str(cache_dir)
        if local_files_only:
            cache_kwargs["local_files_only"] = True

        # Shared DeBERTa encoder - force float32.
        # v3: encoder_name (e.g. the NLI cross-encoder) replaces model_name as the weight source.
        # No silent fallback: a failed load must not quietly change the protocol.
        self.encoder_source = encoder_name or model_name
        self.deberta = AutoModel.from_pretrained(self.encoder_source, dtype=torch.float32, **cache_kwargs)
        self.frozen_layers = 0
        if freeze_layers > 0 and hasattr(self.deberta, "encoder"):
            self.frozen_layers = min(int(freeze_layers), len(self.deberta.encoder.layer))
            for layer in self.deberta.encoder.layer[: self.frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden_size = self.deberta.config.hidden_size  # 768 for deberta-v3-base

        # Shared classification head - ensure float32
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        ).float()  # Force float32

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

    @staticmethod
    def _adjust_logits(
        logits: torch.Tensor,
        class_priors: Optional[torch.Tensor],
        logit_adjust_tau: float,
    ) -> torch.Tensor:
        """
        Training-time logit adjustment ``logits + tau * log(prior)`` (Menon et al., 2021);
        no-op when tau <= 0. The loss is computed on the adjusted logits and predictions use
        the raw logits, which favours rare classes at test time. (v1/v2 used the post-hoc
        sign ``- tau * log(prior)`` during training, which does the opposite.)
        """
        if class_priors is None or float(logit_adjust_tau) <= 0.0:
            return logits
        priors = torch.nan_to_num(class_priors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        priors = priors / priors.sum().clamp(min=1e-8)
        prior_log = torch.log(priors.clamp(min=1e-8)).to(logits.device)
        return logits + float(logit_adjust_tau) * prior_log.unsqueeze(0)

    @staticmethod
    def _per_sample_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_type: str,
        focal_gamma: float,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
        ce = torch.nan_to_num(ce, nan=0.0, posinf=100.0, neginf=100.0)
        if loss_type.lower() == "focal":
            pt = torch.exp(-ce)
            return (1.0 - pt) ** float(focal_gamma) * ce
        return ce

    def compute_supervised_loss(
        self,
        labeled_logits: torch.Tensor,
        labeled_labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        logit_adjust_tau: float = 0.0,
        class_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Supervised loss shared by every ablation method.

        v1 used this focal + logit-adjust loss only when pseudo labels were on
        and plain CE for the supervised-only baseline, so A vs. B/C mixed the
        effect of pseudo labels with the effect of the loss function.
        """
        logits = torch.nan_to_num(labeled_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = self._adjust_logits(logits, class_priors, logit_adjust_tau)
        if loss_type.lower() == "focal":
            return self._per_sample_loss(logits, labeled_labels, "focal", focal_gamma, class_weights).mean()
        # Match nn.CrossEntropyLoss(weight=...) reduction (weighted mean).
        return F.cross_entropy(logits, labeled_labels, weight=class_weights)

    def compute_joint_loss(
        self,
        labeled_logits: torch.Tensor,
        labeled_labels: torch.Tensor,
        pseudo_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        pseudo_weights: torch.Tensor,
        lambda_val: float,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        logit_adjust_tau: float = 0.0,
        class_priors: Optional[torch.Tensor] = None,
        pseudo_loss_type: str = "ce",
        pseudo_weight_norm: str = "mean",
    ) -> dict:
        """
        L = L_sup + lambda * L_pseudo

        v2 additions:
          pseudo_loss_type   : "ce" (default) or "focal". Focal on confident
                               pseudo labels drives their loss to ~0.
          pseudo_weight_norm : "mean" -> mean(w_i * L_i), so the absolute
                               weights matter; "sum" -> sum(w_i * L_i) / sum(w_i)
                               (v1 behaviour, only relative weights matter).
        """
        l_sup = self.compute_supervised_loss(
            labeled_logits,
            labeled_labels,
            class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            logit_adjust_tau=logit_adjust_tau,
            class_priors=class_priors,
        )

        # v3: no logit adjustment on pseudo labels (the balanced pseudo sampler handles imbalance).
        pseudo_logits = torch.nan_to_num(pseudo_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        pseudo_weights = torch.nan_to_num(
            pseudo_weights.float(), nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        per_sample = self._per_sample_loss(pseudo_logits, pseudo_labels, pseudo_loss_type, focal_gamma)

        if pseudo_weight_norm == "sum":
            l_pseudo = (pseudo_weights * per_sample).sum() / pseudo_weights.sum().clamp(min=1e-8)
        else:
            l_pseudo = (pseudo_weights * per_sample).mean()

        l_total = l_sup + lambda_val * l_pseudo
        l_total = torch.nan_to_num(l_total, nan=0.0, posinf=100.0, neginf=100.0)

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
# Utility: class priors (class weights: common.data_utils.class_weights_from_labels)
# ---------------------------------------------------------------------------


def compute_class_priors(train_jsonl_path: str, smoothing: float = 1e-3) -> torch.Tensor:
    """
    Compute class prior probabilities from labeled train data.

    Returns a probability tensor of shape (3,) ordered as
    [SUPPORTS, REFUTES, NOT_ENOUGH_INFO].
    """
    num_classes = len(LABEL2ID)
    counts = [0.0] * num_classes

    with open(train_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label_str = record.get("label", "NOT_ENOUGH_INFO")
            label_id = LABEL2ID.get(label_str, LABEL2ID["NOT_ENOUGH_INFO"])
            counts[label_id] += 1.0

    total = sum(counts)
    if total <= 0:
        raise ValueError(f"No records found in {train_jsonl_path}")

    counts_t = torch.tensor(counts, dtype=torch.float)
    priors = (counts_t + float(smoothing))
    priors = priors / priors.sum().clamp(min=1e-8)
    return priors
