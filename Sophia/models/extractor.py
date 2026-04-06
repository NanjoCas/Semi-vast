"""
TextualFeatureExtractor
=======================
DeBERTa-v3-base fine-tuned for 3-way claim verification:
    SUPPORTS (0) | REFUTES (1) | NOT_ENOUGH_INFO (2)

Input format:
    [CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]  (max 512 tokens)
"""

import sys
import math
from pathlib import Path

# Allow imports from the project root (e.g., run_pipeline.py Dataset classes)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Collate function for UnlabeledClaimDataset
# ---------------------------------------------------------------------------

def _unlabeled_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of sample dicts from UnlabeledClaimDataset into a batch.

    Tensor fields (input_ids, attention_mask, token_type_ids) are stacked.
    String fields (id, source) are kept as plain Python lists.
    """
    collated: dict = {}
    tensor_keys = {"input_ids", "attention_mask", "token_type_ids"}

    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in tensor_keys:
            collated[key] = torch.stack(values)
        else:
            # String / scalar fields — keep as list
            collated[key] = values

    return collated


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextualFeatureExtractor(nn.Module):
    """
    DeBERTa-v3-base with a linear classification head for 3-way fact
    verification (SUPPORTS / REFUTES / NOT_ENOUGH_INFO).

    Args:
        model_name (str): HuggingFace model identifier.
            Defaults to "microsoft/deberta-v3-base".
        num_labels (int): Number of output classes. Defaults to 3.
        dropout (float): Dropout probability applied before the classifier.
            Defaults to 0.1.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        hidden_size: int = config.hidden_size

        # Keep encoder in float32 by default to avoid dtype mismatches on CPU
        # (e.g., Half from pretrained weights vs Float classifier params).
        self.deberta = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.num_labels = num_labels
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run a forward pass and return classification logits.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch, seq_len).
            token_type_ids (torch.Tensor, optional): Token type IDs.
                DeBERTa-v3 does not use them internally but the tokenizer may
                return them; they are forwarded if provided.

        Returns:
            torch.Tensor: Logits of shape (batch, num_labels).
        """
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.deberta(**kwargs)

        # Use the [CLS] token representation (position 0 of the last hidden state)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        cls_representation = cls_representation.to(self.classifier[1].weight.dtype)
        logits = self.classifier(cls_representation)
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference and return predictions with confidence metrics.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch, seq_len).
            token_type_ids (torch.Tensor, optional): Token type IDs.

        Returns:
            tuple:
                - logits (torch.Tensor): Raw logits, shape (batch, num_labels).
                - probabilities (torch.Tensor): Softmax probabilities, shape (batch, num_labels).
                - predicted_class (torch.Tensor): Argmax class indices, shape (batch,).
                - confidence (torch.Tensor): Max probability per sample, shape (batch,).
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)
        return logits, probabilities, predicted_class, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the model state dict to disk.

        Args:
            path (str): Destination file path (e.g., "checkpoints/extractor.pt").
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[TextualFeatureExtractor] Model saved to {save_path}")

    def load(self, path: str, device: str | torch.device | None = None) -> None:
        """
        Load a previously saved state dict from disk.

        Args:
            path (str): Path to the saved state dict file.
            device (str or torch.device, optional): Target device. If None,
                the state dict is loaded with map_location="cpu".
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        map_location = device if device is not None else "cpu"
        state_dict = torch.load(load_path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"[TextualFeatureExtractor] Weights loaded from {load_path}")

    # ------------------------------------------------------------------
    # Pseudo-label generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_dataset,
        logic_scorer,
        discourse_scorer,
        batch_size: int = 64,
        device: str | torch.device | None = None,
        beta1: float = 0.5,
        beta2: float = 0.3,
        beta3: float = 0.2,
        class_priors: list[float] | None = None,
        logit_adjust_tau: float = 0.0,
    ) -> list[dict]:
        """
        Generate pseudo-labels for an unlabeled dataset using DeBERTa predictions
        combined with logic and discourse scores.

        The composite weight for each sample is:

            weight = beta1 * confidence
                   + beta2 * |logic_score|
                   + beta3 * discourse_score

        Args:
            unlabeled_dataset (UnlabeledClaimDataset): Dataset instance whose
                ``.records[idx]["claim"]`` holds the raw claim text.
            logic_scorer (LogicScorer): Instance with a ``score_batch`` method.
                For unlabeled data (no evidence), the evidence string is set to "".
            discourse_scorer: Instance with a ``score_batch`` (or ``score``) method
                that accepts claim text and returns a scalar in [0, 1].
            batch_size (int): Number of samples per forward pass. Defaults to 64.
            device (str or torch.device, optional): Inference device.
                Auto-detected (CUDA > MPS > CPU) when None.
            beta1 (float): Weight for DeBERTa confidence. Defaults to 0.5.
            beta2 (float): Weight for |logic_score|. Defaults to 0.3.
            beta3 (float): Weight for discourse_score. Defaults to 0.2.
            class_priors (list[float] | None): Optional class prior probabilities
                for logit adjustment; must have length == num_labels.
            logit_adjust_tau (float): Strength of prior logit adjustment.
                0.0 disables adjustment.

        Returns:
            list[dict]: One dict per sample with keys:
                - ``id``             (str)  : Original sample identifier.
                - ``claim``          (str)  : Raw claim text.
                - ``pseudo_label``   (int)  : Predicted class (0 / 1 / 2).
                - ``confidence``     (float): max(softmax(logits)).
                - ``logic_score``    (float): LogicScore in [-1, 1].
                - ``discourse_score``(float): Discourse coherence score in [0, 1].
                - ``weight``         (float): Composite sample weight.
                - ``entropy``        (float): Prediction entropy.
                - ``probs``          (list[float]): Per-class probabilities.
        """
        # ── Device resolution ──────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        self.to(device)
        self.eval()

        # ── DataLoader ─────────────────────────────────────────────────────
        loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_unlabeled_collate_fn,
        )

        # Collect per-sample DeBERTa outputs first
        deberta_results: list[dict] = []

        prior_log = None
        if class_priors is not None and float(logit_adjust_tau) > 0.0:
            if len(class_priors) != self.num_labels:
                raise ValueError(
                    f"class_priors length mismatch: expected {self.num_labels}, got {len(class_priors)}"
                )
            priors_t = torch.tensor(class_priors, dtype=torch.float32, device=device)
            priors_t = torch.clamp(priors_t, min=1e-8)
            priors_t = priors_t / priors_t.sum()
            prior_log = torch.log(priors_t)

        for batch in tqdm(loader, desc="DeBERTa inference", unit="batch"):
            ids: list[str] = batch["id"]

            input_ids_b = batch["input_ids"].to(device)
            attention_mask_b = batch["attention_mask"].to(device)

            # token_type_ids is optional in UnlabeledClaimDataset
            token_type_ids_b: torch.Tensor | None = None
            if "token_type_ids" in batch:
                token_type_ids_b = batch["token_type_ids"].to(device)

            logits, raw_probs, _pred_classes, _confidences = self.predict(
                input_ids_b, attention_mask_b, token_type_ids_b
            )

            if prior_log is not None:
                adjusted_logits = logits - float(logit_adjust_tau) * prior_log.unsqueeze(0)
                probs = F.softmax(adjusted_logits, dim=-1)
            else:
                probs = raw_probs

            pred_classes = torch.argmax(probs, dim=-1)
            confidences, _ = torch.max(probs, dim=-1)

            # Entropy: -sum(p * log(p + eps))
            entropy_vals = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            # Move results to CPU for post-processing
            probs_cpu = probs.cpu().tolist()
            pred_classes_cpu = pred_classes.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()
            entropy_cpu = entropy_vals.cpu().tolist()

            for i, sample_id in enumerate(ids):
                deberta_results.append({
                    "id": sample_id,
                    "pseudo_label": pred_classes_cpu[i],
                    "confidence": confidences_cpu[i],
                    "entropy": entropy_cpu[i],
                    "probs": probs_cpu[i],
                })

        # ── Build claim lookup from dataset records ────────────────────────
        # unlabeled_dataset.records[idx]["claim"] holds the raw text
        id_to_claim: dict[str, str] = {
            rec["id"]: rec["claim"] for rec in unlabeled_dataset.records
        }

        # ── Logic scoring (batched, evidence = "" for unlabeled data) ──────
        all_claims = [r["id"] for r in deberta_results]
        claim_evidence_pairs: list[tuple[str, str]] = [
            (id_to_claim[sample_id], "") for sample_id in all_claims
        ]

        logic_scores: list[float] = logic_scorer.score_batch(
            claim_evidence_pairs, batch_size=batch_size
        )

        # ── Discourse scoring ──────────────────────────────────────────────
        claim_texts = [id_to_claim[r["id"]] for r in deberta_results]

        # Support both score_batch(texts) and score(text) interfaces
        if hasattr(discourse_scorer, "score_batch"):
            discourse_scores: list[float] = discourse_scorer.score_batch(claim_texts)
        else:
            discourse_scores = [
                discourse_scorer.score(t)
                for t in tqdm(claim_texts, desc="Discourse scoring", unit="sample")
            ]

        # ── Assemble final results ─────────────────────────────────────────
        results: list[dict] = []
        for i, entry in enumerate(deberta_results):
            sample_id = entry["id"]
            confidence = entry["confidence"]
            logic_score = logic_scores[i]
            discourse_score = discourse_scores[i]

            weight = (
                beta1 * confidence
                + beta2 * abs(logic_score)
                + beta3 * discourse_score
            )

            results.append({
                "id": sample_id,
                "claim": id_to_claim[sample_id],
                "pseudo_label": entry["pseudo_label"],
                "confidence": confidence,
                "logic_score": logic_score,
                "discourse_score": discourse_score,
                "weight": weight,
                "entropy": entry["entropy"],
                "probs": entry["probs"],
            })

        return results
