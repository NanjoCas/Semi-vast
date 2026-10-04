"""
LogicScorer: Computes a logical consistency score between a claim and evidence
using a cross-encoder NLI model.

LogicScore = p_entailment - p_contradiction, in range [-1, 1].
A higher score indicates the evidence more strongly entails the claim.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# The NLI label order is read from the model config (see LogicScorer.__init__).
# v1/v2 hard-coded entailment=2, but cross-encoder/nli-deberta-v3-large uses
# {0: contradiction, 1: entailment, 2: neutral}, so the old score was
# p_neutral - p_contradiction.


class LogicScorer:
    """
    Computes LogicScore = p_entail - p_contradict for claim-evidence pairs.

    Uses the cross-encoder/nli-deberta-v3-large model from HuggingFace to
    perform natural language inference and derive a scalar consistency score.

    Args:
        model_name (str): HuggingFace model identifier.
            Defaults to "cross-encoder/nli-deberta-v3-large".
        device (str or torch.device, optional): The device to run inference on
            ("cpu", "cuda", "mps", etc.). If None, auto-detected.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device=None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        cache_kwargs = {}
        if cache_dir is not None:
            cache_kwargs["cache_dir"] = str(cache_dir)
        if local_files_only:
            cache_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **cache_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        label2id = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in label2id or "contradiction" not in label2id:
            raise ValueError(f"{model_name} has no entailment/contradiction labels: {self.model.config.id2label}")
        self.idx_entailment = label2id["entailment"]
        self.idx_contradiction = label2id["contradiction"]
        self.idx_neutral = label2id.get("neutral")

    def score(self, claim: str, evidence: str) -> float:
        """
        Compute the LogicScore for a single claim-evidence pair.

        The model is run with the evidence as the premise and the claim as the
        hypothesis, following the standard NLI convention for fact-checking.

        Args:
            claim (str): The claim to evaluate.
            evidence (str): The evidence passage to evaluate the claim against.

        Returns:
            float: LogicScore in [-1, 1]. Positive values indicate entailment,
                negative values indicate contradiction.
        """
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        logic_score = (probs[self.idx_entailment] - probs[self.idx_contradiction]).item()
        return logic_score

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[float]:
        """
        Compute LogicScores for a list of (claim, evidence) pairs in batches.

        Args:
            pairs (list[tuple[str, str]]): A list of (claim, evidence) tuples.
            batch_size (int): Number of pairs to process per forward pass.
                Defaults to 32.

        Returns:
            list[float]: A list of LogicScores, one per input pair, each in [-1, 1].
        """
        probs = self.probs_batch(pairs, batch_size=batch_size, desc="Scoring batches")
        return [p[0] - p[1] for p in probs]

    def probs_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        desc: str = "NLI batches",
    ) -> list[list[float]]:
        """
        NLI probabilities for (claim, evidence) pairs, in pseudo-label order
        [P(entailment), P(contradiction), P(neutral)] = [SUPPORTS, REFUTES, NOT_ENOUGH_INFO].

        Same batching and premise/hypothesis order as LogicScore, so
        LogicScore == probs[0] - probs[1] for every pair (v3 method F fuses these
        probabilities with the extractor's, README_v3 10.6).
        """
        if self.idx_neutral is None:
            raise ValueError("NLI model has no neutral label")
        cols = [self.idx_entailment, self.idx_contradiction, self.idx_neutral]
        out: list[list[float]] = []

        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        for batch in tqdm(batches, desc=desc, unit="batch"):
            evidences = [evidence for _, evidence in batch]
            claims = [claim for claim, _ in batch]

            inputs = self.tokenizer(
                evidences,
                claims,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = F.softmax(logits, dim=-1)[:, cols]
            out.extend(probs.tolist())

        return out
