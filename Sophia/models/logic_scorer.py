"""
LogicScorer: Computes a logical consistency score between a claim and evidence
using a cross-encoder NLI model.

LogicScore = p_entailment - p_contradiction, in range [-1, 1].
A higher score indicates the evidence more strongly entails the claim.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# NLI label order for cross-encoder/nli-deberta-v3-large
# Index 0: contradiction, Index 1: neutral, Index 2: entailment
_LABEL_CONTRADICTION = 0
_LABEL_ENTAILMENT = 2


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

    def __init__(self, model_name: str = DEFAULT_MODEL, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

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
        logic_score = (probs[_LABEL_ENTAILMENT] - probs[_LABEL_CONTRADICTION]).item()
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
        all_scores: list[float] = []

        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        for batch in tqdm(batches, desc="Scoring batches", unit="batch"):
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

            probs = F.softmax(logits, dim=-1)
            scores = (
                probs[:, _LABEL_ENTAILMENT] - probs[:, _LABEL_CONTRADICTION]
            ).tolist()
            all_scores.extend(scores)

        return all_scores
