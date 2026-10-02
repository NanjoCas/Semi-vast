"""
DiscourseScorer: Extracts discourse-level features from claim text.

Computes three sub-scores:
  - negation_score:  density of negation cues in the claim
  - modality_score:  density of epistemic modality markers in the claim
  - sentiment_score: absolute value of VADER compound sentiment

A combined discourse_score is the weighted average:
  0.4 * negation + 0.3 * modality + 0.3 * sentiment
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Multi-word phrases must come before single tokens so they are matched first.
_NEGATION_CUES: list[str] = [
    "no evidence",
    "contrary to",
    "not",
    "never",
    "no",
    "neither",
    "nor",
    "false",
    "incorrect",
    "misleading",
    "debunked",
]

_MODALITY_MARKERS: list[str] = [
    "may",
    "might",
    "could",
    "allegedly",
    "claimed",
    "supposedly",
    "reportedly",
    "perhaps",
    "possibly",
    "likely",
    "uncertain",
    "unclear",
]

# Weights for discourse_score
_W_NEGATION = 0.4
_W_MODALITY = 0.3
_W_SENTIMENT = 0.3


def _count_phrase_matches(text_lower: str, phrases: list[str]) -> int:
    """
    Count non-overlapping occurrences of lexicon phrases in lowercased text.

    Multi-word phrases are matched as contiguous substrings; single-word
    entries are matched as whole words to avoid partial matches.
    """
    count = 0
    for phrase in phrases:
        if " " in phrase:
            # Multi-word: simple substring count
            count += text_lower.count(phrase)
        else:
            # Single word: whole-word boundary match
            count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
    return count


class DiscourseScorer:
    """
    Extracts discourse features from claim text for fake news detection.

    All scores are normalised to [0, 1]:
      - negation_score  = negation cue count / word count
      - modality_score  = modality marker count / word count
      - sentiment_score = |VADER compound score|  (already in [0, 1])

    discourse_score = 0.4 * negation + 0.3 * modality + 0.3 * sentiment
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, claim: str) -> dict:
        """
        Compute discourse features for a single claim.

        Args:
            claim (str): The claim text to analyse.

        Returns:
            dict with keys:
                "negation"        (float): negation cue density, clipped to [0, 1]
                "modality"        (float): modality marker density, clipped to [0, 1]
                "sentiment"       (float): |VADER compound|, in [0, 1]
                "discourse_score" (float): weighted average, in [0, 1]
        """
        if not claim or not claim.strip():
            return {
                "negation": 0.0,
                "modality": 0.0,
                "sentiment": 0.0,
                "discourse_score": 0.0,
            }

        text_lower = claim.lower()
        words = claim.split()
        word_count = max(len(words), 1)  # guard against empty-after-split edge case

        negation = self._negation_score(text_lower, word_count)
        modality = self._modality_score(text_lower, word_count)
        sentiment = self._sentiment_score(claim)

        discourse = _W_NEGATION * negation + _W_MODALITY * modality + _W_SENTIMENT * sentiment

        return {
            "negation": negation,
            "modality": modality,
            "sentiment": sentiment,
            "discourse_score": discourse,
        }

    def score_batch(self, claims: list[str]) -> list[dict]:
        """
        Compute discourse features for a list of claims.

        Args:
            claims (list[str]): Claim texts to analyse.

        Returns:
            list[dict]: One result dict per claim (same format as `score`).
        """
        return [self.score(claim) for claim in claims]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _negation_score(self, text_lower: str, word_count: int) -> float:
        """Normalised negation cue count, clipped to [0, 1]."""
        raw = _count_phrase_matches(text_lower, _NEGATION_CUES)
        return min(raw / word_count, 1.0)

    def _modality_score(self, text_lower: str, word_count: int) -> float:
        """Normalised epistemic modality marker count, clipped to [0, 1]."""
        raw = _count_phrase_matches(text_lower, _MODALITY_MARKERS)
        return min(raw / word_count, 1.0)

    def _sentiment_score(self, claim: str) -> float:
        """Absolute VADER compound score, in [0, 1]."""
        compound = self._vader.polarity_scores(claim)["compound"]
        return abs(compound)
