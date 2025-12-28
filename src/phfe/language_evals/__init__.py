"""
Language Acquisition Eval Suite

Measures language competence specifically, not factual knowledge:
- Syntactic validity
- Reference tracking / entity consistency
- Discourse coherence
- Narrative structure
- Repetition / degeneration detection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvalDimension(str, Enum):
    """Evaluation dimensions."""

    SYNTACTIC = "syntactic"
    LEXICAL = "lexical"
    REFERENCE = "reference"
    DISCOURSE = "discourse"
    NARRATIVE = "narrative"
    REPETITION = "repetition"


@dataclass
class LanguageScores:
    """Scores across language competence dimensions."""

    syntactic: float = 0.0
    lexical: float = 0.0
    reference: float = 0.0
    discourse: float = 0.0
    narrative: float = 0.0
    repetition: float = 0.0

    # Weights for aggregate
    _weights: dict[str, float] = None

    def __post_init__(self):
        if self._weights is None:
            self._weights = {
                "syntactic": 1.0,
                "lexical": 0.5,
                "reference": 2.0,
                "discourse": 2.0,
                "narrative": 1.0,
                "repetition": 1.0,
            }

    @property
    def aggregate(self) -> float:
        """Weighted aggregate score."""
        total = 0.0
        weight_sum = 0.0
        for dim, weight in self._weights.items():
            total += getattr(self, dim) * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0


@dataclass
class EvalResult:
    """Result of evaluating a single text."""

    text: str
    scores: LanguageScores
    details: dict = None

    # Per-dimension breakdowns
    syntactic_details: Optional[dict] = None
    reference_details: Optional[dict] = None
    discourse_details: Optional[dict] = None


class SyntacticEvaluator:
    """Evaluate syntactic validity using constituency parsing."""

    def __init__(self, parser: str = "benepar"):
        self.parser = parser
        self._parser_model = None

    def evaluate(self, text: str) -> tuple[float, dict]:
        """
        Evaluate syntactic validity.

        Returns:
            (score, details) where score is fraction of valid parses
        """
        raise NotImplementedError("Implement syntactic evaluation")


class ReferenceEvaluator:
    """Evaluate reference tracking / entity consistency."""

    def __init__(self, coref_model: str = "coreferee"):
        self.coref_model = coref_model
        self._model = None

    def evaluate(self, text: str) -> tuple[float, dict]:
        """
        Evaluate reference consistency.

        Returns:
            (score, details) including any reference errors found
        """
        raise NotImplementedError("Implement reference evaluation")


class DiscourseEvaluator:
    """Evaluate discourse coherence."""

    def __init__(self, method: str = "embedding"):
        self.method = method  # "embedding" or "lm"
        self._embedder = None

    def evaluate(self, text: str) -> tuple[float, dict]:
        """
        Evaluate discourse coherence.

        Returns:
            (score, details) based on adjacent chunk similarity
        """
        raise NotImplementedError("Implement discourse evaluation")


class RepetitionEvaluator:
    """Detect repetitive/degenerate text."""

    def __init__(self, n: int = 4, max_repeats: int = 3):
        self.n = n  # n-gram size
        self.max_repeats = max_repeats

    def evaluate(self, text: str) -> tuple[float, dict]:
        """
        Evaluate repetition level.

        Returns:
            (score, details) where score = 1 - repetition_rate
        """
        # Simple n-gram repetition check
        words = text.split()
        if len(words) < self.n:
            return 1.0, {"ngrams_checked": 0}

        ngram_counts: dict[tuple, int] = {}
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i : i + self.n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Find max repetitions
        max_count = max(ngram_counts.values()) if ngram_counts else 0
        is_degenerate = max_count > self.max_repeats

        # Score: 1.0 if no excessive repetition, lower if degenerate
        score = 0.0 if is_degenerate else 1.0 - (max_count - 1) / (self.max_repeats + 1)
        score = max(0.0, min(1.0, score))

        return score, {
            "max_ngram_count": max_count,
            "unique_ngrams": len(ngram_counts),
            "total_ngrams": len(words) - self.n + 1,
            "is_degenerate": is_degenerate,
        }


def evaluate_generations(
    texts: list[str],
    eval_dimensions: list[EvalDimension] | str = "all",
    return_per_text: bool = False,
) -> LanguageScores | list[EvalResult]:
    """
    Evaluate a batch of generated texts.

    Args:
        texts: List of generated texts to evaluate
        eval_dimensions: Which dimensions to evaluate ("all" or list)
        return_per_text: Return per-text results instead of aggregate

    Returns:
        LanguageScores (aggregate) or list[EvalResult] (per-text)
    """
    if eval_dimensions == "all":
        eval_dimensions = list(EvalDimension)

    # Initialize evaluators
    evaluators = {}
    if EvalDimension.SYNTACTIC in eval_dimensions:
        evaluators["syntactic"] = SyntacticEvaluator()
    if EvalDimension.REFERENCE in eval_dimensions:
        evaluators["reference"] = ReferenceEvaluator()
    if EvalDimension.DISCOURSE in eval_dimensions:
        evaluators["discourse"] = DiscourseEvaluator()
    if EvalDimension.REPETITION in eval_dimensions:
        evaluators["repetition"] = RepetitionEvaluator()

    results = []
    for text in texts:
        scores = LanguageScores()

        # Run repetition eval (the one that's implemented)
        if "repetition" in evaluators:
            score, details = evaluators["repetition"].evaluate(text)
            scores.repetition = score

        results.append(EvalResult(text=text, scores=scores))

    if return_per_text:
        return results

    # Aggregate scores
    aggregate = LanguageScores()
    n = len(results)
    if n > 0:
        aggregate.syntactic = sum(r.scores.syntactic for r in results) / n
        aggregate.lexical = sum(r.scores.lexical for r in results) / n
        aggregate.reference = sum(r.scores.reference for r in results) / n
        aggregate.discourse = sum(r.scores.discourse for r in results) / n
        aggregate.narrative = sum(r.scores.narrative for r in results) / n
        aggregate.repetition = sum(r.scores.repetition for r in results) / n

    return aggregate


__all__ = [
    "EvalDimension",
    "LanguageScores",
    "EvalResult",
    "SyntacticEvaluator",
    "ReferenceEvaluator",
    "DiscourseEvaluator",
    "RepetitionEvaluator",
    "evaluate_generations",
]
