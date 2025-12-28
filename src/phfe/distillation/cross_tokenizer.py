"""
Cross-Tokenizer Knowledge Distillation

The core challenge: teacher and student have different tokenizers.
A teacher token might map to multiple student tokens (or vice versa).

Key insight: We preserve autoregressive token sequences, not just text.
BPE retokenization of detokenized text produces different token boundaries.

Approaches implemented:
1. Sequence-level alignment: Align teacher tokens to student token spans
2. Sparse logit targets: Store top-p logits (not full vocab)
3. Boundary-aware loss: Handle token boundary mismatches

References:
- MultiLevelOT: Optimal transport for token distribution alignment
- CDM: Common token vocabulary mapping
- GOLD: Generalized knowledge distillation with projection
- ULD: Universal logit distillation (vocabulary-agnostic)
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SparseLogits:
    """
    Sparse representation of logits (top-p 99% probability mass).

    Instead of storing full vocab logits (~100K floats per token),
    we store only the tokens that cover 99% of probability mass.
    Typically this is 10-100 tokens per position.
    """

    # Token IDs in the teacher's vocabulary
    token_ids: list[int]

    # Log probabilities for each token
    log_probs: list[float]

    # Cumulative probability (should sum to ~0.99)
    total_prob: float

    # Position in sequence (0-indexed)
    position: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_ids": self.token_ids,
            "log_probs": self.log_probs,
            "total_prob": self.total_prob,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SparseLogits":
        """Create from dictionary."""
        return cls(**d)

    def top_k(self, k: int) -> "SparseLogits":
        """Return top-k subset."""
        if k >= len(self.token_ids):
            return self
        indices = np.argsort(self.log_probs)[-k:][::-1]
        return SparseLogits(
            token_ids=[self.token_ids[i] for i in indices],
            log_probs=[self.log_probs[i] for i in indices],
            total_prob=sum(np.exp(self.log_probs[i]) for i in indices),
            position=self.position,
        )


@dataclass
class TokenAlignment:
    """
    Alignment between teacher and student token sequences.

    Maps each teacher token position to student token span(s).
    """

    # Teacher token positions -> student token spans
    # teacher_pos -> [(start, end), ...]
    teacher_to_student: dict[int, list[tuple[int, int]]]

    # Student token positions -> teacher token spans
    student_to_teacher: dict[int, list[tuple[int, int]]]

    # Character offsets for debugging
    teacher_char_spans: list[tuple[int, int]] = field(default_factory=list)
    student_char_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class DistillationTarget:
    """
    A complete distillation target for training.

    Contains teacher logits aligned to student token positions.
    """

    # Original text
    text: str

    # Teacher data
    teacher_tokens: list[int]
    teacher_logits: list[SparseLogits]
    teacher_model: str

    # Student data (tokens only - student generates logits during training)
    student_tokens: list[int]

    # Alignment
    alignment: TokenAlignment

    # Aggregated targets: student_pos -> aggregated logit distribution
    # This is what the student actually trains on
    student_targets: dict[int, dict[int, float]] = field(default_factory=dict)


class CrossTokenizerAligner:
    """
    Align token sequences between different tokenizers.

    The key challenge: "hello world" might tokenize as:
    - Teacher: ["hello", " world"]  (2 tokens)
    - Student: ["hel", "lo", " wor", "ld"]  (4 tokens)

    We align via character offsets.
    """

    def __init__(
        self,
        teacher_tokenizer,
        student_tokenizer,
    ):
        """
        Initialize aligner with tokenizers.

        Args:
            teacher_tokenizer: Teacher model's tokenizer (has encode/decode)
            student_tokenizer: Student model's tokenizer
        """
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

    def align(
        self,
        text: str,
        teacher_tokens: Optional[list[int]] = None,
        student_tokens: Optional[list[int]] = None,
    ) -> TokenAlignment:
        """
        Align teacher and student tokenizations of the same text.

        Args:
            text: The text being tokenized
            teacher_tokens: Pre-computed teacher tokens (optional)
            student_tokens: Pre-computed student tokens (optional)

        Returns:
            TokenAlignment mapping between token positions
        """
        # Tokenize if not provided
        if teacher_tokens is None:
            teacher_tokens = self.teacher_tokenizer.encode(text)
        if student_tokens is None:
            student_tokens = self.student_tokenizer.encode(text)

        # Get character spans for each token
        teacher_spans = self._get_token_char_spans(
            text, teacher_tokens, self.teacher_tokenizer
        )
        student_spans = self._get_token_char_spans(
            text, student_tokens, self.student_tokenizer
        )

        # Build alignment via span overlap
        teacher_to_student = defaultdict(list)
        student_to_teacher = defaultdict(list)

        for t_idx, t_span in enumerate(teacher_spans):
            for s_idx, s_span in enumerate(student_spans):
                if self._spans_overlap(t_span, s_span):
                    teacher_to_student[t_idx].append((s_idx, s_idx + 1))
                    student_to_teacher[s_idx].append((t_idx, t_idx + 1))

        return TokenAlignment(
            teacher_to_student=dict(teacher_to_student),
            student_to_teacher=dict(student_to_teacher),
            teacher_char_spans=teacher_spans,
            student_char_spans=student_spans,
        )

    def _get_token_char_spans(
        self,
        text: str,
        tokens: list[int],
        tokenizer,
    ) -> list[tuple[int, int]]:
        """
        Get character spans for each token.

        This is tokenizer-dependent. We try multiple strategies.
        """
        spans = []
        current_pos = 0

        for token_id in tokens:
            # Decode single token
            token_text = tokenizer.decode([token_id])

            # Find in remaining text
            # Note: This is approximate - some tokenizers add/remove spaces
            start = text.find(token_text, current_pos)
            if start == -1:
                # Fallback: try stripping
                stripped = token_text.strip()
                start = text.find(stripped, current_pos)
                if start == -1:
                    # Can't find - use current position
                    start = current_pos

            end = start + len(token_text)
            spans.append((start, end))
            current_pos = end

        return spans

    def _spans_overlap(
        self,
        span1: tuple[int, int],
        span2: tuple[int, int],
    ) -> bool:
        """Check if two character spans overlap."""
        return span1[0] < span2[1] and span2[0] < span1[1]


class LogitAggregator:
    """
    Aggregate teacher logits into student token targets.

    When a student token spans multiple teacher tokens, we need
    to combine the teacher's predictions.

    Strategies:
    1. First: Use logits from first overlapping teacher token
    2. Average: Average log-probs (geometric mean of probs)
    3. Product: Product of probs (AND semantics)
    4. Max: Max log-prob per vocab item (OR semantics)
    """

    def __init__(
        self,
        strategy: str = "first",
        teacher_to_student_vocab: Optional[dict[int, list[int]]] = None,
    ):
        """
        Initialize aggregator.

        Args:
            strategy: How to combine multiple teacher positions
            teacher_to_student_vocab: Map teacher token IDs to student IDs
                                     (for vocabulary alignment)
        """
        self.strategy = strategy
        self.vocab_map = teacher_to_student_vocab or {}

    def aggregate(
        self,
        teacher_logits: list[SparseLogits],
        alignment: TokenAlignment,
        num_student_tokens: int,
    ) -> dict[int, dict[int, float]]:
        """
        Aggregate teacher logits into student targets.

        Args:
            teacher_logits: Sparse logits for each teacher position
            alignment: Token alignment
            num_student_tokens: Number of student tokens

        Returns:
            Dict mapping student position -> (token_id -> log_prob)
        """
        student_targets = {}

        for s_pos in range(num_student_tokens):
            # Get overlapping teacher positions
            teacher_spans = alignment.student_to_teacher.get(s_pos, [])
            teacher_positions = [
                t_pos for start, end in teacher_spans for t_pos in range(start, end)
            ]

            if not teacher_positions:
                # No alignment - skip or use uniform
                continue

            # Get logits from overlapping teachers
            overlapping_logits = [
                teacher_logits[t_pos]
                for t_pos in teacher_positions
                if t_pos < len(teacher_logits)
            ]

            if not overlapping_logits:
                continue

            # Aggregate based on strategy
            if self.strategy == "first":
                target = self._aggregate_first(overlapping_logits)
            elif self.strategy == "average":
                target = self._aggregate_average(overlapping_logits)
            elif self.strategy == "max":
                target = self._aggregate_max(overlapping_logits)
            else:
                target = self._aggregate_first(overlapping_logits)

            # Map to student vocabulary if needed
            if self.vocab_map:
                target = self._map_vocabulary(target)

            student_targets[s_pos] = target

        return student_targets

    def _aggregate_first(
        self, logits_list: list[SparseLogits]
    ) -> dict[int, float]:
        """Use first overlapping position."""
        first = logits_list[0]
        return dict(zip(first.token_ids, first.log_probs))

    def _aggregate_average(
        self, logits_list: list[SparseLogits]
    ) -> dict[int, float]:
        """Average log-probs across positions."""
        combined: dict[int, list[float]] = defaultdict(list)
        for logits in logits_list:
            for tid, lp in zip(logits.token_ids, logits.log_probs):
                combined[tid].append(lp)

        return {tid: np.mean(lps) for tid, lps in combined.items()}

    def _aggregate_max(
        self, logits_list: list[SparseLogits]
    ) -> dict[int, float]:
        """Max log-prob across positions."""
        combined: dict[int, float] = {}
        for logits in logits_list:
            for tid, lp in zip(logits.token_ids, logits.log_probs):
                if tid not in combined or lp > combined[tid]:
                    combined[tid] = lp
        return combined

    def _map_vocabulary(
        self, target: dict[int, float]
    ) -> dict[int, float]:
        """Map teacher token IDs to student token IDs."""
        mapped = defaultdict(float)
        for teacher_id, log_prob in target.items():
            student_ids = self.vocab_map.get(teacher_id, [])
            for sid in student_ids:
                # Distribute probability among mapped tokens
                mapped[sid] = max(
                    mapped[sid], log_prob - np.log(len(student_ids))
                )
        return dict(mapped)


class VocabularyMapper:
    """
    Map between teacher and student vocabularies.

    Strategies:
    1. Exact match: Map tokens with identical string representation
    2. Substring: Map if one is substring of other
    3. Learned: Train a mapping (future work)
    """

    def __init__(
        self,
        teacher_tokenizer,
        student_tokenizer,
    ):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self._teacher_to_student: Optional[dict[int, list[int]]] = None
        self._student_to_teacher: Optional[dict[int, list[int]]] = None

    def build_mapping(self) -> dict[int, list[int]]:
        """
        Build teacher -> student vocabulary mapping.

        Returns:
            Dict mapping teacher token ID to list of student token IDs
        """
        if self._teacher_to_student is not None:
            return self._teacher_to_student

        mapping: dict[int, list[int]] = defaultdict(list)

        # Get vocabularies
        teacher_vocab = self._get_vocab(self.teacher_tokenizer)
        student_vocab = self._get_vocab(self.student_tokenizer)

        # Build reverse index: text -> student IDs
        student_text_to_id: dict[str, list[int]] = defaultdict(list)
        for sid, text in student_vocab.items():
            student_text_to_id[text].append(sid)
            # Also index normalized versions
            student_text_to_id[text.lower()].append(sid)
            student_text_to_id[text.strip()].append(sid)

        # Map teacher tokens
        for tid, text in teacher_vocab.items():
            # Exact match
            if text in student_text_to_id:
                mapping[tid].extend(student_text_to_id[text])
                continue

            # Try normalized
            if text.lower() in student_text_to_id:
                mapping[tid].extend(student_text_to_id[text.lower()])
                continue

            if text.strip() in student_text_to_id:
                mapping[tid].extend(student_text_to_id[text.strip()])
                continue

            # No mapping - leave empty

        # Deduplicate
        self._teacher_to_student = {
            tid: list(set(sids)) for tid, sids in mapping.items()
        }

        return self._teacher_to_student

    def _get_vocab(self, tokenizer) -> dict[int, str]:
        """Extract vocabulary from tokenizer."""
        # Try common tokenizer APIs
        if hasattr(tokenizer, "get_vocab"):
            vocab_dict = tokenizer.get_vocab()
            return {v: k for k, v in vocab_dict.items()}
        elif hasattr(tokenizer, "vocab"):
            if isinstance(tokenizer.vocab, dict):
                return {v: k for k, v in tokenizer.vocab.items()}
        elif hasattr(tokenizer, "decoder"):
            return dict(tokenizer.decoder)

        # Fallback: iterate over IDs
        vocab = {}
        for i in range(tokenizer.vocab_size):
            try:
                vocab[i] = tokenizer.decode([i])
            except Exception:
                pass
        return vocab


def compute_gkd_loss(
    student_logits,  # [seq_len, vocab_size] tensor
    teacher_targets: dict[int, dict[int, float]],
    reduction: str = "mean",
) -> float:
    """
    Compute GKD loss between student logits and teacher targets.

    This is forward KL: sum over teacher distribution of
    p_teacher * log(p_teacher / p_student)

    Args:
        student_logits: Student model's logit outputs
        teacher_targets: Sparse teacher targets per position
        reduction: "mean" or "sum"

    Returns:
        Loss value (as float for measurement, or tensor for training)
    """
    import torch

    if not isinstance(student_logits, torch.Tensor):
        student_logits = torch.tensor(student_logits)

    total_loss = 0.0
    count = 0

    for pos, target in teacher_targets.items():
        if pos >= student_logits.shape[0]:
            continue

        student_log_probs = torch.log_softmax(student_logits[pos], dim=-1)

        # Forward KL: E_teacher[log(p_teacher / p_student)]
        pos_loss = 0.0
        for token_id, teacher_log_prob in target.items():
            if token_id >= student_log_probs.shape[0]:
                continue

            teacher_prob = np.exp(teacher_log_prob)
            student_log_prob = student_log_probs[token_id].item()

            # KL term: p_teacher * (log p_teacher - log p_student)
            pos_loss += teacher_prob * (teacher_log_prob - student_log_prob)

        total_loss += pos_loss
        count += 1

    if reduction == "mean" and count > 0:
        return total_loss / count
    return total_loss


__all__ = [
    "SparseLogits",
    "TokenAlignment",
    "DistillationTarget",
    "CrossTokenizerAligner",
    "LogitAggregator",
    "VocabularyMapper",
    "compute_gkd_loss",
]
