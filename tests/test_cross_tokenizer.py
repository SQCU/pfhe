"""
Tests for cross-tokenizer knowledge distillation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from phfe.distillation.cross_tokenizer import (
    SparseLogits,
    TokenAlignment,
    CrossTokenizerAligner,
    LogitAggregator,
    VocabularyMapper,
    compute_gkd_loss,
)


# =============================================================================
# SparseLogits Tests
# =============================================================================


class TestSparseLogits:
    """Tests for SparseLogits dataclass."""

    def test_create(self):
        """Should create sparse logits."""
        sl = SparseLogits(
            token_ids=[1, 2, 3],
            log_probs=[-0.1, -0.5, -1.0],
            total_prob=0.95,
            position=0,
        )
        assert len(sl.token_ids) == 3
        assert sl.position == 0

    def test_to_dict(self):
        """Should serialize to dict."""
        sl = SparseLogits(
            token_ids=[1, 2],
            log_probs=[-0.1, -0.5],
            total_prob=0.9,
            position=5,
        )
        d = sl.to_dict()
        assert d["token_ids"] == [1, 2]
        assert d["position"] == 5

    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {
            "token_ids": [10, 20],
            "log_probs": [-0.2, -0.8],
            "total_prob": 0.85,
            "position": 3,
        }
        sl = SparseLogits.from_dict(d)
        assert sl.token_ids == [10, 20]
        assert sl.position == 3

    def test_top_k(self):
        """Should return top-k subset."""
        sl = SparseLogits(
            token_ids=[1, 2, 3, 4, 5],
            log_probs=[-0.1, -0.5, -0.2, -1.0, -0.3],  # 1, 3, 5, 2, 4 sorted
            total_prob=0.99,
            position=0,
        )
        top3 = sl.top_k(3)
        assert len(top3.token_ids) == 3
        # Should have tokens with highest log probs
        assert 1 in top3.token_ids  # -0.1
        assert 3 in top3.token_ids  # -0.2
        assert 5 in top3.token_ids  # -0.3

    def test_top_k_when_fewer(self):
        """Should return all when k > len."""
        sl = SparseLogits(
            token_ids=[1, 2],
            log_probs=[-0.1, -0.5],
            total_prob=0.9,
            position=0,
        )
        top10 = sl.top_k(10)
        assert len(top10.token_ids) == 2


# =============================================================================
# Mock Tokenizers for Testing
# =============================================================================


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int], char_level: bool = False):
        """
        Args:
            vocab: Text -> token ID mapping
            char_level: If True, tokenize by character
        """
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.char_level = char_level
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.char_level:
            return [self.vocab.get(c, 0) for c in text]

        # Simple word-level tokenization
        tokens = []
        remaining = text
        while remaining:
            found = False
            # Try longest match first
            for length in range(len(remaining), 0, -1):
                prefix = remaining[:length]
                if prefix in self.vocab:
                    tokens.append(self.vocab[prefix])
                    remaining = remaining[length:]
                    found = True
                    break
            if not found:
                # Unknown char
                remaining = remaining[1:]

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.reverse_vocab.get(tid, "?") for tid in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


# =============================================================================
# CrossTokenizerAligner Tests
# =============================================================================


class TestCrossTokenizerAligner:
    """Tests for cross-tokenizer alignment."""

    @pytest.fixture
    def teacher_tokenizer(self):
        """Teacher with word-level tokens."""
        return MockTokenizer({
            "hello": 1,
            " ": 2,
            "world": 3,
            "!": 4,
        })

    @pytest.fixture
    def student_tokenizer(self):
        """Student with subword tokens."""
        return MockTokenizer({
            "hel": 10,
            "lo": 11,
            " wor": 12,
            "ld": 13,
            "!": 14,
        })

    @pytest.fixture
    def aligner(self, teacher_tokenizer, student_tokenizer):
        return CrossTokenizerAligner(teacher_tokenizer, student_tokenizer)

    def test_align_same_tokenizer(self):
        """Same tokenizer should give 1:1 alignment."""
        tokenizer = MockTokenizer({"a": 1, "b": 2, "c": 3})
        aligner = CrossTokenizerAligner(tokenizer, tokenizer)

        alignment = aligner.align("abc")

        # Each position should map to itself
        assert 0 in alignment.teacher_to_student
        assert 1 in alignment.teacher_to_student
        assert 2 in alignment.teacher_to_student

    def test_align_different_tokenizers(self, aligner):
        """Different tokenizers should align via character overlap."""
        text = "hello world!"
        alignment = aligner.align(text)

        # Teacher "hello" (pos 0) should overlap with student "hel" and "lo"
        assert 0 in alignment.teacher_to_student
        student_spans = alignment.teacher_to_student[0]
        assert len(student_spans) >= 1

    def test_alignment_is_symmetric(self, aligner):
        """Student-to-teacher should be inverse of teacher-to-student."""
        text = "hello!"
        alignment = aligner.align(text)

        # If teacher[0] -> student[0,1], then student[0] and [1] -> teacher[0]
        for t_pos, s_spans in alignment.teacher_to_student.items():
            for s_start, s_end in s_spans:
                for s_pos in range(s_start, s_end):
                    assert s_pos in alignment.student_to_teacher

    def test_char_spans_computed(self, aligner):
        """Should compute character spans."""
        text = "hello"
        alignment = aligner.align(text)

        assert len(alignment.teacher_char_spans) > 0
        assert len(alignment.student_char_spans) > 0


# =============================================================================
# LogitAggregator Tests
# =============================================================================


class TestLogitAggregator:
    """Tests for logit aggregation strategies."""

    @pytest.fixture
    def sample_logits(self):
        """Sample sparse logits for testing."""
        return [
            SparseLogits(
                token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.5, -1.0],
                total_prob=0.95,
                position=0,
            ),
            SparseLogits(
                token_ids=[2, 3, 4],
                log_probs=[-0.2, -0.6, -1.2],
                total_prob=0.93,
                position=1,
            ),
            SparseLogits(
                token_ids=[1, 4, 5],
                log_probs=[-0.3, -0.7, -1.5],
                total_prob=0.90,
                position=2,
            ),
        ]

    @pytest.fixture
    def sample_alignment(self):
        """Sample alignment where each student maps to one teacher."""
        return TokenAlignment(
            teacher_to_student={0: [(0, 1)], 1: [(1, 2)], 2: [(2, 3)]},
            student_to_teacher={0: [(0, 1)], 1: [(1, 2)], 2: [(2, 3)]},
        )

    def test_first_strategy(self, sample_logits, sample_alignment):
        """First strategy should use first overlapping logits."""
        agg = LogitAggregator(strategy="first")
        targets = agg.aggregate(sample_logits, sample_alignment, num_student_tokens=3)

        assert 0 in targets
        assert 1 in targets
        # Position 0 should have token 1's logits from teacher pos 0
        assert 1 in targets[0]
        assert targets[0][1] == -0.1

    def test_average_strategy(self, sample_logits):
        """Average strategy should average log-probs."""
        # Alignment where student pos 0 maps to teacher pos 0 and 1
        alignment = TokenAlignment(
            teacher_to_student={0: [(0, 1)], 1: [(0, 1)]},
            student_to_teacher={0: [(0, 2)]},  # Maps to both 0 and 1
        )

        agg = LogitAggregator(strategy="average")
        targets = agg.aggregate(sample_logits, alignment, num_student_tokens=1)

        assert 0 in targets
        # Token 2 appears in both - should have averaged log prob
        assert 2 in targets[0]
        expected = np.mean([-0.5, -0.2])  # From pos 0 and 1
        assert abs(targets[0][2] - expected) < 0.01

    def test_max_strategy(self, sample_logits):
        """Max strategy should take max log-prob."""
        alignment = TokenAlignment(
            teacher_to_student={0: [(0, 1)], 1: [(0, 1)]},
            student_to_teacher={0: [(0, 2)]},
        )

        agg = LogitAggregator(strategy="max")
        targets = agg.aggregate(sample_logits, alignment, num_student_tokens=1)

        assert 0 in targets
        # Token 2: max(-0.5, -0.2) = -0.2
        assert 2 in targets[0]
        assert targets[0][2] == -0.2

    def test_no_alignment(self, sample_logits):
        """Positions without alignment should be skipped."""
        alignment = TokenAlignment(
            teacher_to_student={},
            student_to_teacher={},  # No alignment
        )

        agg = LogitAggregator(strategy="first")
        targets = agg.aggregate(sample_logits, alignment, num_student_tokens=3)

        assert len(targets) == 0


# =============================================================================
# VocabularyMapper Tests
# =============================================================================


class TestVocabularyMapper:
    """Tests for vocabulary mapping between tokenizers."""

    def test_exact_match(self):
        """Should map tokens with identical text."""
        teacher = MockTokenizer({"hello": 1, "world": 2})
        student = MockTokenizer({"hello": 10, "world": 20, "extra": 30})

        mapper = VocabularyMapper(teacher, student)
        mapping = mapper.build_mapping()

        assert 1 in mapping
        assert 10 in mapping[1]
        assert 2 in mapping
        assert 20 in mapping[2]

    def test_case_insensitive(self):
        """Should match case-insensitively."""
        teacher = MockTokenizer({"Hello": 1})
        student = MockTokenizer({"hello": 10})

        mapper = VocabularyMapper(teacher, student)
        mapping = mapper.build_mapping()

        assert 1 in mapping
        assert 10 in mapping[1]

    def test_no_match(self):
        """Unmatched tokens should have empty mapping."""
        teacher = MockTokenizer({"abc": 1})
        student = MockTokenizer({"xyz": 10})

        mapper = VocabularyMapper(teacher, student)
        mapping = mapper.build_mapping()

        # Token 1 has no match
        assert 1 not in mapping or len(mapping[1]) == 0


# =============================================================================
# GKD Loss Tests
# =============================================================================


class TestGKDLoss:
    """Tests for GKD loss computation."""

    def test_loss_perfect_match(self):
        """Loss should be low when student matches teacher."""
        import torch

        # Student logits that match teacher distribution
        # Teacher says token 0 has prob 0.9, token 1 has prob 0.1
        student_logits = torch.tensor([
            [2.0, 0.0, -10.0, -10.0],  # softmax ≈ [0.88, 0.12, 0, 0]
        ])

        teacher_targets = {
            0: {0: np.log(0.9), 1: np.log(0.1)},
        }

        loss = compute_gkd_loss(student_logits, teacher_targets)
        assert loss < 0.5  # Should be low

    def test_loss_mismatch(self):
        """Loss should be high when student differs from teacher."""
        import torch

        # Student predicts token 1, teacher predicts token 0
        student_logits = torch.tensor([
            [-10.0, 2.0, -10.0, -10.0],  # softmax ≈ [0, 0.99, 0, 0]
        ])

        teacher_targets = {
            0: {0: np.log(0.99)},  # Teacher strongly prefers 0
        }

        loss = compute_gkd_loss(student_logits, teacher_targets)
        assert loss > 1.0  # Should be high

    def test_loss_multiple_positions(self):
        """Should compute loss over multiple positions."""
        import torch

        student_logits = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        teacher_targets = {
            0: {0: np.log(0.8)},
            1: {1: np.log(0.8)},
            2: {2: np.log(0.8)},
        }

        loss = compute_gkd_loss(student_logits, teacher_targets, reduction="mean")
        assert loss >= 0

    def test_loss_skips_missing_positions(self):
        """Should handle positions without targets."""
        import torch

        student_logits = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])

        teacher_targets = {
            0: {0: np.log(0.9)},
            # Position 1 missing
            2: {0: np.log(0.9)},
        }

        # Should not crash
        loss = compute_gkd_loss(student_logits, teacher_targets)
        assert loss >= 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end tests for cross-tokenizer distillation."""

    def test_full_pipeline(self):
        """Test complete alignment -> aggregation -> loss pipeline."""
        import torch

        # Setup tokenizers
        teacher_tok = MockTokenizer({
            "the": 1, " ": 2, "cat": 3, "sat": 4,
        })
        student_tok = MockTokenizer({
            "th": 10, "e": 11, " ca": 12, "t": 13, " sa": 14,
        })

        # Align
        aligner = CrossTokenizerAligner(teacher_tok, student_tok)
        text = "the cat"
        alignment = aligner.align(text)

        # Create teacher logits
        teacher_logits = [
            SparseLogits([1, 2], [-0.1, -2.0], 0.95, 0),  # "the"
            SparseLogits([2, 3], [-0.2, -1.5], 0.93, 1),  # " "
            SparseLogits([3, 4], [-0.3, -1.0], 0.90, 2),  # "cat"
        ]

        # Aggregate
        aggregator = LogitAggregator(strategy="first")
        targets = aggregator.aggregate(
            teacher_logits, alignment, num_student_tokens=5
        )

        # Compute loss with random student logits
        student_logits = torch.randn(5, 20)
        loss = compute_gkd_loss(student_logits, targets)

        assert loss >= 0
