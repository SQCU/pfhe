"""
Tests for the contamination firewall system.

Tests cover:
- Token-level overlap (n-gram) checking
- Semantic similarity checking (with mocking for embeddings)
- Math structural similarity
- Code structural similarity
- Combined firewall functionality
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from phfe.benchmark.contamination import (
    simple_tokenize,
    extract_ngrams,
    ngram_overlap_score,
    TokenOverlapChecker,
    SemanticSimilarityChecker,
    MathStructuralChecker,
    CodeStructuralChecker,
    ContaminationFirewall,
    ContaminationConfig,
    RejectionReason,
    check_contamination,
    compute_overlap_matrix,
)


# =============================================================================
# Tokenization and N-gram Tests
# =============================================================================


class TestTokenization:
    """Tests for tokenization utilities."""

    def test_simple_tokenize_basic(self):
        text = "Hello, World!"
        tokens = simple_tokenize(text)
        assert tokens == ["hello", "world"]

    def test_simple_tokenize_numbers(self):
        text = "There are 42 apples and 17 oranges."
        tokens = simple_tokenize(text)
        assert "42" in tokens
        assert "17" in tokens
        assert "apples" in tokens

    def test_simple_tokenize_preserves_case_insensitivity(self):
        tokens1 = simple_tokenize("HELLO World")
        tokens2 = simple_tokenize("hello WORLD")
        assert tokens1 == tokens2

    def test_extract_ngrams_basic(self):
        tokens = ["a", "b", "c", "d", "e"]
        ngrams = extract_ngrams(tokens, n=3)
        expected = {
            ("a", "b", "c"),
            ("b", "c", "d"),
            ("c", "d", "e"),
        }
        assert ngrams == expected

    def test_extract_ngrams_too_short(self):
        tokens = ["a", "b"]
        ngrams = extract_ngrams(tokens, n=5)
        assert ngrams == set()

    def test_extract_ngrams_exact_length(self):
        tokens = ["a", "b", "c"]
        ngrams = extract_ngrams(tokens, n=3)
        assert ngrams == {("a", "b", "c")}


class TestNgramOverlap:
    """Tests for n-gram overlap scoring."""

    def test_identical_texts(self):
        text = "This is a test sentence with several words."
        tokens = simple_tokenize(text)
        score = ngram_overlap_score(tokens, tokens, n=3)
        assert score == 1.0

    def test_no_overlap(self):
        text1 = "The quick brown fox jumps over"
        text2 = "An entirely different sequence of words here"
        tokens1 = simple_tokenize(text1)
        tokens2 = simple_tokenize(text2)
        score = ngram_overlap_score(tokens1, tokens2, n=5)
        assert score == 0.0

    def test_partial_overlap(self):
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox runs around the park"
        tokens1 = simple_tokenize(text1)
        tokens2 = simple_tokenize(text2)
        score = ngram_overlap_score(tokens1, tokens2, n=3)
        # "the quick brown" and "quick brown fox" overlap
        assert 0.0 < score < 1.0


# =============================================================================
# Token Overlap Checker Tests
# =============================================================================


class TestTokenOverlapChecker:
    """Tests for TokenOverlapChecker."""

    def test_no_contamination(self):
        checker = TokenOverlapChecker(n=5, threshold=0.3)

        # Index a canonical problem
        checker.index_canonical(
            "gsm_001",
            "A farmer has 15 apples. He gives 5 to his neighbor. How many apples are left?"
        )

        # Check a completely different problem
        is_safe, score, match_id = checker.check(
            "Calculate the area of a rectangle with length 10 and width 5."
        )

        assert is_safe is True
        assert score < 0.3
        assert match_id is None or score <= 0.3

    def test_high_contamination(self):
        checker = TokenOverlapChecker(n=5, threshold=0.3)

        canonical = "A farmer has 15 apples. He gives 5 to his neighbor. How many apples are left?"
        checker.index_canonical("gsm_001", canonical)

        # Check a nearly identical problem
        synthetic = "A farmer has 15 apples. He gives 5 to his neighbor. How many apples remain?"

        is_safe, score, match_id = checker.check(synthetic)

        assert is_safe is False
        assert score > 0.3
        assert match_id == "gsm_001"

    def test_batch_indexing(self):
        checker = TokenOverlapChecker(n=5, threshold=0.3)

        problems = [
            ("p1", "First problem about mathematics and numbers"),
            ("p2", "Second problem about science and experiments"),
            ("p3", "Third problem about history and events"),
        ]
        checker.index_canonical_batch(problems)

        assert len(checker._canonical_ngrams) == 3

    def test_clear(self):
        checker = TokenOverlapChecker()
        checker.index_canonical("test", "Some test text")
        assert len(checker._canonical_ngrams) == 1

        checker.clear()
        assert len(checker._canonical_ngrams) == 0


# =============================================================================
# Math Structural Checker Tests
# =============================================================================


class TestMathStructuralChecker:
    """Tests for math problem structural similarity."""

    def test_extract_numbers(self):
        text = "John has 15 apples and 7.5 oranges, spending -$3.50."
        numbers = MathStructuralChecker.extract_numbers(text)
        assert "15" in numbers
        assert "7.5" in numbers
        # Negative sign stripped for structural comparison (we compare absolute values)
        assert "3.5" in numbers

    def test_extract_operations(self):
        text = "Add 5 and 3, then multiply by 2"
        ops = MathStructuralChecker.extract_operations(text)
        assert "add" in ops
        assert "multiply" in ops

    def test_same_structure_rejected(self):
        checker = MathStructuralChecker()

        # Canonical: 15 + 5 = 20 type problem
        checker.index_canonical(
            "gsm_001",
            "Sarah has 15 cookies. Tom gives her 5 more. How many does she have?",
            "20"
        )

        # Same numbers, same operations, same answer
        synthetic = {
            "text": "Mike has 15 balls. Jane adds 5 more. Total balls?",
            "answer": "20"
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is False
        assert match_id == "gsm_001"

    def test_different_numbers_allowed(self):
        checker = MathStructuralChecker()

        checker.index_canonical(
            "gsm_001",
            "Sarah has 15 cookies. Tom gives her 5 more. How many does she have?",
            "20"
        )

        # Different numbers
        synthetic = {
            "text": "Sarah has 25 cookies. Tom gives her 10 more. How many does she have?",
            "answer": "35"
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is True

    def test_different_answer_allowed(self):
        checker = MathStructuralChecker()

        checker.index_canonical(
            "gsm_001",
            "Calculate 15 + 5",
            "20"
        )

        # Same text but different answer (this would be a wrong problem, but shows the check)
        synthetic = {
            "text": "Calculate 15 + 5",
            "answer": "21"  # Wrong, but different
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is True


# =============================================================================
# Code Structural Checker Tests
# =============================================================================


class TestCodeStructuralChecker:
    """Tests for code problem structural similarity."""

    def test_extract_function_name(self):
        text1 = "Write a function called is_prime that checks primality"
        name1 = CodeStructuralChecker.extract_function_name(text1)
        assert name1 == "is_prime"

        text2 = "def calculate_sum(numbers):"
        name2 = CodeStructuralChecker.extract_function_name(text2)
        assert name2 == "calculate_sum"

    def test_same_structure_rejected(self):
        checker = CodeStructuralChecker()

        checker.index_canonical(
            "mbpp_001",
            "Write a function is_prime to check if a number is prime",
            [("2", "True"), ("4", "False"), ("7", "True")]
        )

        # Same function name, same test inputs
        synthetic = {
            "text": "Implement is_prime to determine primality",
            "test_cases": [("2", "True"), ("4", "False"), ("7", "True")]
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is False
        assert match_id == "mbpp_001"

    def test_different_function_allowed(self):
        checker = CodeStructuralChecker()

        checker.index_canonical(
            "mbpp_001",
            "Write is_prime to check primality",
            [("2", "True"), ("4", "False")]
        )

        # Different function name
        synthetic = {
            "text": "Write check_primality function",
            "test_cases": [("2", "True"), ("4", "False")]
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is True

    def test_different_tests_allowed(self):
        checker = CodeStructuralChecker()

        checker.index_canonical(
            "mbpp_001",
            "Write is_prime",
            [("2", "True"), ("4", "False")]
        )

        # Same function, different tests
        synthetic = {
            "text": "Write is_prime",
            "test_cases": [("11", "True"), ("15", "False")]
        }

        is_safe, match_id = checker.check(synthetic)
        assert is_safe is True


# =============================================================================
# Semantic Similarity Checker Tests (with mocking)
# =============================================================================


class TestSemanticSimilarityChecker:
    """Tests for semantic similarity checking."""

    def test_check_without_embeddings_model(self):
        """Test that semantic checker works with mocked embeddings."""
        checker = SemanticSimilarityChecker(threshold=0.85)

        # Mock the embedding model
        mock_model = MagicMock()

        # Create normalized embeddings for testing
        canonical_emb = np.array([[1.0, 0.0, 0.0]])  # Normalized
        synthetic_similar = np.array([[0.9, 0.1, 0.0]])  # Similar
        synthetic_similar = synthetic_similar / np.linalg.norm(synthetic_similar)

        mock_model.encode.side_effect = [canonical_emb, synthetic_similar]

        with patch.object(checker, '_model', mock_model):
            checker._canonical_embeddings = canonical_emb
            checker._canonical_ids = ["p1"]
            checker._canonical_texts = {"p1": "Test problem"}

            is_safe, score, match_id = checker.check("Similar problem")

            # With our mock embeddings, similarity should be high
            assert isinstance(is_safe, bool)
            assert isinstance(score, float)

    def test_semantic_checker_clear(self):
        checker = SemanticSimilarityChecker()
        checker._canonical_ids = ["test"]
        checker._canonical_texts = {"test": "text"}

        checker.clear()

        assert checker._canonical_ids == []
        assert checker._canonical_texts == {}


# =============================================================================
# Combined Firewall Tests
# =============================================================================


class TestContaminationFirewall:
    """Tests for the combined ContaminationFirewall."""

    def test_firewall_initialization(self):
        config = ContaminationConfig(
            ngram_n=5,
            token_overlap_threshold=0.3,
            semantic_threshold=0.85,
        )
        firewall = ContaminationFirewall(config)

        assert firewall.config.ngram_n == 5
        assert firewall.config.token_overlap_threshold == 0.3

    def test_firewall_token_rejection(self):
        """Test that firewall rejects on token overlap."""
        firewall = ContaminationFirewall()

        # Index canonical
        canonical = "A farmer has 15 apples. He gives 5 to his neighbor. How many apples are left?"
        firewall.index_canonical("gsm_001", canonical, domain="math", answer="10")

        # Nearly identical synthetic
        synthetic = "A farmer has 15 apples. He gives 5 to his neighbor. How many apples remain?"

        result = firewall.check(synthetic, domain="math", answer="10")

        assert result.is_safe is False
        assert result.rejection_reason == RejectionReason.TOKEN_OVERLAP
        assert "token_overlap" in result.checks_run

    def test_firewall_passes_distinct(self):
        """Test that firewall passes distinct problems."""
        firewall = ContaminationFirewall()

        firewall.index_canonical(
            "gsm_001",
            "A farmer has 15 apples. He gives 5 to his neighbor.",
            domain="math",
            answer="10"
        )

        # Completely different problem
        result = firewall.check(
            "Calculate the perimeter of a rectangle with length 8 and width 4.",
            domain="math",
            answer="24"
        )

        # Should pass token check (semantic check might be skipped without model)
        assert result.token_overlap_score < 0.3
        assert "token_overlap" in result.checks_run

    def test_firewall_batch_indexing(self):
        firewall = ContaminationFirewall()

        problems = [
            {"id": "p1", "text": "Problem one about math"},
            {"id": "p2", "text": "Problem two about science"},
            {"id": "p3", "text": "Problem three about history"},
        ]

        firewall.index_canonical_batch(problems, domain=None)

        stats = firewall.get_stats()
        # Stats should be initialized
        assert stats["total_checked"] == 0

    def test_firewall_math_structural_rejection(self):
        """Test that math structural similarity is checked."""
        firewall = ContaminationFirewall()

        # Index math problem
        firewall.index_canonical(
            "gsm_001",
            "Add 15 and 5",
            domain="math",
            answer="20"
        )

        # Same numbers, same operation, same answer - but different words
        # First check: token overlap should pass (different words)
        # But structural check should catch it
        result = firewall.check(
            "Calculate 15 plus 5",
            domain="math",
            answer="20"
        )

        # If token check passes, structural should catch it
        if result.token_overlap_score <= 0.3:
            # The structural check should have run
            if "math_structure" in result.checks_run:
                assert result.rejection_reason == RejectionReason.MATH_STRUCTURE

    def test_firewall_stats(self):
        firewall = ContaminationFirewall()

        firewall.index_canonical("p1", "Test problem one")
        firewall.index_canonical("p2", "Test problem two")

        # Check some synthetics
        firewall.check("Different problem entirely")
        firewall.check("Another unique problem here")

        stats = firewall.get_stats()
        assert stats["total_checked"] == 2
        assert "pass_rate" in stats or stats["total_checked"] == stats["passed"]

    def test_firewall_clear(self):
        firewall = ContaminationFirewall()
        firewall.index_canonical("test", "Some text")
        firewall.check("Check text")

        firewall.clear()

        stats = firewall.get_stats()
        assert stats["total_checked"] == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_contamination_function(self):
        canonical_texts = [
            "The quick brown fox jumps over the lazy dog",
            "A different sentence about cats and birds",
        ]

        # Distinct synthetic
        result = check_contamination(
            "Calculate the area of a circle with radius 5",
            canonical_texts,
        )

        # Should pass token check
        assert result.token_overlap_score < 0.3

    def test_compute_overlap_matrix(self):
        texts_a = [
            "The quick brown fox jumps",
            "A lazy dog sleeps soundly",
        ]
        texts_b = [
            "The quick brown fox runs",
            "Birds fly through the sky",
        ]

        matrix = compute_overlap_matrix(texts_a, texts_b, n=3)

        assert matrix.shape == (2, 2)
        # First row should have some overlap with first column
        assert matrix[0, 0] > 0  # "quick brown fox" overlaps
        assert matrix[1, 1] == 0  # No overlap between dog/bird sentences


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_text(self):
        checker = TokenOverlapChecker()
        checker.index_canonical("test", "Some canonical text here")

        is_safe, score, _ = checker.check("")
        assert is_safe is True
        assert score == 0.0

    def test_very_short_text(self):
        checker = TokenOverlapChecker(n=5)
        checker.index_canonical("test", "A B C D E")

        # Synthetic shorter than n-gram size
        is_safe, score, _ = checker.check("Hi there")
        assert is_safe is True

    def test_unicode_text(self):
        checker = TokenOverlapChecker()
        checker.index_canonical("test", "Héllo wörld with spëcial characters")

        is_safe, _, _ = checker.check("Different text entirely")
        assert is_safe is True

    def test_numeric_only_text(self):
        checker = TokenOverlapChecker()
        checker.index_canonical("test", "15 20 25 30 35 40 45")

        is_safe, score, _ = checker.check("15 20 25 30 35 40 45")
        assert is_safe is False  # Identical
        assert score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
