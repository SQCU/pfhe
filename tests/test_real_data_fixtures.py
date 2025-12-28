"""
Tests that verify real data fixtures work correctly.

These tests use actual HuggingFace data to validate that our
loaders, transformers, and contamination checkers work with
type-identical real data, not just mocks.
"""

import pytest

from phfe.benchmark import BenchmarkType, BenchmarkProblem


class TestRealBenchmarkLoading:
    """Verify real benchmark samples load correctly."""

    def test_real_samples_dict_not_empty(self, real_benchmark_samples):
        """Should load at least some benchmarks."""
        assert len(real_benchmark_samples) > 0, "No benchmarks loaded"

    def test_real_samples_are_benchmark_problems(self, real_benchmark_samples):
        """Each sample should be a BenchmarkProblem instance."""
        for benchmark_type, problems in real_benchmark_samples.items():
            assert isinstance(benchmark_type, BenchmarkType)
            for problem in problems:
                assert isinstance(problem, BenchmarkProblem)
                assert problem.benchmark == benchmark_type

    def test_gsm1k_has_math_content(self, real_gsm1k_samples):
        """GSM1K samples should have math-like content."""
        for problem in real_gsm1k_samples:
            assert problem.text, "GSM1K problem missing text"
            assert problem.answer, "GSM1K problem missing answer"
            assert problem.answer_type == "number"
            # GSM answers contain #### marker
            assert "####" in problem.answer or problem.answer.strip().isdigit() or any(
                c.isdigit() for c in problem.answer
            )

    def test_arc_easy_has_options(self, real_arc_easy_samples):
        """ARC-Easy samples should have multiple choice options."""
        for problem in real_arc_easy_samples:
            assert problem.text, "ARC-Easy problem missing text"
            assert problem.options is not None, "ARC-Easy missing options"
            assert len(problem.options) >= 2, "ARC-Easy should have 2+ options"
            assert problem.answer_type == "multiple_choice"

    def test_mbpp_has_code(self, real_mbpp_samples):
        """MBPP samples should have code answer type."""
        for problem in real_mbpp_samples:
            assert problem.text, "MBPP problem missing text"
            assert problem.answer_type == "code"
            # MBPP should have test cases
            assert problem.test_cases is not None, "MBPP should have test_cases"

    def test_boolq_has_passage(self, real_boolq_samples):
        """BoolQ samples should have combined passage+question."""
        for problem in real_boolq_samples:
            assert problem.text, "BoolQ problem missing text"
            assert problem.answer in ("true", "false"), f"BoolQ answer should be true/false, got {problem.answer}"

    def test_winogrande_has_binary_options(self, real_winogrande_samples):
        """WinoGrande should have exactly 2 options."""
        for problem in real_winogrande_samples:
            assert problem.text, "WinoGrande problem missing text"
            assert problem.options is not None
            assert len(problem.options) == 2, "WinoGrande should have exactly 2 options"


class TestRealICRTransform:
    """Verify ICR transformation works on real data."""

    def test_transform_real_samples(self, real_icr_transformed_samples):
        """Should transform real samples successfully."""
        assert len(real_icr_transformed_samples) > 0, "No transformed samples"

        for benchmark_type, instances in real_icr_transformed_samples.items():
            for instance in instances:
                # ICRInstance should have context (method library) and question
                assert instance.context, "Missing context (method library)"
                assert instance.question, "Missing question"
                assert instance.full_prompt, "Missing full_prompt"
                # Full prompt should contain both context and question
                assert len(instance.full_prompt) > len(instance.question)


class TestRealContaminationChecking:
    """Verify contamination checking works on real data."""

    def test_indexed_firewall_has_problems(self, indexed_firewall):
        """Firewall should be indexed with real problems."""
        # Check that token checker has indexed problems
        indexed_count = len(indexed_firewall.token_checker._canonical_ngrams)
        assert indexed_count > 0, "No problems indexed in token checker"

    def test_exact_match_detected(self, indexed_firewall, any_real_problem):
        """Exact match to canonical should be detected as contaminated."""
        from phfe.benchmark import BENCHMARK_DOMAINS, RejectionReason

        domain = BENCHMARK_DOMAINS.get(any_real_problem.benchmark)
        result = indexed_firewall.check(
            synthetic_text=any_real_problem.text,  # Exact same text
            domain=domain,
            answer=any_real_problem.answer if domain == "math" else None,
            test_cases=any_real_problem.test_cases if domain == "code" else None,
        )

        # Should be rejected (contaminated)
        assert result.rejection_reason != RejectionReason.PASSED, (
            f"Exact match should be detected as contaminated, got {result.rejection_reason}"
        )

    def test_novel_text_passes(self, indexed_firewall):
        """Completely novel text should pass contamination check."""
        from phfe.benchmark import RejectionReason

        # Totally different text that can't match any benchmark
        novel_text = (
            "In the mystical kingdom of Zorbax, the purple elephants "
            "dance under three moons while singing ancient algorithms. "
            "Calculate the total number of tentacles if each elephant "
            "has sqrt(42) tentacles and there are pi elephants."
        )

        result = indexed_firewall.check(
            synthetic_text=novel_text,
            domain="math",
            answer="42",
        )

        assert result.rejection_reason == RejectionReason.PASSED, (
            f"Novel text should pass, got {result.rejection_reason}"
        )


class TestConvenienceFixtures:
    """Test the convenience fixtures work."""

    def test_any_real_problem(self, any_real_problem):
        """Should get a real problem."""
        assert isinstance(any_real_problem, BenchmarkProblem)
        assert any_real_problem.text

    def test_real_math_problem(self, real_math_problem):
        """Should get a math problem."""
        assert real_math_problem.benchmark == BenchmarkType.GSM1K

    def test_real_code_problem(self, real_code_problem):
        """Should get a code problem."""
        assert real_code_problem.benchmark == BenchmarkType.MBPP

    def test_real_multiple_choice_problem(self, real_multiple_choice_problem):
        """Should get a multiple choice problem."""
        assert real_multiple_choice_problem.options is not None
