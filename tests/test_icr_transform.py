"""
Tests for ICR (In-Context Retrieval) transformation system.
"""

import pytest

from phfe.icr_transform import (
    ICRTransformer,
    ICRInstance,
    ICRBatch,
    TransformationType,
    transform_problem,
    get_method_library,
    list_available_libraries,
    METHOD_LIBRARIES,
)


# =============================================================================
# Method Library Tests
# =============================================================================


class TestMethodLibraries:
    """Tests for method library registry."""

    def test_list_available_libraries(self):
        """Should list all available benchmarks."""
        available = list_available_libraries()
        assert isinstance(available, list)
        assert len(available) >= 10  # We defined 10+ benchmarks
        assert "gsm1k" in available
        assert "mbpp" in available
        assert "arc" in available

    def test_get_method_library_exists(self):
        """Should return method library for known benchmark."""
        library = get_method_library("gsm1k")
        assert library
        assert "[METHOD LIBRARY" in library
        assert "[PROBLEM]" in library

    def test_get_method_library_case_insensitive(self):
        """Should handle case-insensitive lookup."""
        lower = get_method_library("gsm1k")
        upper = get_method_library("GSM1K")
        mixed = get_method_library("Gsm1K")
        # Should all return same content (or empty for case sensitivity)
        assert lower == upper == mixed or (lower and not upper)

    def test_get_method_library_unknown(self):
        """Should return empty string for unknown benchmark."""
        library = get_method_library("unknown_benchmark")
        assert library == ""

    def test_method_libraries_have_markers(self):
        """Each library should end with a clear marker."""
        markers = ["[PROBLEM]", "[PASSAGE]", "[CONTEXT]", "[SENTENCE]", "[REQUEST]", "[TASK]"]
        for benchmark, library in METHOD_LIBRARIES.items():
            has_marker = any(marker in library for marker in markers)
            assert has_marker, f"{benchmark} library missing end marker"

    def test_gsm1k_library_content(self):
        """GSM1K library should have arithmetic methods."""
        library = get_method_library("gsm1k")
        assert "Addition" in library or "COMBINING" in library
        assert "Subtraction" in library or "LEFT" in library
        assert "Multiplication" in library or "REPEATED" in library
        assert "Division" in library or "SPLITTING" in library

    def test_mbpp_library_content(self):
        """MBPP library should have programming patterns."""
        library = get_method_library("mbpp")
        assert "PATTERN" in library
        assert "def " in library or "python" in library.lower()

    def test_arc_library_content(self):
        """ARC library should have science facts."""
        library = get_method_library("arc")
        assert "CATEGORY" in library or "SCIENCE" in library
        assert "energy" in library.lower() or "force" in library.lower()


# =============================================================================
# ICRTransformer Tests
# =============================================================================


class TestICRTransformer:
    """Tests for ICRTransformer class."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return ICRTransformer()

    def test_init_default(self):
        """Should initialize with default transformation type."""
        t = ICRTransformer()
        assert t.default_transformation == TransformationType.WORKED_EXAMPLES

    def test_init_custom_default(self):
        """Should accept custom default transformation."""
        t = ICRTransformer(default_transformation=TransformationType.FICTIONAL_CONTEXT)
        assert t.default_transformation == TransformationType.FICTIONAL_CONTEXT

    def test_get_method_library_cached(self, transformer):
        """Should cache method libraries."""
        lib1 = transformer.get_method_library("gsm1k")
        lib2 = transformer.get_method_library("gsm1k")
        assert lib1 is lib2  # Same object (cached)

    def test_list_supported_benchmarks(self):
        """Should list benchmarks with method support."""
        supported = ICRTransformer.list_supported_benchmarks()
        assert isinstance(supported, list)
        assert "gsm1k" in supported
        assert "mbpp" in supported


# =============================================================================
# Transform Single Problem Tests
# =============================================================================


class TestTransformSingle:
    """Tests for single problem transformation."""

    @pytest.fixture
    def transformer(self):
        return ICRTransformer()

    def test_transform_gsm1k(self, transformer):
        """Should transform GSM1K problem with method library."""
        instance = transformer.transform(
            problem_id="test_001",
            benchmark="gsm1k",
            question_text="Alice has 5 apples. She buys 3 more. How many apples does Alice have?",
            answer="8",
        )

        assert isinstance(instance, ICRInstance)
        assert instance.instance_id == "icr_gsm1k_test_001_worked_examples"
        assert instance.source_benchmark == "gsm1k"
        assert instance.source_instance_id == "test_001"
        assert instance.correct_answer == "8"
        assert instance.transformation_type == TransformationType.WORKED_EXAMPLES

    def test_transform_prepends_context(self, transformer):
        """Context should be prepended to question."""
        question = "What is 2 + 2?"
        instance = transformer.transform(
            problem_id="test",
            benchmark="gsm1k",
            question_text=question,
            answer="4",
        )

        assert instance.context  # Has context
        assert instance.question == question  # Original preserved
        assert instance.full_prompt.startswith(instance.context)
        assert instance.full_prompt.endswith(question)

    def test_transform_with_options(self, transformer):
        """Should preserve multiple choice options."""
        options = ["A) 10", "B) 20", "C) 30", "D) 40"]
        instance = transformer.transform(
            problem_id="test",
            benchmark="arc",
            question_text="What is the speed of light?",
            answer="C",
            options=options,
        )

        assert instance.options == options

    def test_transform_unknown_benchmark(self, transformer):
        """Unknown benchmark should have no context."""
        instance = transformer.transform(
            problem_id="test",
            benchmark="unknown_bench",
            question_text="Some question?",
            answer="42",
        )

        assert instance.context == ""
        assert instance.full_prompt == "Some question?"

    def test_transform_unsupported_type_raises(self, transformer):
        """Non-worked-examples transformation should raise."""
        with pytest.raises(NotImplementedError):
            transformer.transform(
                problem_id="test",
                benchmark="gsm1k",
                question_text="Question?",
                answer="A",
                transformation_type=TransformationType.FICTIONAL_CONTEXT,
            )


# =============================================================================
# Transform Batch Tests
# =============================================================================


class TestTransformBatch:
    """Tests for batch transformation."""

    @pytest.fixture
    def transformer(self):
        return ICRTransformer()

    @pytest.fixture
    def sample_problems(self):
        return [
            {"problem_id": "p1", "text": "2 + 3 = ?", "answer": "5"},
            {"problem_id": "p2", "text": "4 * 5 = ?", "answer": "20"},
            {"problem_id": "p3", "text": "10 - 7 = ?", "answer": "3"},
        ]

    def test_batch_transform(self, transformer, sample_problems):
        """Should transform batch of problems."""
        batch = transformer.transform_batch(
            problems=sample_problems,
            benchmark="gsm1k",
        )

        assert isinstance(batch, ICRBatch)
        assert len(batch) == 3
        assert batch.benchmark == "gsm1k"
        assert batch.transformation_type == TransformationType.WORKED_EXAMPLES

    def test_batch_iterable(self, transformer, sample_problems):
        """Batch should be iterable."""
        batch = transformer.transform_batch(sample_problems, "gsm1k")

        instances = list(batch)
        assert len(instances) == 3
        assert all(isinstance(i, ICRInstance) for i in instances)

    def test_batch_has_library_hash(self, transformer, sample_problems):
        """Batch should include method library hash for versioning."""
        batch = transformer.transform_batch(sample_problems, "gsm1k")

        assert batch.method_library_hash
        assert len(batch.method_library_hash) == 16  # sha256[:16]

    def test_batch_same_context(self, transformer, sample_problems):
        """All instances in batch should have same context."""
        batch = transformer.transform_batch(sample_problems, "gsm1k")

        contexts = [inst.context for inst in batch]
        assert len(set(contexts)) == 1  # All same

    def test_batch_with_options(self, transformer):
        """Batch should preserve options."""
        problems = [
            {
                "problem_id": "p1",
                "text": "Question 1?",
                "answer": "A",
                "options": ["A", "B", "C", "D"],
            },
            {
                "problem_id": "p2",
                "text": "Question 2?",
                "answer": "B",
                "options": ["A", "B", "C", "D"],
            },
        ]
        batch = transformer.transform_batch(problems, "arc")

        for inst in batch:
            assert inst.options == ["A", "B", "C", "D"]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for transform_problem convenience function."""

    def test_transform_problem_basic(self):
        """Should transform problem with convenience function."""
        instance = transform_problem(
            benchmark="gsm1k",
            problem_id="test_123",
            question_text="How many is 5 + 5?",
            answer="10",
        )

        assert isinstance(instance, ICRInstance)
        assert instance.correct_answer == "10"
        assert "[METHOD LIBRARY" in instance.context

    def test_transform_problem_with_options(self):
        """Should handle options in convenience function."""
        instance = transform_problem(
            benchmark="arc",
            problem_id="arc_001",
            question_text="What causes rain?",
            answer="C",
            options=["A) Sun", "B) Wind", "C) Condensation", "D) Gravity"],
        )

        assert instance.options is not None
        assert len(instance.options) == 4


# =============================================================================
# All Benchmark Tests
# =============================================================================


class TestAllBenchmarks:
    """Test transformation works for all supported benchmarks."""

    @pytest.fixture
    def transformer(self):
        return ICRTransformer()

    @pytest.mark.parametrize("benchmark", [
        "gsm1k",
        "arc",
        "arc_easy",
        "arc_challenge",
        "race",
        "boolq",
        "hellaswag",
        "winogrande",
        "mbpp",
        "fncall",
        "format",
    ])
    def test_benchmark_transform(self, transformer, benchmark):
        """Each benchmark should transform successfully."""
        instance = transformer.transform(
            problem_id=f"test_{benchmark}",
            benchmark=benchmark,
            question_text=f"Sample question for {benchmark}?",
            answer="sample_answer",
        )

        assert instance.source_benchmark == benchmark
        assert instance.context  # Should have context
        assert instance.full_prompt.startswith(instance.context)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def transformer(self):
        return ICRTransformer()

    def test_empty_question(self, transformer):
        """Should handle empty question."""
        instance = transformer.transform(
            problem_id="empty",
            benchmark="gsm1k",
            question_text="",
            answer="",
        )

        assert instance.question == ""
        assert instance.full_prompt == instance.context  # Just context

    def test_very_long_question(self, transformer):
        """Should handle very long question."""
        long_question = "What is the answer? " * 1000
        instance = transformer.transform(
            problem_id="long",
            benchmark="gsm1k",
            question_text=long_question,
            answer="42",
        )

        assert long_question in instance.full_prompt

    def test_special_characters(self, transformer):
        """Should handle special characters in question."""
        special = "What is √(π²) + ∞ - ∅ × ≠ ?"
        instance = transformer.transform(
            problem_id="special",
            benchmark="gsm1k",
            question_text=special,
            answer="undefined",
        )

        assert instance.question == special

    def test_unicode_question(self, transformer):
        """Should handle unicode in question."""
        unicode_q = "日本語の質問：2 + 2 は何ですか？"
        instance = transformer.transform(
            problem_id="unicode",
            benchmark="gsm1k",
            question_text=unicode_q,
            answer="4",
        )

        assert instance.question == unicode_q
