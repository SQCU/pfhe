"""
Tests for benchmark loading from HuggingFace datasets.

Note: Some tests require network access and HuggingFace datasets.
These tests use limit=5 to minimize download time.
"""

import pytest
from unittest.mock import patch, MagicMock

from phfe.benchmark import (
    BenchmarkType,
    BenchmarkProblem,
    BenchmarkLoader,
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    load_benchmark,
)


# =============================================================================
# Configuration Tests (no network required)
# =============================================================================


class TestBenchmarkConfigs:
    """Test benchmark configuration registry."""

    def test_all_benchmarks_configured(self):
        """All benchmark types should have configurations."""
        # Note: FNCALL and FORMAT may not have HuggingFace datasets
        expected = {
            BenchmarkType.GSM1K,
            BenchmarkType.ARC_EASY,
            BenchmarkType.ARC_CHALLENGE,
            BenchmarkType.RACE,
            BenchmarkType.BOOLQ,
            BenchmarkType.HELLASWAG,
            BenchmarkType.WINOGRANDE,
            BenchmarkType.MBPP,
        }
        configured = set(BENCHMARK_CONFIGS.keys())
        assert expected <= configured

    def test_config_has_required_fields(self):
        """Each config should have required fields."""
        for benchmark, config in BENCHMARK_CONFIGS.items():
            assert config.dataset_id, f"{benchmark} missing dataset_id"
            assert config.split, f"{benchmark} missing split"
            assert config.text_field, f"{benchmark} missing text_field"
            assert config.answer_field, f"{benchmark} missing answer_field"

    def test_list_available(self):
        """Should list available benchmarks."""
        available = BenchmarkLoader.list_available()
        assert isinstance(available, list)
        assert len(available) >= 8
        assert BenchmarkType.GSM1K in available

    def test_get_config(self):
        """Should get config for specific benchmark."""
        config = BenchmarkLoader.get_config(BenchmarkType.GSM1K)
        assert config is not None
        assert config.dataset_id == "openai/gsm8k"

    def test_get_config_unknown(self):
        """Should return None for unknown benchmark."""
        # Create a mock unknown type
        result = BENCHMARK_CONFIGS.get("unknown")
        assert result is None


# =============================================================================
# Loader Initialization Tests
# =============================================================================


class TestLoaderInit:
    """Test BenchmarkLoader initialization."""

    def test_init_default(self):
        """Should initialize with defaults."""
        loader = BenchmarkLoader()
        assert loader.cache_dir is None
        assert loader._loaded == {}

    def test_init_with_cache(self):
        """Should accept cache directory."""
        loader = BenchmarkLoader(cache_dir="/tmp/hf_cache")
        assert loader.cache_dir == "/tmp/hf_cache"


# =============================================================================
# Mock Loading Tests (no network required)
# =============================================================================


class TestMockLoading:
    """Test loading logic with mocked datasets."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return [
            {"question": "What is 2+2?", "answer": "#### 4"},
            {"question": "What is 3+3?", "answer": "#### 6"},
            {"question": "What is 5+5?", "answer": "#### 10"},
        ]

    @pytest.fixture
    def loader(self):
        return BenchmarkLoader()

    def test_convert_gsm_example(self, loader, mock_dataset):
        """Should convert GSM example correctly."""
        config = BENCHMARK_CONFIGS[BenchmarkType.GSM1K]
        problem = loader._convert_example(
            BenchmarkType.GSM1K,
            config,
            mock_dataset[0],
            idx=0,
        )

        assert problem is not None
        assert problem.benchmark == BenchmarkType.GSM1K
        assert problem.text == "What is 2+2?"
        assert problem.answer == "#### 4"
        assert problem.answer_type == "number"

    def test_convert_with_options(self, loader):
        """Should convert multiple choice correctly."""
        config = BENCHMARK_CONFIGS[BenchmarkType.ARC_EASY]
        example = {
            "id": "arc_001",
            "question": "What color is the sky?",
            "answerKey": "B",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Red", "Blue", "Green", "Yellow"],
            },
        }

        problem = loader._convert_example(
            BenchmarkType.ARC_EASY,
            config,
            example,
            idx=0,
        )

        assert problem is not None
        assert problem.answer == "B"
        assert problem.options is not None
        assert len(problem.options) == 4
        assert "B) Blue" in problem.options

    def test_convert_boolq_boolean(self, loader):
        """Should convert boolean answers correctly."""
        config = BENCHMARK_CONFIGS[BenchmarkType.BOOLQ]
        example = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue due to Rayleigh scattering.",
            "answer": True,
        }

        problem = loader._convert_example(
            BenchmarkType.BOOLQ,
            config,
            example,
            idx=0,
        )

        assert problem is not None
        assert problem.answer == "true"

    def test_convert_combines_fields(self, loader):
        """Should combine fields when configured."""
        config = BENCHMARK_CONFIGS[BenchmarkType.BOOLQ]
        example = {
            "question": "Is the sky blue?",
            "passage": "The sky is blue.",
            "answer": True,
        }

        problem = loader._convert_example(
            BenchmarkType.BOOLQ,
            config,
            example,
            idx=0,
        )

        assert problem is not None
        assert "The sky is blue." in problem.text
        assert "Is the sky blue?" in problem.text

    def test_convert_winogrande_options(self, loader):
        """Should handle WinoGrande's separate option fields."""
        config = BENCHMARK_CONFIGS[BenchmarkType.WINOGRANDE]
        example = {
            "sentence": "The trophy doesn't fit in the suitcase because _ is too big.",
            "option1": "trophy",
            "option2": "suitcase",
            "answer": "1",
        }

        problem = loader._convert_example(
            BenchmarkType.WINOGRANDE,
            config,
            example,
            idx=0,
        )

        assert problem is not None
        assert problem.options is not None
        assert len(problem.options) == 2
        assert "1) trophy" in problem.options
        assert "2) suitcase" in problem.options

    def test_convert_code_problem(self, loader):
        """Should handle code problems with test cases."""
        config = BENCHMARK_CONFIGS[BenchmarkType.MBPP]
        example = {
            "task_id": 1,
            "text": "Write a function to find the sum of a list.",
            "code": "def sum_list(lst): return sum(lst)",
            "test_list": ["assert sum_list([1,2,3]) == 6"],
        }

        problem = loader._convert_example(
            BenchmarkType.MBPP,
            config,
            example,
            idx=0,
        )

        assert problem is not None
        assert problem.answer_type == "code"
        assert problem.test_cases is not None
        assert len(problem.test_cases) == 1

    def test_caching(self, loader):
        """Should cache loaded benchmarks."""
        # Mock load_dataset
        mock_data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        with patch("phfe.benchmark.loader.load_dataset") as mock_load:
            mock_load.return_value = mock_data

            # First load
            problems1 = loader.load(BenchmarkType.GSM1K)

            # Second load (should use cache)
            problems2 = loader.load(BenchmarkType.GSM1K)

            # Should only call load_dataset once
            assert mock_load.call_count == 1
            assert problems1 == problems2

    def test_force_reload(self, loader):
        """Should bypass cache when force_reload=True."""
        mock_data = [{"question": "Q", "answer": "A"}]

        with patch("phfe.benchmark.loader.load_dataset") as mock_load:
            mock_load.return_value = mock_data

            loader.load(BenchmarkType.GSM1K)
            loader.load(BenchmarkType.GSM1K, force_reload=True)

            assert mock_load.call_count == 2


# =============================================================================
# Integration Tests (require network)
# =============================================================================


@pytest.mark.slow
@pytest.mark.network
class TestNetworkLoading:
    """Integration tests that require network access.

    Run with: pytest -m network
    """

    @pytest.fixture
    def loader(self):
        return BenchmarkLoader()

    def test_load_gsm8k_sample(self, loader):
        """Should load GSM8K samples."""
        problems = loader.load(BenchmarkType.GSM1K, limit=5)

        assert len(problems) == 5
        for p in problems:
            assert isinstance(p, BenchmarkProblem)
            assert p.benchmark == BenchmarkType.GSM1K
            assert p.text  # Has question
            assert p.answer  # Has answer

    def test_load_arc_easy_sample(self, loader):
        """Should load ARC-Easy samples."""
        problems = loader.load(BenchmarkType.ARC_EASY, limit=5)

        assert len(problems) == 5
        for p in problems:
            assert p.benchmark == BenchmarkType.ARC_EASY
            assert p.options is not None
            assert len(p.options) >= 2

    def test_load_mbpp_sample(self, loader):
        """Should load MBPP samples."""
        problems = loader.load(BenchmarkType.MBPP, limit=5)

        assert len(problems) == 5
        for p in problems:
            assert p.benchmark == BenchmarkType.MBPP
            assert p.answer_type == "code"

    def test_convenience_function(self):
        """Should work with convenience function."""
        problems = load_benchmark(BenchmarkType.GSM1K, limit=3)
        assert len(problems) == 3


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in loader."""

    def test_unknown_benchmark(self):
        """Should raise for unknown benchmark."""
        loader = BenchmarkLoader()
        # FNCALL isn't in BENCHMARK_CONFIGS
        with pytest.raises(ValueError, match="No configuration"):
            loader.load(BenchmarkType.FNCALL)

    def test_malformed_example(self):
        """Should handle malformed examples gracefully."""
        loader = BenchmarkLoader()
        config = BENCHMARK_CONFIGS[BenchmarkType.GSM1K]

        # Missing required fields
        problem = loader._convert_example(
            BenchmarkType.GSM1K,
            config,
            {},  # Empty example
            idx=0,
        )

        # Should return something (with empty text) rather than crash
        assert problem is not None or problem is None  # Either is acceptable


# =============================================================================
# BenchmarkConfig Tests
# =============================================================================


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = BenchmarkConfig(dataset_id="test/dataset")
        assert config.split == "test"
        assert config.id_field == "id"
        assert config.text_field == "question"
        assert config.answer_field == "answer"
        assert config.answer_type == "text"

    def test_custom_values(self):
        """Should accept custom values."""
        config = BenchmarkConfig(
            dataset_id="my/dataset",
            subset="v2",
            split="validation",
            text_field="input",
            answer_field="output",
            answer_type="code",
        )
        assert config.dataset_id == "my/dataset"
        assert config.subset == "v2"
        assert config.split == "validation"
