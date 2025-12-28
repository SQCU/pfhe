"""
Benchmark Loading from HuggingFace Datasets

Loads canonical benchmarks and converts them to BenchmarkProblem format.
Each benchmark has a specific HuggingFace dataset ID and field mappings.
"""

from dataclasses import dataclass
from typing import Optional, Iterator
import logging

try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None

from . import BenchmarkType, BenchmarkProblem

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for loading a specific benchmark."""

    dataset_id: str
    subset: Optional[str] = None
    split: str = "test"

    # Field mappings (dataset field -> our field)
    id_field: str = "id"
    text_field: str = "question"
    answer_field: str = "answer"
    options_field: Optional[str] = None  # For multiple choice

    # Answer type
    answer_type: str = "text"  # "text", "number", "multiple_choice", "code"

    # Optional processing
    text_template: Optional[str] = None  # Format string for text
    combine_fields: Optional[list[str]] = None  # Fields to combine into text


# Benchmark configurations
BENCHMARK_CONFIGS: dict[BenchmarkType, BenchmarkConfig] = {
    BenchmarkType.GSM1K: BenchmarkConfig(
        dataset_id="openai/gsm8k",  # GSM1K is a subset, but we use GSM8K
        subset="main",
        split="test",
        id_field="question",  # No explicit ID, use question hash
        text_field="question",
        answer_field="answer",
        answer_type="number",
    ),
    BenchmarkType.ARC_EASY: BenchmarkConfig(
        dataset_id="allenai/ai2_arc",
        subset="ARC-Easy",
        split="test",
        id_field="id",
        text_field="question",
        answer_field="answerKey",
        options_field="choices",
        answer_type="multiple_choice",
    ),
    BenchmarkType.ARC_CHALLENGE: BenchmarkConfig(
        dataset_id="allenai/ai2_arc",
        subset="ARC-Challenge",
        split="test",
        id_field="id",
        text_field="question",
        answer_field="answerKey",
        options_field="choices",
        answer_type="multiple_choice",
    ),
    BenchmarkType.RACE: BenchmarkConfig(
        dataset_id="ehovy/race",
        subset="all",
        split="test",
        id_field="example_id",
        text_field="article",  # Need to combine with question
        answer_field="answer",
        options_field="options",
        answer_type="multiple_choice",
        combine_fields=["article", "question"],
    ),
    BenchmarkType.BOOLQ: BenchmarkConfig(
        dataset_id="google/boolq",
        split="validation",  # BoolQ test set is hidden
        id_field="question",  # No explicit ID
        text_field="question",
        answer_field="answer",
        answer_type="text",  # true/false as text
        combine_fields=["passage", "question"],
    ),
    BenchmarkType.HELLASWAG: BenchmarkConfig(
        dataset_id="Rowan/hellaswag",
        split="validation",  # Test set labels hidden
        id_field="ind",
        text_field="ctx",
        answer_field="label",
        options_field="endings",
        answer_type="multiple_choice",
    ),
    BenchmarkType.WINOGRANDE: BenchmarkConfig(
        dataset_id="allenai/winogrande",
        subset="winogrande_xl",
        split="validation",  # Test set labels hidden
        id_field="sentence",  # No explicit ID
        text_field="sentence",
        answer_field="answer",
        options_field=None,  # Options are option1, option2
        answer_type="multiple_choice",
    ),
    BenchmarkType.MBPP: BenchmarkConfig(
        dataset_id="google-research-datasets/mbpp",
        subset="full",
        split="test",
        id_field="task_id",
        text_field="text",
        answer_field="code",
        answer_type="code",
    ),
}


class BenchmarkLoader:
    """Load and iterate canonical benchmarks from HuggingFace."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library required for benchmark loading. "
                "Install with: pip install datasets"
            )
        self.cache_dir = cache_dir
        self._loaded: dict[BenchmarkType, list[BenchmarkProblem]] = {}

    def load(
        self,
        benchmark: BenchmarkType,
        limit: Optional[int] = None,
        force_reload: bool = False,
    ) -> list[BenchmarkProblem]:
        """
        Load a canonical benchmark.

        Args:
            benchmark: Which benchmark to load
            limit: Maximum number of problems to load (for testing)
            force_reload: Bypass cache and reload from HuggingFace

        Returns:
            List of BenchmarkProblem instances
        """
        if benchmark in self._loaded and not force_reload:
            problems = self._loaded[benchmark]
            return problems[:limit] if limit else problems

        if benchmark not in BENCHMARK_CONFIGS:
            raise ValueError(
                f"No configuration for benchmark {benchmark}. "
                f"Available: {list(BENCHMARK_CONFIGS.keys())}"
            )

        config = BENCHMARK_CONFIGS[benchmark]
        problems = list(self._load_benchmark(benchmark, config, limit))

        if not limit:  # Only cache full loads
            self._loaded[benchmark] = problems

        logger.info(f"Loaded {len(problems)} problems from {benchmark.value}")
        return problems

    def _load_benchmark(
        self,
        benchmark: BenchmarkType,
        config: BenchmarkConfig,
        limit: Optional[int],
    ) -> Iterator[BenchmarkProblem]:
        """Load and convert a benchmark dataset."""
        # Load from HuggingFace
        logger.info(
            f"Loading {config.dataset_id}"
            f"{f'/{config.subset}' if config.subset else ''} ({config.split})"
        )

        try:
            if config.subset:
                dataset = load_dataset(
                    config.dataset_id,
                    config.subset,
                    split=config.split,
                    cache_dir=self.cache_dir,
                )
            else:
                dataset = load_dataset(
                    config.dataset_id,
                    split=config.split,
                    cache_dir=self.cache_dir,
                )
        except Exception as e:
            logger.error(f"Failed to load {benchmark}: {e}")
            raise

        # Convert each example
        for idx, example in enumerate(dataset):
            if limit and idx >= limit:
                break

            problem = self._convert_example(benchmark, config, example, idx)
            if problem:
                yield problem

    def _convert_example(
        self,
        benchmark: BenchmarkType,
        config: BenchmarkConfig,
        example: dict,
        idx: int,
    ) -> Optional[BenchmarkProblem]:
        """Convert a dataset example to BenchmarkProblem."""
        try:
            # Extract problem ID
            if config.id_field in example:
                problem_id = str(example[config.id_field])
            else:
                # Generate ID from index
                problem_id = f"{benchmark.value}_{idx}"

            # Extract text (possibly combining fields)
            if config.combine_fields:
                text_parts = []
                for field in config.combine_fields:
                    if field in example:
                        text_parts.append(str(example[field]))
                text = "\n\n".join(text_parts)
            else:
                text = str(example.get(config.text_field, ""))

            # Apply template if specified
            if config.text_template:
                text = config.text_template.format(**example)

            # Extract answer
            answer = example.get(config.answer_field, "")
            if isinstance(answer, bool):
                answer = "true" if answer else "false"
            else:
                answer = str(answer)

            # Extract options for multiple choice
            options = None
            if config.options_field and config.options_field in example:
                raw_options = example[config.options_field]
                if isinstance(raw_options, dict):
                    # ARC format: {"label": ["A", "B", ...], "text": ["opt1", "opt2", ...]}
                    if "text" in raw_options:
                        labels = raw_options.get("label", [])
                        texts = raw_options.get("text", [])
                        options = [f"{l}) {t}" for l, t in zip(labels, texts)]
                elif isinstance(raw_options, list):
                    options = list(raw_options)

            # Special handling for WinoGrande (options are separate fields)
            if benchmark == BenchmarkType.WINOGRANDE:
                opt1 = example.get("option1", "")
                opt2 = example.get("option2", "")
                options = [f"1) {opt1}", f"2) {opt2}"]

            # Extract test cases for code problems
            test_cases = None
            if config.answer_type == "code":
                if "test_list" in example:
                    test_cases = example["test_list"]
                elif "test_setup_code" in example and "test_string" in example:
                    test_cases = [example["test_string"]]

            return BenchmarkProblem(
                problem_id=problem_id,
                benchmark=benchmark,
                text=text,
                answer=answer,
                answer_type=config.answer_type,
                options=options,
                test_cases=test_cases,
                metadata={
                    "dataset_id": config.dataset_id,
                    "subset": config.subset,
                    "split": config.split,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to convert example {idx}: {e}")
            return None

    def load_all(
        self,
        benchmarks: Optional[list[BenchmarkType]] = None,
        limit_per_benchmark: Optional[int] = None,
    ) -> dict[BenchmarkType, list[BenchmarkProblem]]:
        """
        Load multiple benchmarks.

        Args:
            benchmarks: Which benchmarks to load (default: all configured)
            limit_per_benchmark: Max problems per benchmark

        Returns:
            Dict mapping benchmark type to problem list
        """
        if benchmarks is None:
            benchmarks = list(BENCHMARK_CONFIGS.keys())

        result = {}
        for benchmark in benchmarks:
            try:
                result[benchmark] = self.load(benchmark, limit=limit_per_benchmark)
            except Exception as e:
                logger.error(f"Failed to load {benchmark}: {e}")
                result[benchmark] = []

        return result

    @staticmethod
    def list_available() -> list[BenchmarkType]:
        """List benchmarks with loading configurations."""
        return list(BENCHMARK_CONFIGS.keys())

    @staticmethod
    def get_config(benchmark: BenchmarkType) -> Optional[BenchmarkConfig]:
        """Get the loading configuration for a benchmark."""
        return BENCHMARK_CONFIGS.get(benchmark)


def load_benchmark(
    benchmark: BenchmarkType,
    limit: Optional[int] = None,
) -> list[BenchmarkProblem]:
    """
    Convenience function to load a benchmark.

    Example:
        >>> problems = load_benchmark(BenchmarkType.GSM1K, limit=10)
        >>> print(len(problems))
        10
    """
    loader = BenchmarkLoader()
    return loader.load(benchmark, limit=limit)
