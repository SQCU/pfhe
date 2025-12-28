"""
Pytest configuration and shared fixtures.

Provides session-scoped real data fixtures for testing with actual
HuggingFace benchmark samples instead of mocks.
"""

import logging
import pytest
from typing import Optional

# Configure logging for test runs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Real Data Fixtures (Session-Scoped)
# =============================================================================

# Number of real samples to load per benchmark (small for speed, but real)
SAMPLES_PER_BENCHMARK = 3


@pytest.fixture(scope="session")
def benchmark_loader():
    """Session-scoped BenchmarkLoader instance."""
    from phfe.benchmark import BenchmarkLoader
    return BenchmarkLoader()


@pytest.fixture(scope="session")
def real_gsm1k_samples(benchmark_loader):
    """Real GSM1K samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.GSM1K, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real GSM1K samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load GSM1K: {e}")
        pytest.skip(f"GSM1K unavailable: {e}")


@pytest.fixture(scope="session")
def real_arc_easy_samples(benchmark_loader):
    """Real ARC-Easy samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.ARC_EASY, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real ARC-Easy samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load ARC-Easy: {e}")
        pytest.skip(f"ARC-Easy unavailable: {e}")


@pytest.fixture(scope="session")
def real_arc_challenge_samples(benchmark_loader):
    """Real ARC-Challenge samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.ARC_CHALLENGE, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real ARC-Challenge samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load ARC-Challenge: {e}")
        pytest.skip(f"ARC-Challenge unavailable: {e}")


@pytest.fixture(scope="session")
def real_race_samples(benchmark_loader):
    """Real RACE samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.RACE, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real RACE samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load RACE: {e}")
        pytest.skip(f"RACE unavailable: {e}")


@pytest.fixture(scope="session")
def real_boolq_samples(benchmark_loader):
    """Real BoolQ samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.BOOLQ, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real BoolQ samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load BoolQ: {e}")
        pytest.skip(f"BoolQ unavailable: {e}")


@pytest.fixture(scope="session")
def real_hellaswag_samples(benchmark_loader):
    """Real HellaSwag samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.HELLASWAG, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real HellaSwag samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load HellaSwag: {e}")
        pytest.skip(f"HellaSwag unavailable: {e}")


@pytest.fixture(scope="session")
def real_winogrande_samples(benchmark_loader):
    """Real WinoGrande samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.WINOGRANDE, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real WinoGrande samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load WinoGrande: {e}")
        pytest.skip(f"WinoGrande unavailable: {e}")


@pytest.fixture(scope="session")
def real_mbpp_samples(benchmark_loader):
    """Real MBPP samples from HuggingFace."""
    from phfe.benchmark import BenchmarkType
    try:
        samples = benchmark_loader.load(BenchmarkType.MBPP, limit=SAMPLES_PER_BENCHMARK)
        logger.info(f"Loaded {len(samples)} real MBPP samples")
        return samples
    except Exception as e:
        logger.warning(f"Failed to load MBPP: {e}")
        pytest.skip(f"MBPP unavailable: {e}")


@pytest.fixture(scope="session")
def real_benchmark_samples(benchmark_loader):
    """
    All real benchmark samples in a single dict, keyed by BenchmarkType.

    Loads SAMPLES_PER_BENCHMARK samples from each configured benchmark.
    Skips benchmarks that fail to load (network issues, etc).
    """
    from phfe.benchmark import BenchmarkType, BENCHMARK_CONFIGS

    samples = {}
    failed = []

    for benchmark_type in BENCHMARK_CONFIGS.keys():
        try:
            loaded = benchmark_loader.load(benchmark_type, limit=SAMPLES_PER_BENCHMARK)
            samples[benchmark_type] = loaded
            logger.info(f"Loaded {len(loaded)} samples for {benchmark_type.value}")
        except Exception as e:
            logger.warning(f"Failed to load {benchmark_type.value}: {e}")
            failed.append(benchmark_type.value)

    if not samples:
        pytest.skip(f"No benchmarks available. Failed: {failed}")

    logger.info(f"Loaded {len(samples)} benchmarks, {sum(len(v) for v in samples.values())} total samples")
    return samples


# =============================================================================
# Convenience Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def any_real_problem(real_benchmark_samples):
    """A single real problem from any benchmark (for quick sanity checks)."""
    for problems in real_benchmark_samples.values():
        if problems:
            return problems[0]
    pytest.skip("No real problems available")


@pytest.fixture(scope="session")
def real_math_problem(real_gsm1k_samples):
    """A real math problem (GSM1K)."""
    if real_gsm1k_samples:
        return real_gsm1k_samples[0]
    pytest.skip("No GSM1K problems available")


@pytest.fixture(scope="session")
def real_code_problem(real_mbpp_samples):
    """A real code problem (MBPP)."""
    if real_mbpp_samples:
        return real_mbpp_samples[0]
    pytest.skip("No MBPP problems available")


@pytest.fixture(scope="session")
def real_multiple_choice_problem(real_arc_easy_samples):
    """A real multiple choice problem (ARC-Easy)."""
    if real_arc_easy_samples:
        return real_arc_easy_samples[0]
    pytest.skip("No ARC-Easy problems available")


# =============================================================================
# ICR Transform Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def icr_transformer():
    """Session-scoped ICRTransformer instance."""
    from phfe.icr_transform import ICRTransformer
    return ICRTransformer()


@pytest.fixture(scope="session")
def real_icr_transformed_samples(real_benchmark_samples, icr_transformer):
    """
    Real benchmark samples with ICR transformation applied.

    Returns dict mapping BenchmarkType -> list of ICRInstance.
    """
    transformed = {}
    for benchmark_type, problems in real_benchmark_samples.items():
        try:
            instances = []
            for problem in problems:
                instance = icr_transformer.transform(
                    problem_id=problem.problem_id,
                    benchmark=problem.benchmark.value,
                    question_text=problem.text,
                    answer=problem.answer,
                    options=problem.options,
                )
                instances.append(instance)
            transformed[benchmark_type] = instances
            logger.info(f"Transformed {len(instances)} {benchmark_type.value} problems")
        except Exception as e:
            logger.warning(f"Failed to transform {benchmark_type.value}: {e}")

    return transformed


# =============================================================================
# Contamination Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def contamination_firewall():
    """Session-scoped ContaminationFirewall instance."""
    from phfe.benchmark import ContaminationFirewall
    return ContaminationFirewall()


@pytest.fixture(scope="session")
def indexed_firewall(contamination_firewall, real_benchmark_samples):
    """
    ContaminationFirewall pre-indexed with real benchmark samples.

    Use this to test contamination checking against actual canonical problems.
    """
    from phfe.benchmark import BENCHMARK_DOMAINS

    indexed_count = 0
    for benchmark_type, problems in real_benchmark_samples.items():
        domain = BENCHMARK_DOMAINS.get(benchmark_type)
        for problem in problems:
            contamination_firewall.index_canonical(
                problem_id=problem.problem_id,
                text=problem.text,
                domain=domain,
                answer=problem.answer if domain == "math" else None,
                test_cases=problem.test_cases if domain == "code" else None,
            )
            indexed_count += 1

    logger.info(f"Indexed {indexed_count} canonical problems in firewall")
    return contamination_firewall


# =============================================================================
# Cross-Tokenizer Fixtures (lightweight, no model loading)
# =============================================================================


@pytest.fixture
def sparse_logits_factory():
    """Factory for creating SparseLogits test instances."""
    from phfe.distillation.cross_tokenizer import SparseLogits

    def create(token_probs: Optional[dict[int, float]] = None) -> SparseLogits:
        if token_probs is None:
            # Default: simple distribution
            token_probs = {100: 0.5, 200: 0.3, 300: 0.2}
        return SparseLogits(token_probs=token_probs)

    return create


# =============================================================================
# Task Queue Fixtures
# =============================================================================


@pytest.fixture
def temp_task_queue(tmp_path):
    """Task queue with temporary storage directory."""
    from phfe.orchestrator.task_queue import TaskQueue
    return TaskQueue(storage_dir=str(tmp_path / "tasks"))


# =============================================================================
# Markers Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "network: marks tests requiring network access")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
