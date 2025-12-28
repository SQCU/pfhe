"""
PFE Benchmark Suite

Implements the three-split architecture:
1. Canonical eval (untouched original benchmarks)
2. Synthetic training corpus (contamination-checked)
3. ICR-augmented eval (same problems with method libraries)

Key components:
- ContaminationFirewall: Verify synthetic problems are distinct from canonical
- BenchmarkLoader: Load and iterate canonical benchmarks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Import contamination checking components
from .contamination import (
    ContaminationConfig,
    ContaminationFirewall,
    ContaminationResult,
    RejectionReason,
    TokenOverlapChecker,
    SemanticSimilarityChecker,
    MathStructuralChecker,
    CodeStructuralChecker,
    check_contamination,
    compute_overlap_matrix,
    simple_tokenize,
    extract_ngrams,
)


class BenchmarkType(str, Enum):
    """Supported benchmark types."""

    GSM1K = "gsm1k"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    RACE = "race"
    BOOLQ = "boolq"
    HELLASWAG = "hellaswag"
    WINOGRANDE = "winogrande"
    MBPP = "mbpp"
    FNCALL = "fncall"
    FORMAT = "format"


# Domain mapping for structural checks
BENCHMARK_DOMAINS = {
    BenchmarkType.GSM1K: "math",
    BenchmarkType.ARC_EASY: None,
    BenchmarkType.ARC_CHALLENGE: None,
    BenchmarkType.RACE: None,
    BenchmarkType.BOOLQ: None,
    BenchmarkType.HELLASWAG: None,
    BenchmarkType.WINOGRANDE: None,
    BenchmarkType.MBPP: "code",
    BenchmarkType.FNCALL: "code",
    BenchmarkType.FORMAT: None,
}


@dataclass
class BenchmarkProblem:
    """A single benchmark problem."""

    problem_id: str
    benchmark: BenchmarkType
    text: str
    answer: str
    answer_type: str  # "number", "multiple_choice", "code", etc.
    options: Optional[list[str]] = None
    test_cases: Optional[list] = None  # For code problems
    metadata: dict = field(default_factory=dict)

    @property
    def domain(self) -> Optional[str]:
        """Get the domain for structural checking."""
        return BENCHMARK_DOMAINS.get(self.benchmark)


class CanonicalIndex:
    """
    Index of all canonical problems for contamination checking.

    Wraps ContaminationFirewall with benchmark-aware loading.
    """

    def __init__(self, benchmarks: list[BenchmarkType]):
        self.benchmarks = benchmarks
        self._problems: dict[BenchmarkType, list[BenchmarkProblem]] = {}
        self._firewall = ContaminationFirewall()
        self._loaded = False

    def add_problem(self, problem: BenchmarkProblem) -> None:
        """Add a single canonical problem to the index."""
        if problem.benchmark not in self._problems:
            self._problems[problem.benchmark] = []
        self._problems[problem.benchmark].append(problem)

        # Add to firewall
        self._firewall.index_canonical(
            problem_id=problem.problem_id,
            text=problem.text,
            domain=problem.domain,
            answer=problem.answer if problem.domain == "math" else None,
            test_cases=problem.test_cases if problem.domain == "code" else None,
        )

    def add_problems(self, problems: list[BenchmarkProblem]) -> None:
        """Add multiple canonical problems to the index."""
        for problem in problems:
            self.add_problem(problem)
        self._loaded = True

    def check_contamination(self, synthetic: BenchmarkProblem) -> ContaminationResult:
        """Check if a synthetic problem overlaps with canonical."""
        return self._firewall.check(
            synthetic_text=synthetic.text,
            domain=synthetic.domain,
            answer=synthetic.answer if synthetic.domain == "math" else None,
            test_cases=synthetic.test_cases if synthetic.domain == "code" else None,
        )

    def get_stats(self) -> dict:
        """Get contamination checking statistics."""
        stats = self._firewall.get_stats()
        stats["problems_indexed"] = sum(len(p) for p in self._problems.values())
        stats["benchmarks_indexed"] = list(self._problems.keys())
        return stats


# Import the actual loader implementation
from .loader import (
    BenchmarkLoader,
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    load_benchmark,
)


__all__ = [
    # Benchmark types
    "BenchmarkType",
    "BenchmarkProblem",
    "BENCHMARK_DOMAINS",
    # Index
    "CanonicalIndex",
    # Loading
    "BenchmarkLoader",
    "BenchmarkConfig",
    "BENCHMARK_CONFIGS",
    "load_benchmark",
    # Contamination
    "ContaminationConfig",
    "ContaminationFirewall",
    "ContaminationResult",
    "RejectionReason",
    "TokenOverlapChecker",
    "SemanticSimilarityChecker",
    "MathStructuralChecker",
    "CodeStructuralChecker",
    "check_contamination",
    "compute_overlap_matrix",
    "simple_tokenize",
    "extract_ngrams",
]
