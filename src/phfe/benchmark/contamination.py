"""
Contamination Firewall - Verify synthetic problems are distinct from canonical benchmarks.

Three-layer checking system:
1. Token-level overlap: n-gram Jaccard similarity
2. Semantic similarity: Embedding cosine similarity
3. Structural similarity: Domain-specific pattern matching

Every synthetic training problem must pass ALL checks before inclusion.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np


class RejectionReason(str, Enum):
    """Reasons a synthetic problem may be rejected."""

    PASSED = "passed"
    TOKEN_OVERLAP = "token_overlap"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    MATH_STRUCTURE = "math_structure"
    CODE_STRUCTURE = "code_structure"
    EXACT_MATCH = "exact_match"


@dataclass
class ContaminationResult:
    """Result of a contamination check."""

    is_safe: bool
    rejection_reason: RejectionReason = RejectionReason.PASSED

    # Detailed metrics
    token_overlap_score: float = 0.0
    semantic_similarity_score: float = 0.0
    structural_match: bool = False

    # Which canonical problem triggered rejection (if any)
    matched_canonical_id: Optional[str] = None
    matched_canonical_text: Optional[str] = None

    # Check execution info
    checks_run: list[str] = field(default_factory=list)


@dataclass
class ContaminationConfig:
    """Configuration for contamination checking thresholds."""

    # Token overlap settings
    ngram_n: int = 5
    token_overlap_threshold: float = 0.3  # Reject if >30% n-gram overlap

    # Semantic similarity settings
    semantic_threshold: float = 0.85  # Reject if cosine sim > 0.85
    embedding_model: str = "all-MiniLM-L6-v2"

    # Structural settings (domain-specific)
    require_structural_check: bool = True

    # Performance settings
    use_blocking: bool = True  # Use LSH/blocking to speed up comparisons
    block_size: int = 1000


# =============================================================================
# Tokenization and N-gram Utilities
# =============================================================================


def simple_tokenize(text: str) -> list[str]:
    """
    Simple word tokenization for n-gram extraction.

    Lowercases, removes punctuation except for numbers.
    """
    # Lowercase
    text = text.lower()

    # Keep alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Split on whitespace
    tokens = text.split()

    return tokens


def extract_ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Extract n-grams from a token sequence."""
    if len(tokens) < n:
        return set()

    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap_score(
    synthetic_tokens: Sequence[str],
    canonical_tokens: Sequence[str],
    n: int = 5,
) -> float:
    """
    Compute n-gram overlap between synthetic and canonical text.

    Returns the fraction of synthetic n-grams that appear in canonical.
    """
    synthetic_ngrams = extract_ngrams(synthetic_tokens, n)
    canonical_ngrams = extract_ngrams(canonical_tokens, n)

    if len(synthetic_ngrams) == 0:
        return 0.0

    overlap = len(synthetic_ngrams & canonical_ngrams)
    return overlap / len(synthetic_ngrams)


# =============================================================================
# Token-Level Overlap Checker
# =============================================================================


class TokenOverlapChecker:
    """
    Check for n-gram overlap between synthetic and canonical problems.

    Rejects if synthetic shares too many n-grams with any canonical problem.
    """

    def __init__(
        self,
        n: int = 5,
        threshold: float = 0.3,
        tokenizer: Callable[[str], list[str]] = simple_tokenize,
    ):
        self.n = n
        self.threshold = threshold
        self.tokenizer = tokenizer

        # Pre-computed canonical data
        self._canonical_ngrams: dict[str, set[tuple[str, ...]]] = {}
        self._canonical_texts: dict[str, str] = {}

    def index_canonical(self, problem_id: str, text: str) -> None:
        """Add a canonical problem to the index."""
        tokens = self.tokenizer(text)
        ngrams = extract_ngrams(tokens, self.n)
        self._canonical_ngrams[problem_id] = ngrams
        self._canonical_texts[problem_id] = text

    def index_canonical_batch(self, problems: list[tuple[str, str]]) -> None:
        """Add multiple canonical problems to the index."""
        for problem_id, text in problems:
            self.index_canonical(problem_id, text)

    def check(self, synthetic_text: str) -> tuple[bool, float, Optional[str]]:
        """
        Check if synthetic text overlaps with any canonical problem.

        Returns:
            (is_safe, max_overlap_score, matched_problem_id)
        """
        synthetic_tokens = self.tokenizer(synthetic_text)
        synthetic_ngrams = extract_ngrams(synthetic_tokens, self.n)

        if len(synthetic_ngrams) == 0:
            return True, 0.0, None

        max_overlap = 0.0
        max_overlap_id: Optional[str] = None

        for problem_id, canonical_ngrams in self._canonical_ngrams.items():
            if len(canonical_ngrams) == 0:
                continue

            overlap = len(synthetic_ngrams & canonical_ngrams)
            score = overlap / len(synthetic_ngrams)

            if score > max_overlap:
                max_overlap = score
                max_overlap_id = problem_id

            # Early exit if already over threshold
            if score > self.threshold:
                return False, score, problem_id

        is_safe = max_overlap <= self.threshold
        return is_safe, max_overlap, max_overlap_id

    def clear(self) -> None:
        """Clear the canonical index."""
        self._canonical_ngrams.clear()
        self._canonical_texts.clear()


# =============================================================================
# Semantic Similarity Checker
# =============================================================================


class SemanticSimilarityChecker:
    """
    Check for semantic similarity using embeddings.

    Uses sentence-transformers for efficient similarity computation.
    Rejects if cosine similarity exceeds threshold.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = threshold
        self.model_name = model_name

        # Lazy-loaded model
        self._model = None
        self._canonical_embeddings: Optional[np.ndarray] = None
        self._canonical_ids: list[str] = []
        self._canonical_texts: dict[str, str] = {}

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for semantic similarity. "
                "Install with: uv add sentence-transformers"
            ) from e

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def index_canonical(self, problem_id: str, text: str) -> None:
        """Add a canonical problem to the index."""
        self._canonical_ids.append(problem_id)
        self._canonical_texts[problem_id] = text
        # Invalidate cached embeddings
        self._canonical_embeddings = None

    def index_canonical_batch(self, problems: list[tuple[str, str]]) -> None:
        """Add multiple canonical problems and compute embeddings."""
        for problem_id, text in problems:
            self._canonical_ids.append(problem_id)
            self._canonical_texts[problem_id] = text

        # Compute embeddings for all at once
        texts = [self._canonical_texts[pid] for pid in self._canonical_ids]
        self._canonical_embeddings = self._embed(texts)

    def _ensure_embeddings(self) -> None:
        """Ensure canonical embeddings are computed."""
        if self._canonical_embeddings is None and self._canonical_ids:
            texts = [self._canonical_texts[pid] for pid in self._canonical_ids]
            self._canonical_embeddings = self._embed(texts)

    def check(self, synthetic_text: str) -> tuple[bool, float, Optional[str]]:
        """
        Check if synthetic text is too similar to any canonical problem.

        Returns:
            (is_safe, max_similarity_score, matched_problem_id)
        """
        if not self._canonical_ids:
            return True, 0.0, None

        self._ensure_embeddings()

        # Embed synthetic text
        synthetic_embedding = self._embed([synthetic_text])[0]

        # Compute similarities (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(self._canonical_embeddings, synthetic_embedding)

        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])
        max_id = self._canonical_ids[max_idx]

        is_safe = max_similarity <= self.threshold
        return is_safe, max_similarity, max_id if not is_safe else None

    def clear(self) -> None:
        """Clear the canonical index."""
        self._canonical_embeddings = None
        self._canonical_ids.clear()
        self._canonical_texts.clear()


# =============================================================================
# Structural Similarity Checkers (Domain-Specific)
# =============================================================================


class StructuralChecker(ABC):
    """Abstract base class for domain-specific structural checks."""

    @abstractmethod
    def check(
        self, synthetic: dict, canonical_set: list[dict]
    ) -> tuple[bool, Optional[str]]:
        """
        Check if synthetic has same structure as any canonical.

        Returns:
            (is_safe, matched_problem_id)
        """
        ...


class MathStructuralChecker(StructuralChecker):
    """
    Check for structural similarity in math problems.

    Rejects if synthetic has same numbers AND same operations AND same answer.
    """

    def __init__(self):
        self._canonical_signatures: dict[str, tuple] = {}

    @staticmethod
    def extract_numbers(text: str) -> frozenset[str]:
        """Extract all numbers from text (absolute values only for comparison)."""
        # Match integers and decimals, strip sign for structural comparison
        # We compare absolute values since "loses $5" and "gains $5" are structurally similar
        raw_numbers = re.findall(r"-?\d+\.?\d*", text)
        # Normalize: strip leading minus, strip trailing zeros after decimal
        normalized = set()
        for n in raw_numbers:
            n = n.lstrip("-")
            if "." in n:
                n = n.rstrip("0").rstrip(".")
            if n:
                normalized.add(n)
        return frozenset(normalized)

    @staticmethod
    def extract_operations(text: str) -> frozenset[str]:
        """Extract mathematical operations from text."""
        # Look for operation keywords and symbols
        ops = set()
        text_lower = text.lower()

        # Addition patterns
        add_patterns = ["+", "plus", "add", "gives", "gave", "more", "total", "sum", "combine"]
        if any(p in text_lower for p in add_patterns):
            ops.add("add")

        # Subtraction patterns
        sub_patterns = ["-", "minus", "subtract", "takes", "took", "left", "remain", "fewer", "less", "lose", "lost"]
        if any(p in text_lower for p in sub_patterns):
            ops.add("subtract")

        # Multiplication patterns
        mul_patterns = ["*", "×", "times", "multipl", "each", "per", "every"]
        if any(p in text_lower for p in mul_patterns):
            ops.add("multiply")

        # Division patterns
        div_patterns = ["/", "÷", "divide", "split", "share", "equally", "half", "third", "quarter"]
        if any(p in text_lower for p in div_patterns):
            ops.add("divide")

        # Percentage patterns
        pct_patterns = ["%", "percent"]
        if any(p in text_lower for p in pct_patterns):
            ops.add("percent")

        return frozenset(ops)

    def compute_signature(self, text: str, answer: str) -> tuple:
        """Compute structural signature for a math problem."""
        numbers = self.extract_numbers(text)
        operations = self.extract_operations(text)
        return (numbers, operations, answer.strip())

    def index_canonical(self, problem_id: str, text: str, answer: str) -> None:
        """Add a canonical math problem to the index."""
        sig = self.compute_signature(text, answer)
        self._canonical_signatures[problem_id] = sig

    def check(
        self, synthetic: dict, canonical_set: list[dict] | None = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if synthetic math problem matches any canonical.

        Args:
            synthetic: Dict with 'text' and 'answer' keys
            canonical_set: Ignored if index is pre-built

        Returns:
            (is_safe, matched_problem_id)
        """
        synthetic_sig = self.compute_signature(
            synthetic.get("text", ""),
            synthetic.get("answer", ""),
        )

        for problem_id, canonical_sig in self._canonical_signatures.items():
            if synthetic_sig == canonical_sig:
                return False, problem_id

        return True, None

    def clear(self) -> None:
        """Clear the canonical index."""
        self._canonical_signatures.clear()


class CodeStructuralChecker(StructuralChecker):
    """
    Check for structural similarity in code problems.

    Rejects if synthetic has same function signature AND same test cases.
    """

    def __init__(self):
        self._canonical_signatures: dict[str, tuple] = {}

    @staticmethod
    def extract_function_name(text: str) -> Optional[str]:
        """Extract function name from problem text or code."""
        # Look for "def function_name" pattern (actual code)
        match = re.search(r"def\s+(\w+)\s*\(", text)
        if match:
            return match.group(1).lower()

        # Look for "called X", "named X" patterns (most specific)
        match = re.search(r"(?:called|named)\s+[`'\"]?(\w+)[`'\"]?", text, re.I)
        if match:
            return match.group(1).lower()

        # Look for "function X", "implement X", "write X" where X is the function name
        # But skip articles like "a", "an", "the"
        match = re.search(r"(?:function|implement|create|write)\s+(?:a\s+)?(?:function\s+)?[`'\"]?([a-z_]\w*)[`'\"]?", text, re.I)
        if match:
            name = match.group(1).lower()
            # Skip common articles/words that aren't function names
            if name not in {"a", "an", "the", "function", "that", "which", "to"}:
                return name

        return None

    @staticmethod
    def extract_test_inputs(test_cases: list) -> frozenset[str]:
        """Extract unique test inputs."""
        inputs = set()
        for test in test_cases:
            if isinstance(test, (list, tuple)) and len(test) >= 1:
                inputs.add(str(test[0]))
            elif isinstance(test, dict) and "input" in test:
                inputs.add(str(test["input"]))
        return frozenset(inputs)

    def compute_signature(
        self, text: str, test_cases: list
    ) -> tuple[Optional[str], frozenset[str]]:
        """Compute structural signature for a code problem."""
        func_name = self.extract_function_name(text)
        test_inputs = self.extract_test_inputs(test_cases)
        return (func_name, test_inputs)

    def index_canonical(
        self, problem_id: str, text: str, test_cases: list
    ) -> None:
        """Add a canonical code problem to the index."""
        sig = self.compute_signature(text, test_cases)
        self._canonical_signatures[problem_id] = sig

    def check(
        self, synthetic: dict, canonical_set: list[dict] | None = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if synthetic code problem matches any canonical.

        Args:
            synthetic: Dict with 'text' and 'test_cases' keys

        Returns:
            (is_safe, matched_problem_id)
        """
        synthetic_sig = self.compute_signature(
            synthetic.get("text", ""),
            synthetic.get("test_cases", []),
        )

        for problem_id, canonical_sig in self._canonical_signatures.items():
            if synthetic_sig == canonical_sig:
                return False, problem_id

        return True, None

    def clear(self) -> None:
        """Clear the canonical index."""
        self._canonical_signatures.clear()


# =============================================================================
# Combined Contamination Firewall
# =============================================================================


class ContaminationFirewall:
    """
    Combined contamination checking system.

    Runs all checks (token overlap, semantic similarity, structural)
    and rejects synthetic problems that fail any check.
    """

    def __init__(self, config: Optional[ContaminationConfig] = None):
        self.config = config or ContaminationConfig()

        # Initialize checkers
        self.token_checker = TokenOverlapChecker(
            n=self.config.ngram_n,
            threshold=self.config.token_overlap_threshold,
        )
        self.semantic_checker = SemanticSimilarityChecker(
            threshold=self.config.semantic_threshold,
            model_name=self.config.embedding_model,
        )

        # Domain-specific checkers
        self.math_checker = MathStructuralChecker()
        self.code_checker = CodeStructuralChecker()

        # Statistics
        self._stats = {
            "total_checked": 0,
            "passed": 0,
            "rejected_token": 0,
            "rejected_semantic": 0,
            "rejected_structure": 0,
        }

    def index_canonical(
        self,
        problem_id: str,
        text: str,
        domain: Optional[str] = None,
        answer: Optional[str] = None,
        test_cases: Optional[list] = None,
    ) -> None:
        """
        Add a canonical problem to all relevant indices.

        Args:
            problem_id: Unique identifier for the problem
            text: Problem text
            domain: "math", "code", or None for general
            answer: For math problems, the correct answer
            test_cases: For code problems, list of test cases
        """
        # Token overlap index
        self.token_checker.index_canonical(problem_id, text)

        # Semantic similarity index
        self.semantic_checker.index_canonical(problem_id, text)

        # Domain-specific indices
        if domain == "math" and answer is not None:
            self.math_checker.index_canonical(problem_id, text, answer)
        elif domain == "code" and test_cases is not None:
            self.code_checker.index_canonical(problem_id, text, test_cases)

    def index_canonical_batch(
        self,
        problems: list[dict],
        domain: Optional[str] = None,
    ) -> None:
        """
        Index multiple canonical problems at once.

        Args:
            problems: List of dicts with 'id', 'text', and optionally 'answer'/'test_cases'
            domain: Domain for all problems in batch
        """
        # Batch index for efficiency
        text_pairs = [(p["id"], p["text"]) for p in problems]
        self.token_checker.index_canonical_batch(text_pairs)
        self.semantic_checker.index_canonical_batch(text_pairs)

        # Domain-specific indexing
        for p in problems:
            if domain == "math" and "answer" in p:
                self.math_checker.index_canonical(p["id"], p["text"], p["answer"])
            elif domain == "code" and "test_cases" in p:
                self.code_checker.index_canonical(p["id"], p["text"], p["test_cases"])

    def check(
        self,
        synthetic_text: str,
        domain: Optional[str] = None,
        answer: Optional[str] = None,
        test_cases: Optional[list] = None,
    ) -> ContaminationResult:
        """
        Run all contamination checks on a synthetic problem.

        Args:
            synthetic_text: The synthetic problem text
            domain: "math", "code", or None
            answer: For math problems
            test_cases: For code problems

        Returns:
            ContaminationResult with all check details
        """
        self._stats["total_checked"] += 1
        checks_run = []

        # Check 1: Token overlap
        token_safe, token_score, token_match = self.token_checker.check(synthetic_text)
        checks_run.append("token_overlap")

        if not token_safe:
            self._stats["rejected_token"] += 1
            return ContaminationResult(
                is_safe=False,
                rejection_reason=RejectionReason.TOKEN_OVERLAP,
                token_overlap_score=token_score,
                matched_canonical_id=token_match,
                matched_canonical_text=self.token_checker._canonical_texts.get(token_match),
                checks_run=checks_run,
            )

        # Check 2: Semantic similarity
        semantic_safe, semantic_score, semantic_match = self.semantic_checker.check(
            synthetic_text
        )
        checks_run.append("semantic_similarity")

        if not semantic_safe:
            self._stats["rejected_semantic"] += 1
            return ContaminationResult(
                is_safe=False,
                rejection_reason=RejectionReason.SEMANTIC_SIMILARITY,
                token_overlap_score=token_score,
                semantic_similarity_score=semantic_score,
                matched_canonical_id=semantic_match,
                matched_canonical_text=self.semantic_checker._canonical_texts.get(semantic_match),
                checks_run=checks_run,
            )

        # Check 3: Structural similarity (domain-specific)
        if self.config.require_structural_check and domain:
            if domain == "math" and answer is not None:
                struct_safe, struct_match = self.math_checker.check(
                    {"text": synthetic_text, "answer": answer}
                )
                checks_run.append("math_structure")

                if not struct_safe:
                    self._stats["rejected_structure"] += 1
                    return ContaminationResult(
                        is_safe=False,
                        rejection_reason=RejectionReason.MATH_STRUCTURE,
                        token_overlap_score=token_score,
                        semantic_similarity_score=semantic_score,
                        structural_match=True,
                        matched_canonical_id=struct_match,
                        checks_run=checks_run,
                    )

            elif domain == "code" and test_cases is not None:
                struct_safe, struct_match = self.code_checker.check(
                    {"text": synthetic_text, "test_cases": test_cases}
                )
                checks_run.append("code_structure")

                if not struct_safe:
                    self._stats["rejected_structure"] += 1
                    return ContaminationResult(
                        is_safe=False,
                        rejection_reason=RejectionReason.CODE_STRUCTURE,
                        token_overlap_score=token_score,
                        semantic_similarity_score=semantic_score,
                        structural_match=True,
                        matched_canonical_id=struct_match,
                        checks_run=checks_run,
                    )

        # All checks passed
        self._stats["passed"] += 1
        return ContaminationResult(
            is_safe=True,
            rejection_reason=RejectionReason.PASSED,
            token_overlap_score=token_score,
            semantic_similarity_score=semantic_score,
            structural_match=False,
            checks_run=checks_run,
        )

    def check_batch(
        self,
        synthetics: list[dict],
        domain: Optional[str] = None,
    ) -> list[ContaminationResult]:
        """
        Check multiple synthetic problems.

        Args:
            synthetics: List of dicts with 'text' and optionally 'answer'/'test_cases'
            domain: Domain for all problems

        Returns:
            List of ContaminationResult for each input
        """
        results = []
        for s in synthetics:
            result = self.check(
                synthetic_text=s.get("text", ""),
                domain=domain,
                answer=s.get("answer"),
                test_cases=s.get("test_cases"),
            )
            results.append(result)
        return results

    def get_stats(self) -> dict:
        """Get contamination checking statistics."""
        stats = dict(self._stats)
        if stats["total_checked"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_checked"]
            stats["rejection_rate"] = 1.0 - stats["pass_rate"]
        return stats

    def clear(self) -> None:
        """Clear all indices and reset statistics."""
        self.token_checker.clear()
        self.semantic_checker.clear()
        self.math_checker.clear()
        self.code_checker.clear()
        self._stats = {
            "total_checked": 0,
            "passed": 0,
            "rejected_token": 0,
            "rejected_semantic": 0,
            "rejected_structure": 0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def check_contamination(
    synthetic_text: str,
    canonical_texts: list[str],
    config: Optional[ContaminationConfig] = None,
) -> ContaminationResult:
    """
    One-shot contamination check.

    Convenience function for checking a single synthetic against a list of canonical texts.
    """
    firewall = ContaminationFirewall(config)

    # Index canonical texts
    for i, text in enumerate(canonical_texts):
        firewall.index_canonical(f"canonical_{i}", text)

    return firewall.check(synthetic_text)


def compute_overlap_matrix(
    texts_a: list[str],
    texts_b: list[str],
    n: int = 5,
) -> np.ndarray:
    """
    Compute pairwise n-gram overlap between two sets of texts.

    Returns:
        Matrix of shape (len(texts_a), len(texts_b)) with overlap scores.
    """
    # Tokenize and extract n-grams
    ngrams_a = [extract_ngrams(simple_tokenize(t), n) for t in texts_a]
    ngrams_b = [extract_ngrams(simple_tokenize(t), n) for t in texts_b]

    matrix = np.zeros((len(texts_a), len(texts_b)))

    for i, ng_a in enumerate(ngrams_a):
        if len(ng_a) == 0:
            continue
        for j, ng_b in enumerate(ngrams_b):
            overlap = len(ng_a & ng_b)
            matrix[i, j] = overlap / len(ng_a)

    return matrix
