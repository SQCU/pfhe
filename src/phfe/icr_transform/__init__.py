"""
ICR (In-Context Retrieval) Transform System

Transforms standard benchmarks into ICR format where:
- Questions are paired with method library contexts
- Correct answers depend on applying methods from context
- The same problem structure can be solved by referencing worked examples

Transformation types:
- worked_examples: Provide method descriptions and worked examples (primary)
- fictional_context: Generate alternate-world contexts (future)
- explicit_setup: Make implicit commonsense explicit (future)
- document_grounding: Ground in synthetic documents (future)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib

from .method_libraries import (
    METHOD_LIBRARIES,
    get_method_library,
    list_available_libraries,
)


class TransformationType(str, Enum):
    """Types of ICR transformations."""

    WORKED_EXAMPLES = "worked_examples"
    FICTIONAL_CONTEXT = "fictional_context"
    EXPLICIT_SETUP = "explicit_setup"
    DOCUMENT_GROUNDING = "document_grounding"


@dataclass
class ICRInstance:
    """A single ICR-transformed benchmark instance."""

    instance_id: str
    source_benchmark: str
    source_instance_id: str

    # The ICR instance
    context: str
    question: str
    full_prompt: str  # context + question combined
    options: Optional[list[str]] = None
    correct_answer: str = ""
    reasoning_trace: str = ""

    # Training data (filled by tutor)
    context_token_ids: list[int] = field(default_factory=list)
    reasoning_token_ids: list[int] = field(default_factory=list)
    tutor_model: str = ""

    # Metadata
    transformation_type: TransformationType = TransformationType.WORKED_EXAMPLES
    source_corpus: Optional[str] = None
    difficulty_estimate: float = 0.5

    # Verification
    verified: bool = False
    verification_method: str = ""


@dataclass
class ICRBatch:
    """A batch of ICR-transformed instances."""

    benchmark: str
    transformation_type: TransformationType
    instances: list[ICRInstance]
    method_library_hash: str  # For versioning

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)


class ICRTransformer:
    """Transform benchmarks to ICR format by prepending method libraries."""

    def __init__(
        self,
        default_transformation: TransformationType = TransformationType.WORKED_EXAMPLES,
    ):
        self.default_transformation = default_transformation
        self._library_cache: dict[str, str] = {}

    def get_method_library(self, benchmark: str) -> str:
        """Get the method library for a benchmark (cached)."""
        if benchmark not in self._library_cache:
            library = get_method_library(benchmark)
            self._library_cache[benchmark] = library
        return self._library_cache[benchmark]

    def transform(
        self,
        problem_id: str,
        benchmark: str,
        question_text: str,
        answer: str,
        options: Optional[list[str]] = None,
        transformation_type: Optional[TransformationType] = None,
    ) -> ICRInstance:
        """
        Transform a single problem to ICR format.

        Args:
            problem_id: Unique identifier for the source problem
            benchmark: Benchmark name (e.g., "gsm1k", "arc", "mbpp")
            question_text: The original question/problem text
            answer: The correct answer
            options: Multiple choice options if applicable
            transformation_type: Type of transformation (defaults to WORKED_EXAMPLES)

        Returns:
            ICRInstance with method library context prepended
        """
        trans_type = transformation_type or self.default_transformation

        if trans_type != TransformationType.WORKED_EXAMPLES:
            raise NotImplementedError(
                f"Transformation type {trans_type} not yet implemented. "
                "Only WORKED_EXAMPLES is currently supported."
            )

        # Get the method library for this benchmark
        context = self.get_method_library(benchmark)

        if not context:
            # No method library for this benchmark - use question as-is
            full_prompt = question_text
        else:
            # The method libraries end with a marker like "[PROBLEM]" or "[PASSAGE]"
            # The question follows directly after
            full_prompt = context + question_text

        # Generate unique instance ID
        instance_id = self._generate_instance_id(
            benchmark, problem_id, trans_type.value
        )

        return ICRInstance(
            instance_id=instance_id,
            source_benchmark=benchmark,
            source_instance_id=problem_id,
            context=context,
            question=question_text,
            full_prompt=full_prompt,
            options=options,
            correct_answer=answer,
            transformation_type=trans_type,
        )

    def transform_batch(
        self,
        problems: list[dict],
        benchmark: str,
        transformation_type: Optional[TransformationType] = None,
    ) -> ICRBatch:
        """
        Transform a batch of problems.

        Args:
            problems: List of dicts with keys: problem_id, text, answer, options (optional)
            benchmark: Benchmark name
            transformation_type: Type of transformation

        Returns:
            ICRBatch containing all transformed instances
        """
        trans_type = transformation_type or self.default_transformation

        instances = []
        for prob in problems:
            instance = self.transform(
                problem_id=prob["problem_id"],
                benchmark=benchmark,
                question_text=prob["text"],
                answer=prob["answer"],
                options=prob.get("options"),
                transformation_type=trans_type,
            )
            instances.append(instance)

        # Hash the method library for versioning
        library = self.get_method_library(benchmark)
        library_hash = hashlib.sha256(library.encode()).hexdigest()[:16]

        return ICRBatch(
            benchmark=benchmark,
            transformation_type=trans_type,
            instances=instances,
            method_library_hash=library_hash,
        )

    def _generate_instance_id(
        self, benchmark: str, problem_id: str, transform_type: str
    ) -> str:
        """Generate a unique instance ID."""
        return f"icr_{benchmark}_{problem_id}_{transform_type}"

    @staticmethod
    def list_supported_benchmarks() -> list[str]:
        """List benchmarks with method library support."""
        return list_available_libraries()


def transform_problem(
    benchmark: str,
    problem_id: str,
    question_text: str,
    answer: str,
    options: Optional[list[str]] = None,
) -> ICRInstance:
    """
    Convenience function to transform a single problem.

    Example:
        >>> instance = transform_problem(
        ...     benchmark="gsm1k",
        ...     problem_id="train_001",
        ...     question_text="Alice has 5 apples. Bob gives her 3 more. How many does she have?",
        ...     answer="8",
        ... )
        >>> print(instance.full_prompt[:50])
        '[METHOD LIBRARY: Arithmetic Word Problems]...'
    """
    transformer = ICRTransformer()
    return transformer.transform(
        problem_id=problem_id,
        benchmark=benchmark,
        question_text=question_text,
        answer=answer,
        options=options,
    )


__all__ = [
    # Types
    "TransformationType",
    "ICRInstance",
    "ICRBatch",
    # Transformer
    "ICRTransformer",
    # Convenience
    "transform_problem",
    # Method libraries
    "get_method_library",
    "list_available_libraries",
    "METHOD_LIBRARIES",
]
