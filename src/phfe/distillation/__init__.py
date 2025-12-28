"""
Offline Distillation Data Format & Training Protocol

Key insight: Tutor rollouts are expensive. Tutor prefills are cheap. Student rollouts are cheap.

Therefore:
- Generate tutor outputs OFFLINE (once, before training)
- Log everything: tokens, top-p logits, strings
- During training, only run student rollouts online
- Load pre-computed tutor logits, compute GKD gradient
- Mix with next-token regularization on diverse corpora
"""

from dataclasses import dataclass, field
from typing import Optional

from ..orchestrator.observability import SparseLogits


@dataclass
class DistillationExample:
    """
    A single example for distillation training.

    Contains the sequence, tutor logits, and metadata needed
    for GKD training.
    """

    # Identity
    example_id: str
    source: str  # "gsm1k_synth", "arc_synth", etc.

    # The sequence
    token_ids: list[int]
    text: str

    # Tutor logits (the distillation signal)
    logits: list[SparseLogits]
    # len(logits) == len(token_ids)
    # logits[i] is the distribution BEFORE generating token_ids[i]

    # Generation metadata
    tutor_model: str
    tutor_tokenizer: str
    generation_config: dict = field(default_factory=dict)

    # Task-specific fields
    context: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    reasoning_trace: Optional[str] = None
    task_type: Optional[str] = None

    # Quality signals
    verified: bool = False
    filter_scores: Optional[dict[str, float]] = None


@dataclass
class TrainingConfig:
    """Configuration for GKD training."""

    # Distillation settings
    distill_weight: float = 1.0
    regularization_weight: float = 1.0
    divergence: str = "jsd"  # "forward_kl", "reverse_kl", "jsd"
    jsd_beta: float = 0.1

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_steps: int = 100000

    # Data paths
    distillation_data_dir: str = "distillation_data"
    regularization_data_dir: str = "regularization_corpora"


class DistillationDataset:
    """Dataset for offline distillation training."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._examples: list[DistillationExample] = []

    def load(self) -> None:
        """Load distillation data from parquet files."""
        raise NotImplementedError("Implement data loading")

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> DistillationExample:
        return self._examples[idx]


def gkd_loss(
    student_logits: object,  # torch.Tensor
    tutor_sparse_logits: list[list[SparseLogits]],
    divergence: str = "jsd",
    jsd_beta: float = 0.1,
) -> object:  # torch.Tensor
    """
    Compute GKD loss between student logits and cached tutor logits.

    The tutor logits are sparse (only top-p tokens stored), so we need
    to handle the missing probability mass.
    """
    raise NotImplementedError("Implement GKD loss computation")


# Import cross-tokenizer components
from .cross_tokenizer import (
    SparseLogits as CrossTokenizerSparseLogits,
    TokenAlignment,
    DistillationTarget,
    CrossTokenizerAligner,
    LogitAggregator,
    VocabularyMapper,
    compute_gkd_loss,
)


__all__ = [
    # Examples and config
    "DistillationExample",
    "TrainingConfig",
    "DistillationDataset",
    "gkd_loss",
    # Cross-tokenizer
    "TokenAlignment",
    "DistillationTarget",
    "CrossTokenizerAligner",
    "LogitAggregator",
    "VocabularyMapper",
    "compute_gkd_loss",
]
