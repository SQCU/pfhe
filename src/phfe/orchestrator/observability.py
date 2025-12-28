"""
Observability Infrastructure for Tutor Calls and Workflows.

Provides structured logging, tracing, and persistence for debugging
and analysis of tutor orchestration workflows.

Key concepts:
    - SubagentLog: A single API call to a tutor model
    - WorkflowTrace: A complete workflow (generate -> verify -> filter -> persist)
    - TraceStore: Persistent storage for traces (JSONL files)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TutorType(str, Enum):
    """The tutor models in the ensemble."""

    DEEPSEEK_R1 = "deepseek_r1"
    KIMI_K2 = "kimi_k2"
    QWEN_72B = "qwen_72b"
    GPT_4O = "gpt_4o"
    CLAUDE_SONNET = "claude_sonnet"
    GEMMA_27B = "gemma_27b"
    MISTRAL_LARGE = "mistral_large"
    LLAMA_70B = "llama_70b"


class WorkflowStatus(str, Enum):
    """Status of a workflow."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # Contamination firewall rejected


@dataclass
class SparseLogits:
    """
    Sparse representation of logits at a single position.
    Only stores top-p or top-k tokens to save space.
    """

    token_ids: list[int]
    logit_values: list[float]
    coverage: float  # What probability mass is covered
    method: str  # "top_p" or "top_k"
    threshold: float  # The p or k value used


@dataclass
class SubagentLog:
    """
    A single tutor invocation with full context for debugging.

    Captures everything needed to reproduce and diagnose issues:
    - What was sent (input)
    - What came back (output)
    - How long it took
    - What model was used
    - Logits for distillation
    """

    # Identity
    log_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_id: str = ""
    tutor_type: TutorType = TutorType.GPT_4O

    # Model info
    model: str = ""
    system_prompt_hash: str = ""

    # Request
    input_payload: dict[str, Any] = field(default_factory=dict)
    user_message: str = ""

    # Response
    output_raw: str = ""
    output_parsed: Optional[dict[str, Any]] = None
    output_tokens: list[int] = field(default_factory=list)
    output_logits: list[SparseLogits] = field(default_factory=list)
    parse_success: bool = False
    parse_errors: list[str] = field(default_factory=list)

    # Metrics
    latency_ms: int = 0
    input_tokens: int = 0
    output_token_count: int = 0

    # Timing
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["tutor_type"] = self.tutor_type.value
        # Convert SparseLogits to dicts
        d["output_logits"] = [asdict(l) for l in self.output_logits]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubagentLog:
        """Reconstruct from dictionary."""
        d = d.copy()
        d["tutor_type"] = TutorType(d["tutor_type"])
        # Reconstruct SparseLogits
        d["output_logits"] = [SparseLogits(**l) for l in d.get("output_logits", [])]
        return cls(**d)

    @property
    def cost_usd(self) -> float:
        """Estimate cost in USD based on model and tokens."""
        # Approximate prices per 1M tokens (input, output) as of late 2024
        prices = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
            "claude-sonnet-4-20250514": (3.00, 15.00),
        }
        # Open-weight models are effectively free (just compute)
        input_price, output_price = prices.get(self.model, (0.0, 0.0))
        return (
            self.input_tokens * input_price + self.output_token_count * output_price
        ) / 1_000_000


@dataclass
class WorkflowTrace:
    """
    A complete workflow trace for synthetic generation or answer key collection.

    Links together all tutor calls that contributed to a single
    generation attempt.
    """

    # Identity
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: WorkflowStatus = WorkflowStatus.IN_PROGRESS
    workflow_type: str = ""  # "synthetic_generation", "answer_key", "icr_transform"

    # Context
    benchmark: str = ""
    problem_id: str = ""

    # Tutor calls (in order)
    tutor_logs: list[SubagentLog] = field(default_factory=list)

    # Outputs
    final_output: Optional[dict[str, Any]] = None

    # Failure info
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None

    # Contamination check
    contamination_check_passed: Optional[bool] = None
    contamination_rejection_reason: Optional[str] = None

    # Metrics
    total_latency_ms: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    def add_tutor_log(self, log: SubagentLog) -> None:
        """Record a tutor call."""
        log.workflow_id = self.workflow_id
        self.tutor_logs.append(log)
        self.total_latency_ms += log.latency_ms

    def complete(self, output: Optional[dict[str, Any]] = None) -> None:
        """Mark workflow as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.final_output = output
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, stage: str, reason: str) -> None:
        """Mark workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.failure_stage = stage
        self.failure_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def reject(self, reason: str) -> None:
        """Mark workflow as rejected by contamination firewall."""
        self.status = WorkflowStatus.REJECTED
        self.contamination_check_passed = False
        self.contamination_rejection_reason = reason
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def total_cost_usd(self) -> float:
        """Total cost of all tutor calls."""
        return sum(log.cost_usd for log in self.tutor_logs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "workflow_type": self.workflow_type,
            "benchmark": self.benchmark,
            "problem_id": self.problem_id,
            "tutor_logs": [log.to_dict() for log in self.tutor_logs],
            "final_output": self.final_output,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "contamination_check_passed": self.contamination_check_passed,
            "contamination_rejection_reason": self.contamination_rejection_reason,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowTrace:
        """Reconstruct from dictionary."""
        trace = cls(
            workflow_id=d["workflow_id"],
            status=WorkflowStatus(d["status"]),
            workflow_type=d.get("workflow_type", ""),
            benchmark=d.get("benchmark", ""),
            problem_id=d.get("problem_id", ""),
            final_output=d.get("final_output"),
            failure_stage=d.get("failure_stage"),
            failure_reason=d.get("failure_reason"),
            contamination_check_passed=d.get("contamination_check_passed"),
            contamination_rejection_reason=d.get("contamination_rejection_reason"),
            total_latency_ms=d.get("total_latency_ms", 0),
            started_at=d["started_at"],
            completed_at=d.get("completed_at"),
        )
        if d.get("tutor_logs"):
            trace.tutor_logs = [SubagentLog.from_dict(l) for l in d["tutor_logs"]]
        return trace


class TraceStore:
    """
    Persistent storage for workflow traces.

    Writes to JSONL files for easy streaming/analysis.
    One file per day to keep file sizes manageable.
    """

    def __init__(self, base_dir: Path | str = "traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}

    def _current_file(self) -> Path:
        """Get today's trace file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.base_dir / f"traces_{date_str}.jsonl"

    def save(self, trace: WorkflowTrace) -> None:
        """Append trace to today's file."""
        filepath = self._current_file()
        with open(filepath, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")
        self._index[trace.workflow_id] = filepath

    def load(self, workflow_id: str) -> Optional[WorkflowTrace]:
        """Load a specific trace by ID."""
        if workflow_id in self._index:
            return self._search_file(self._index[workflow_id], workflow_id)

        for filepath in sorted(self.base_dir.glob("traces_*.jsonl"), reverse=True):
            trace = self._search_file(filepath, workflow_id)
            if trace:
                self._index[workflow_id] = filepath
                return trace
        return None

    def _search_file(
        self, filepath: Path, workflow_id: str
    ) -> Optional[WorkflowTrace]:
        """Search a single file for a workflow ID."""
        if not filepath.exists():
            return None
        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                if data["workflow_id"] == workflow_id:
                    return WorkflowTrace.from_dict(data)
        return None

    def list_recent(self, limit: int = 100) -> list[WorkflowTrace]:
        """List recent traces (most recent first)."""
        traces = []
        for filepath in sorted(self.base_dir.glob("traces_*.jsonl"), reverse=True):
            with open(filepath) as f:
                for line in f:
                    traces.append(WorkflowTrace.from_dict(json.loads(line)))
                    if len(traces) >= limit:
                        return traces
        return traces

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all traces."""
        stats: dict[str, Any] = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0,
            "by_benchmark": {},
            "by_workflow_type": {},
        }

        for filepath in self.base_dir.glob("traces_*.jsonl"):
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    stats["total"] += 1
                    status = data["status"]
                    if status == "completed":
                        stats["completed"] += 1
                    elif status == "failed":
                        stats["failed"] += 1
                    elif status == "rejected":
                        stats["rejected"] += 1

                    stats["total_cost_usd"] += data.get("total_cost_usd", 0)
                    stats["total_latency_ms"] += data.get("total_latency_ms", 0)

                    bench = data.get("benchmark", "unknown")
                    stats["by_benchmark"][bench] = (
                        stats["by_benchmark"].get(bench, 0) + 1
                    )

                    wtype = data.get("workflow_type", "unknown")
                    stats["by_workflow_type"][wtype] = (
                        stats["by_workflow_type"].get(wtype, 0) + 1
                    )

        if stats["total"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total"]
            stats["avg_cost_usd"] = stats["total_cost_usd"] / stats["total"]
            stats["success_rate"] = stats["completed"] / stats["total"]

        return stats


def hash_prompt(prompt: str) -> str:
    """Create a short hash of a system prompt for logging."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


class Timer:
    """Context manager for timing operations."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.elapsed_ms: int = 0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)
