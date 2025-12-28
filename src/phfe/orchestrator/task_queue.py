"""
PHFE Task Queue Orchestrator

Uniform interface for task distribution to:
- Claude Code subagents (via tool use)
- External scripts (direct Python)
- vLLM/API inference servers

Key design: tool use makes everything easy — same claim/submit interface
regardless of worker type.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Iterator
import json
import uuid


class TaskStatus(str, Enum):
    """Status of a task in the queue."""
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class QueueType(str, Enum):
    """Types of task queues in PHFE pipeline."""
    GENERATE = "generate"  # Synthetic problem generation
    CONTAMINATE_CHECK = "contaminate_check"  # Firewall verification
    TUTOR_INFERENCE = "tutor_inference"  # Answer key generation
    VERIFY_ANSWER = "verify_answer"  # Answer verification
    ICR_TRANSFORM = "icr_transform"  # ICR augmentation


class ConcernLevel(str, Enum):
    """Levels for worker concerns."""
    INFO = "info"  # Just noting something
    REVIEW = "review"  # Needs orchestrator attention
    RETRY = "retry"  # Transient failure, retry suggested
    ERROR = "error"  # Task failed
    ESCALATE = "escalate"  # Context inadequate, pause queue


@dataclass
class WorkerConcern:
    """A concern raised by a worker during task processing."""
    level: ConcernLevel
    message: str
    suggestion: Optional[str] = None
    context_sample: Optional[str] = None  # What the worker saw (for debug)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "context_sample": self.context_sample,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerConcern":
        return cls(
            level=ConcernLevel(d["level"]),
            message=d["message"],
            suggestion=d.get("suggestion"),
            context_sample=d.get("context_sample"),
        )


@dataclass
class Task:
    """A single task in the queue."""
    task_id: str
    queue: QueueType
    input_data: dict
    status: TaskStatus = TaskStatus.PENDING

    # Claim info
    claimed_by: Optional[str] = None  # worker_type
    claimed_at: Optional[str] = None

    # Result
    result: Optional[dict] = None
    concerns: list[WorkerConcern] = field(default_factory=list)

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Debug mode
    context_presented: Optional[str] = None  # Full context for debugging

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "queue": self.queue.value,
            "input_data": self.input_data,
            "status": self.status.value,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "result": self.result,
            "concerns": [c.to_dict() for c in self.concerns],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "context_presented": self.context_presented,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        concerns = [WorkerConcern.from_dict(c) for c in d.get("concerns", [])]
        return cls(
            task_id=d["task_id"],
            queue=QueueType(d["queue"]),
            input_data=d["input_data"],
            status=TaskStatus(d["status"]),
            claimed_by=d.get("claimed_by"),
            claimed_at=d.get("claimed_at"),
            result=d.get("result"),
            concerns=concerns,
            created_at=d.get("created_at", datetime.now().isoformat()),
            completed_at=d.get("completed_at"),
            context_presented=d.get("context_presented"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class QueueStatus:
    """Summary status of a queue."""
    queue: QueueType
    total: int
    pending: int
    claimed: int
    in_progress: int
    completed: int
    failed: int
    needs_review: int

    def to_dict(self) -> dict:
        return {
            "queue": self.queue.value,
            "total": self.total,
            "pending": self.pending,
            "claimed": self.claimed,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "needs_review": self.needs_review,
        }


class TaskQueue:
    """
    Task queue with uniform interface for all worker types.

    Workers (Claude Code agents, scripts, vLLM) all use the same interface:
    - claim_task() -> get work
    - submit_result() -> deliver output
    - report_concern() -> flag issues

    Persistence: JSONL files for crash recovery.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize task queue.

        Args:
            storage_dir: Directory for persistence (optional)
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            (self.storage_dir / "queues").mkdir(exist_ok=True)
            (self.storage_dir / "completed").mkdir(exist_ok=True)
            (self.storage_dir / "concerns").mkdir(exist_ok=True)

        # In-memory queues
        self._queues: dict[QueueType, list[Task]] = {q: [] for q in QueueType}
        self._task_index: dict[str, Task] = {}  # task_id -> Task

    # =========================================================================
    # Core Operations
    # =========================================================================

    def enqueue(
        self,
        queue: QueueType,
        input_data: dict,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Add a task to a queue.

        Args:
            queue: Which queue to add to
            input_data: Task input payload
            task_id: Optional custom task ID

        Returns:
            task_id
        """
        task = Task(
            task_id=task_id or str(uuid.uuid4())[:12],
            queue=queue,
            input_data=input_data,
        )
        self._queues[queue].append(task)
        self._task_index[task.task_id] = task

        if self.storage_dir:
            self._persist_task(task)

        return task.task_id

    def enqueue_batch(
        self,
        queue: QueueType,
        inputs: list[dict],
    ) -> list[str]:
        """Enqueue multiple tasks at once."""
        return [self.enqueue(queue, inp) for inp in inputs]

    def claim_task(
        self,
        queue: QueueType,
        worker_type: str,
        worker_id: Optional[str] = None,
        debug_mode: bool = False,
    ) -> Optional[Task]:
        """
        Claim the next available task from a queue.

        Args:
            queue: Which queue to claim from
            worker_type: Type of worker claiming (for logging)
            worker_id: Optional unique worker ID
            debug_mode: If True, include full context for debugging

        Returns:
            Task if available, None if queue empty
        """
        for task in self._queues[queue]:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CLAIMED
                task.claimed_by = f"{worker_type}" + (f":{worker_id}" if worker_id else "")
                task.claimed_at = datetime.now().isoformat()

                if debug_mode:
                    task.context_presented = self._format_context(task)

                if self.storage_dir:
                    self._persist_task(task)

                return task

        return None

    def submit_result(
        self,
        task_id: str,
        result: dict,
        concerns: Optional[list[WorkerConcern]] = None,
    ) -> bool:
        """
        Submit result for a claimed task.

        Args:
            task_id: ID of the task
            result: Result payload
            concerns: Optional list of worker concerns

        Returns:
            True if successful
        """
        task = self._task_index.get(task_id)
        if task is None:
            return False

        task.result = result
        task.concerns = concerns or []
        task.completed_at = datetime.now().isoformat()

        # Determine final status based on concerns
        concern_levels = {c.level for c in task.concerns}

        if ConcernLevel.ERROR in concern_levels:
            task.status = TaskStatus.FAILED
        elif ConcernLevel.ESCALATE in concern_levels or ConcernLevel.REVIEW in concern_levels:
            task.status = TaskStatus.NEEDS_REVIEW
        else:
            task.status = TaskStatus.COMPLETED

        if self.storage_dir:
            self._persist_task(task)
            if task.concerns:
                self._persist_concerns(task)

        return True

    def update_progress(
        self,
        task_id: str,
        status: TaskStatus,
        notes: Optional[str] = None,
    ) -> bool:
        """Update task status without completing it."""
        task = self._task_index.get(task_id)
        if task is None:
            return False

        task.status = status
        if notes and task.result is None:
            task.result = {"progress_notes": notes}
        elif notes:
            task.result["progress_notes"] = notes

        if self.storage_dir:
            self._persist_task(task)

        return True

    def retry_task(self, task_id: str) -> bool:
        """Re-queue a failed task."""
        task = self._task_index.get(task_id)
        if task is None or task.status not in (TaskStatus.FAILED, TaskStatus.NEEDS_REVIEW):
            return False

        task.status = TaskStatus.PENDING
        task.claimed_by = None
        task.claimed_at = None
        task.result = None
        task.concerns = []

        if self.storage_dir:
            self._persist_task(task)

        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._task_index.get(task_id)

    def get_queue_status(self, queue: QueueType) -> QueueStatus:
        """Get status summary for a queue."""
        tasks = self._queues[queue]
        return QueueStatus(
            queue=queue,
            total=len(tasks),
            pending=sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            claimed=sum(1 for t in tasks if t.status == TaskStatus.CLAIMED),
            in_progress=sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
            completed=sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            failed=sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            needs_review=sum(1 for t in tasks if t.status == TaskStatus.NEEDS_REVIEW),
        )

    def get_all_status(self) -> dict[QueueType, QueueStatus]:
        """Get status for all queues."""
        return {q: self.get_queue_status(q) for q in QueueType}

    def get_tasks_needing_review(self) -> list[Task]:
        """Get all tasks that need review."""
        return [
            t for tasks in self._queues.values()
            for t in tasks
            if t.status == TaskStatus.NEEDS_REVIEW
        ]

    def get_failed_tasks(self) -> list[Task]:
        """Get all failed tasks."""
        return [
            t for tasks in self._queues.values()
            for t in tasks
            if t.status == TaskStatus.FAILED
        ]

    def list_tasks(
        self,
        queue: Optional[QueueType] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks with optional filters."""
        if queue:
            tasks = self._queues[queue]
        else:
            tasks = [t for q in self._queues.values() for t in q]

        if status:
            tasks = [t for t in tasks if t.status == status]

        return tasks[:limit]

    # =========================================================================
    # Reporting (for Claude Code orchestration)
    # =========================================================================

    def get_compact_report(self) -> str:
        """
        Get compact summary for Claude Code orchestration.

        This is what Claude Code sees to understand queue health.
        """
        lines = ["QUEUE STATUS:"]
        for queue in QueueType:
            s = self.get_queue_status(queue)
            status_str = f"{s.completed}/{s.total} done"
            if s.pending > 0:
                status_str += f", {s.pending} pending"
            if s.failed > 0:
                status_str += f", {s.failed} FAILED"
            if s.needs_review > 0:
                status_str += f", {s.needs_review} need review"
            lines.append(f"  {queue.value}: {status_str}")

        # Add alerts
        total_failed = sum(self.get_queue_status(q).failed for q in QueueType)
        total_review = sum(self.get_queue_status(q).needs_review for q in QueueType)

        if total_failed > 0 or total_review > 0:
            lines.append("")
            lines.append("ALERTS:")
            if total_failed > 0:
                lines.append(f"  {total_failed} tasks failed - run 'phfe task list --status failed'")
            if total_review > 0:
                lines.append(f"  {total_review} tasks need review - run 'phfe task review'")

        return "\n".join(lines)

    def get_concern_summary(self) -> dict[str, int]:
        """Get summary of concerns across all tasks."""
        summary: dict[str, int] = {}
        for tasks in self._queues.values():
            for task in tasks:
                for concern in task.concerns:
                    key = concern.level.value
                    summary[key] = summary.get(key, 0) + 1
        return summary

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist_task(self, task: Task) -> None:
        """Persist task to JSONL file."""
        if not self.storage_dir:
            return

        filepath = self.storage_dir / "queues" / f"{task.queue.value}.jsonl"
        # Append mode - simple but not efficient for updates
        # In production, would use proper database
        with open(filepath, "a") as f:
            f.write(task.to_json() + "\n")

    def _persist_concerns(self, task: Task) -> None:
        """Persist concerns to separate file for review."""
        if not self.storage_dir or not task.concerns:
            return

        filepath = self.storage_dir / "concerns" / "needs_review.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps({
                "task_id": task.task_id,
                "queue": task.queue.value,
                "concerns": [c.to_dict() for c in task.concerns],
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

    def load_from_storage(self) -> int:
        """
        Load tasks from storage (for crash recovery).

        Returns number of tasks loaded.
        """
        if not self.storage_dir:
            return 0

        count = 0
        for queue in QueueType:
            filepath = self.storage_dir / "queues" / f"{queue.value}.jsonl"
            if not filepath.exists():
                continue

            # Load all tasks, keeping only latest version by task_id
            task_versions: dict[str, Task] = {}
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        task = Task.from_dict(json.loads(line))
                        task_versions[task.task_id] = task

            # Add to queue
            self._queues[queue] = list(task_versions.values())
            for task in self._queues[queue]:
                self._task_index[task.task_id] = task
                count += 1

        return count

    # =========================================================================
    # Context Formatting (for debug mode)
    # =========================================================================

    def _format_context(self, task: Task) -> str:
        """Format task context for debug presentation."""
        lines = [
            f"=== TASK CONTEXT ===",
            f"Task ID: {task.task_id}",
            f"Queue: {task.queue.value}",
            f"",
            f"=== INPUT DATA ===",
            json.dumps(task.input_data, indent=2),
        ]
        return "\n".join(lines)


# =============================================================================
# Convenience Functions (for CLI / simple use)
# =============================================================================

_default_queue: Optional[TaskQueue] = None


def get_queue(storage_dir: Optional[Path] = None) -> TaskQueue:
    """Get or create the default task queue."""
    global _default_queue
    if _default_queue is None:
        _default_queue = TaskQueue(storage_dir)
    return _default_queue


def enqueue_generation(
    benchmark: str,
    difficulty: float = 0.5,
    count: int = 100,
) -> str:
    """Convenience: enqueue a generation task."""
    return get_queue().enqueue(
        QueueType.GENERATE,
        {
            "benchmark": benchmark,
            "difficulty": difficulty,
            "count": count,
        },
    )


def enqueue_contamination_check(
    problem: dict,
    domain: str,
) -> str:
    """Convenience: enqueue a contamination check task."""
    return get_queue().enqueue(
        QueueType.CONTAMINATE_CHECK,
        {
            "problem": problem,
            "domain": domain,
        },
    )


def enqueue_tutor_inference(
    problem_text: str,
    problem_id: str,
    tutor_model: str = "deepseek-r1",
    extract_logits: bool = True,
) -> str:
    """Convenience: enqueue a tutor inference task."""
    return get_queue().enqueue(
        QueueType.TUTOR_INFERENCE,
        {
            "problem_text": problem_text,
            "problem_id": problem_id,
            "tutor_model": tutor_model,
            "extract_logits": extract_logits,
        },
    )


__all__ = [
    # Enums
    "TaskStatus",
    "QueueType",
    "ConcernLevel",
    # Data classes
    "WorkerConcern",
    "Task",
    "QueueStatus",
    # Main class
    "TaskQueue",
    # Convenience
    "get_queue",
    "enqueue_generation",
    "enqueue_contamination_check",
    "enqueue_tutor_inference",
]
