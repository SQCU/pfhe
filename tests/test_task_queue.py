"""
Tests for PHFE Task Queue Orchestrator.

Tests the uniform interface for task distribution to Claude Code subagents,
external scripts, and vLLM/API inference servers.
"""

import json
import tempfile
from pathlib import Path

import pytest

from phfe.orchestrator import (
    TaskQueue,
    TaskStatus,
    QueueType,
    ConcernLevel,
    WorkerConcern,
    Task,
    QueueStatus,
    get_queue,
    enqueue_generation,
)


class TestTaskQueueBasics:
    """Basic task queue operations."""

    def test_create_queue_without_storage(self):
        """Queue works without persistence."""
        queue = TaskQueue()
        assert queue.storage_dir is None

    def test_create_queue_with_storage(self, tmp_path):
        """Queue creates storage directories."""
        queue = TaskQueue(storage_dir=tmp_path / "orchestrator")
        assert queue.storage_dir.exists()
        assert (queue.storage_dir / "queues").exists()
        assert (queue.storage_dir / "completed").exists()
        assert (queue.storage_dir / "concerns").exists()

    def test_enqueue_returns_task_id(self):
        """Enqueue returns a task ID."""
        queue = TaskQueue()
        task_id = queue.enqueue(
            QueueType.GENERATE,
            {"benchmark": "gsm1k", "difficulty": 0.5},
        )
        assert task_id is not None
        assert len(task_id) == 12  # UUID prefix

    def test_enqueue_with_custom_id(self):
        """Can specify custom task ID."""
        queue = TaskQueue()
        task_id = queue.enqueue(
            QueueType.GENERATE,
            {"benchmark": "gsm1k"},
            task_id="custom-123",
        )
        assert task_id == "custom-123"

    def test_enqueue_batch(self):
        """Batch enqueue creates multiple tasks."""
        queue = TaskQueue()
        inputs = [{"i": 0}, {"i": 1}, {"i": 2}]
        task_ids = queue.enqueue_batch(QueueType.GENERATE, inputs)
        assert len(task_ids) == 3
        assert len(set(task_ids)) == 3  # All unique


class TestClaimAndSubmit:
    """Task claiming and result submission."""

    def test_claim_returns_pending_task(self):
        """Claim returns a pending task."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {"test": "data"})

        task = queue.claim_task(QueueType.GENERATE, worker_type="test_worker")
        assert task is not None
        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == "test_worker"
        assert task.claimed_at is not None

    def test_claim_with_worker_id(self):
        """Claim includes worker ID if provided."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {"test": "data"})

        task = queue.claim_task(
            QueueType.GENERATE,
            worker_type="generator",
            worker_id="instance-1",
        )
        assert task.claimed_by == "generator:instance-1"

    def test_claim_empty_queue_returns_none(self):
        """Claim on empty queue returns None."""
        queue = TaskQueue()
        task = queue.claim_task(QueueType.GENERATE, worker_type="test")
        assert task is None

    def test_claim_skips_already_claimed(self):
        """Second claim gets a different task."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {"i": 0})
        queue.enqueue(QueueType.GENERATE, {"i": 1})

        task1 = queue.claim_task(QueueType.GENERATE, worker_type="w1")
        task2 = queue.claim_task(QueueType.GENERATE, worker_type="w2")

        assert task1.task_id != task2.task_id
        assert task1.input_data["i"] == 0
        assert task2.input_data["i"] == 1

    def test_submit_result_completes_task(self):
        """Submit marks task as completed."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {"test": "data"})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        success = queue.submit_result(task_id, {"result": "done"})
        assert success

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": "done"}
        assert task.completed_at is not None

    def test_submit_nonexistent_task_fails(self):
        """Submit to nonexistent task returns False."""
        queue = TaskQueue()
        success = queue.submit_result("nonexistent", {"result": "done"})
        assert not success


class TestWorkerConcerns:
    """Worker concern handling."""

    def test_info_concern_still_completes(self):
        """Info-level concern doesn't affect completion."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        concerns = [
            WorkerConcern(
                level=ConcernLevel.INFO,
                message="Just noting something",
            )
        ]
        queue.submit_result(task_id, {}, concerns=concerns)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED

    def test_review_concern_needs_review(self):
        """Review-level concern marks task for review."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        concerns = [
            WorkerConcern(
                level=ConcernLevel.REVIEW,
                message="Need orchestrator attention",
                suggestion="Check the input format",
            )
        ]
        queue.submit_result(task_id, {}, concerns=concerns)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.NEEDS_REVIEW

    def test_error_concern_fails_task(self):
        """Error-level concern marks task as failed."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        concerns = [
            WorkerConcern(
                level=ConcernLevel.ERROR,
                message="Cannot complete task",
            )
        ]
        queue.submit_result(task_id, {}, concerns=concerns)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    def test_escalate_concern_needs_review(self):
        """Escalate-level concern marks for review."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        concerns = [
            WorkerConcern(
                level=ConcernLevel.ESCALATE,
                message="Context inadequate for this assignment",
                suggestion="Need specification update",
                context_sample="The input was: ...",
            )
        ]
        queue.submit_result(task_id, {}, concerns=concerns)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.NEEDS_REVIEW

    def test_multiple_concerns_uses_worst(self):
        """Multiple concerns use worst-case status."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        concerns = [
            WorkerConcern(level=ConcernLevel.INFO, message="Note 1"),
            WorkerConcern(level=ConcernLevel.ERROR, message="Fatal error"),
            WorkerConcern(level=ConcernLevel.REVIEW, message="Also this"),
        ]
        queue.submit_result(task_id, {}, concerns=concerns)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED  # ERROR takes precedence


class TestQueueStatus:
    """Queue status and reporting."""

    def test_queue_status_counts(self):
        """Queue status reports correct counts."""
        queue = TaskQueue()

        # Add tasks in various states
        queue.enqueue(QueueType.GENERATE, {"i": 0})
        queue.enqueue(QueueType.GENERATE, {"i": 1})
        task_id = queue.enqueue(QueueType.GENERATE, {"i": 2})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(task_id, {})

        status = queue.get_queue_status(QueueType.GENERATE)
        assert status.total == 3
        assert status.pending == 1
        assert status.claimed == 1
        assert status.completed == 1

    def test_get_all_status(self):
        """Get status for all queues."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {})
        queue.enqueue(QueueType.VERIFY_ANSWER, {})

        all_status = queue.get_all_status()
        assert len(all_status) == len(QueueType)
        assert all_status[QueueType.GENERATE].total == 1
        assert all_status[QueueType.VERIFY_ANSWER].total == 1

    def test_get_tasks_needing_review(self):
        """Get all tasks that need review."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(
            task_id,
            {},
            concerns=[WorkerConcern(level=ConcernLevel.REVIEW, message="Review me")],
        )

        reviews = queue.get_tasks_needing_review()
        assert len(reviews) == 1
        assert reviews[0].task_id == task_id

    def test_get_failed_tasks(self):
        """Get all failed tasks."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(
            task_id,
            {},
            concerns=[WorkerConcern(level=ConcernLevel.ERROR, message="Failed")],
        )

        failed = queue.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].task_id == task_id

    def test_compact_report_format(self):
        """Compact report has expected format."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {})
        queue.enqueue(QueueType.VERIFY_ANSWER, {})

        report = queue.get_compact_report()
        assert "QUEUE STATUS:" in report
        assert "generate:" in report
        assert "verify_answer:" in report
        assert "pending" in report

    def test_compact_report_shows_alerts(self):
        """Compact report shows alerts for failures."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(
            task_id,
            {},
            concerns=[WorkerConcern(level=ConcernLevel.ERROR, message="Failed")],
        )

        report = queue.get_compact_report()
        assert "ALERTS:" in report
        assert "failed" in report.lower()


class TestRetryAndProgress:
    """Task retry and progress updates."""

    def test_retry_failed_task(self):
        """Can retry a failed task."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(
            task_id,
            {},
            concerns=[WorkerConcern(level=ConcernLevel.ERROR, message="Failed")],
        )

        success = queue.retry_task(task_id)
        assert success

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.PENDING
        assert task.claimed_by is None
        assert task.result is None

    def test_retry_pending_task_fails(self):
        """Cannot retry a pending task."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})

        success = queue.retry_task(task_id)
        assert not success

    def test_update_progress(self):
        """Can update task progress without completing."""
        queue = TaskQueue()
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")

        success = queue.update_progress(
            task_id,
            TaskStatus.IN_PROGRESS,
            notes="Working on it...",
        )
        assert success

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.result["progress_notes"] == "Working on it..."


class TestDebugMode:
    """Debug mode for context presentation review."""

    def test_debug_mode_includes_context(self):
        """Debug mode includes formatted context."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {"benchmark": "gsm1k", "difficulty": 0.5})

        task = queue.claim_task(
            QueueType.GENERATE,
            worker_type="test",
            debug_mode=True,
        )

        assert task.context_presented is not None
        assert "TASK CONTEXT" in task.context_presented
        assert "INPUT DATA" in task.context_presented
        assert "gsm1k" in task.context_presented


class TestPersistence:
    """JSONL persistence for crash recovery."""

    def test_persistence_on_enqueue(self, tmp_path):
        """Tasks are persisted on enqueue."""
        storage = tmp_path / "orchestrator"
        queue = TaskQueue(storage_dir=storage)

        queue.enqueue(QueueType.GENERATE, {"test": "data"})

        filepath = storage / "queues" / "generate.jsonl"
        assert filepath.exists()
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_load_from_storage(self, tmp_path):
        """Can load tasks from storage."""
        storage = tmp_path / "orchestrator"

        # Create and persist
        queue1 = TaskQueue(storage_dir=storage)
        queue1.enqueue(QueueType.GENERATE, {"i": 0})
        queue1.enqueue(QueueType.GENERATE, {"i": 1})

        # Load in new queue
        queue2 = TaskQueue(storage_dir=storage)
        count = queue2.load_from_storage()

        assert count == 2
        assert len(queue2.list_tasks(QueueType.GENERATE)) == 2

    def test_concerns_persisted_separately(self, tmp_path):
        """Concerns are saved to needs_review.jsonl."""
        storage = tmp_path / "orchestrator"
        queue = TaskQueue(storage_dir=storage)

        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(
            task_id,
            {},
            concerns=[
                WorkerConcern(
                    level=ConcernLevel.REVIEW,
                    message="Check this",
                )
            ],
        )

        filepath = storage / "concerns" / "needs_review.jsonl"
        assert filepath.exists()


class TestDataclassSerialization:
    """Test dataclass serialization/deserialization."""

    def test_task_to_dict_and_back(self):
        """Task can round-trip through dict."""
        task = Task(
            task_id="test-123",
            queue=QueueType.GENERATE,
            input_data={"benchmark": "gsm1k"},
        )
        task.status = TaskStatus.COMPLETED
        task.result = {"problems": []}
        task.concerns = [
            WorkerConcern(level=ConcernLevel.INFO, message="Note")
        ]

        d = task.to_dict()
        restored = Task.from_dict(d)

        assert restored.task_id == task.task_id
        assert restored.queue == task.queue
        assert restored.status == task.status
        assert restored.result == task.result
        assert len(restored.concerns) == 1
        assert restored.concerns[0].level == ConcernLevel.INFO

    def test_worker_concern_to_dict(self):
        """WorkerConcern serializes properly."""
        concern = WorkerConcern(
            level=ConcernLevel.ESCALATE,
            message="Context inadequate",
            suggestion="Update spec",
            context_sample="The input was...",
        )

        d = concern.to_dict()
        restored = WorkerConcern.from_dict(d)

        assert restored.level == ConcernLevel.ESCALATE
        assert restored.message == "Context inadequate"
        assert restored.suggestion == "Update spec"
        assert restored.context_sample == "The input was..."


class TestConvenienceFunctions:
    """Convenience functions for common operations."""

    def test_enqueue_generation(self):
        """enqueue_generation helper works."""
        # Reset global queue
        import phfe.orchestrator.task_queue as tq
        tq._default_queue = None

        task_id = enqueue_generation(
            benchmark="gsm1k",
            difficulty=0.7,
            count=50,
        )

        queue = get_queue()
        task = queue.get_task(task_id)
        assert task.input_data["benchmark"] == "gsm1k"
        assert task.input_data["difficulty"] == 0.7
        assert task.input_data["count"] == 50


class TestQueueTypes:
    """All queue types work correctly."""

    @pytest.mark.parametrize("queue_type", list(QueueType))
    def test_all_queue_types_work(self, queue_type):
        """Can enqueue and claim from all queue types."""
        queue = TaskQueue()

        task_id = queue.enqueue(queue_type, {"test": "data"})
        task = queue.claim_task(queue_type, worker_type="test")

        assert task is not None
        assert task.queue == queue_type


class TestListTasks:
    """Task listing with filters."""

    def test_list_tasks_all(self):
        """List all tasks."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {})
        queue.enqueue(QueueType.VERIFY_ANSWER, {})

        tasks = queue.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_by_queue(self):
        """Filter by queue type."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {})
        queue.enqueue(QueueType.VERIFY_ANSWER, {})

        tasks = queue.list_tasks(queue=QueueType.GENERATE)
        assert len(tasks) == 1
        assert tasks[0].queue == QueueType.GENERATE

    def test_list_tasks_by_status(self):
        """Filter by status."""
        queue = TaskQueue()
        queue.enqueue(QueueType.GENERATE, {})
        task_id = queue.enqueue(QueueType.GENERATE, {})
        queue.claim_task(QueueType.GENERATE, worker_type="test")
        queue.submit_result(task_id, {})

        pending = queue.list_tasks(status=TaskStatus.PENDING)
        completed = queue.list_tasks(status=TaskStatus.COMPLETED)

        assert len(pending) == 0
        assert len(completed) == 1

    def test_list_tasks_limit(self):
        """Respect limit parameter."""
        queue = TaskQueue()
        for i in range(10):
            queue.enqueue(QueueType.GENERATE, {"i": i})

        tasks = queue.list_tasks(limit=3)
        assert len(tasks) == 3


class TestConcernSummary:
    """Concern summary across all tasks."""

    def test_concern_summary(self):
        """Get summary of concerns by level."""
        queue = TaskQueue()

        # Create some tasks with concerns
        for level in [ConcernLevel.INFO, ConcernLevel.INFO, ConcernLevel.ERROR]:
            task_id = queue.enqueue(QueueType.GENERATE, {})
            queue.claim_task(QueueType.GENERATE, worker_type="test")
            queue.submit_result(
                task_id,
                {},
                concerns=[WorkerConcern(level=level, message="Test")],
            )

        summary = queue.get_concern_summary()
        assert summary["info"] == 2
        assert summary["error"] == 1
