"""
Orchestrator - Subagent dispatch and observability infrastructure.

Provides:
- TutorCaller: Wrapper for calling tutor models with logit capture
- WorkflowTrace: Full tracing of multi-step workflows
- TraceStore: Persistent storage for debugging and analysis
- TaskQueue: Uniform task distribution for Claude Code agents, scripts, and vLLM
"""

from .observability import (
    SubagentLog,
    WorkflowTrace,
    WorkflowStatus,
    TraceStore,
    TutorType,
    Timer,
    hash_prompt,
)
from .tutor import (
    TutorCaller,
    TutorConfig,
    MODEL_MAP,
)
from .task_queue import (
    # Enums
    TaskStatus,
    QueueType,
    ConcernLevel,
    # Data classes
    WorkerConcern,
    Task,
    QueueStatus,
    # Main class
    TaskQueue,
    # Convenience
    get_queue,
    enqueue_generation,
    enqueue_contamination_check,
    enqueue_tutor_inference,
)

__all__ = [
    # Observability
    "SubagentLog",
    "WorkflowTrace",
    "WorkflowStatus",
    "TraceStore",
    "TutorType",
    "Timer",
    "hash_prompt",
    # Tutor caller
    "TutorCaller",
    "TutorConfig",
    "MODEL_MAP",
    # Task Queue
    "TaskStatus",
    "QueueType",
    "ConcernLevel",
    "WorkerConcern",
    "Task",
    "QueueStatus",
    "TaskQueue",
    "get_queue",
    "enqueue_generation",
    "enqueue_contamination_check",
    "enqueue_tutor_inference",
]
