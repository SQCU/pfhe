# Claudefile: PHFE Task Orchestrator Specification

## Context

Adapting the subagent task distribution pattern from `reference_dialogue_yoinker/` for PHFE's synthetic corpus generation. The key insight: **tool use makes everything easy** — Claude Code subagents can claim tasks, process them, and submit results all via uniform tool interfaces.

## Design Goals

1. **Uniform Interface**: Same API whether consumed by:
   - Claude Code subagents (via tool use within sessions)
   - External scripts (via REST API / direct Python calls)
   - Orchestrator monitoring progress

2. **Debug-Friendly**: Early integration runs need visibility into:
   - What context was presented to NLP tools
   - Why tasks failed (context inadequacy, assignment unclear, etc.)
   - Compact summaries for Claude Code orchestration

3. **Capability-Agnostic**: Interface adapts to different NLP tool capability levels:
   - Claude Opus for complex reasoning (curator, verifier)
   - Claude Haiku for fast labeling (contamination check, format)
   - vLLM models for generation (tutor inference)
   - Future tools with unknown capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK ORCHESTRATOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Task Queue  │    │ Task Queue  │    │ Task Queue  │    │
│  │  (generate) │    │  (verify)   │    │ (distill)   │    │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              WORKER INTERFACE (TOOLS)               │  │
│  │                                                     │  │
│  │  claim_task(queue, worker_type) -> Task            │  │
│  │  submit_result(task_id, result, concerns) -> Ack   │  │
│  │  report_progress(task_id, status, notes) -> Ack    │  │
│  │  get_queue_status(queue) -> Stats                  │  │
│  │                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│         ▲                  ▲                  ▲            │
│         │                  │                  │            │
│  ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐   │
│  │ CC Agent:   │    │ CC Agent:   │    │ CC Agent:   │   │
│  │ generator   │    │ verifier    │    │ distiller   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│         ▲                  ▲                  ▲            │
│         │                  │                  │            │
│  ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐   │
│  │ Ext Script: │    │ vLLM:       │    │ API:        │   │
│  │ batch gen   │    │ tutor infer │    │ orchestrate │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Task Queue Types for PHFE

### 1. `generate` - Synthetic Problem Generation

**Input**: Benchmark type, difficulty range, count
**Output**: Candidate synthetic problems
**Workers**: Claude (generation), external scripts

```python
@dataclass
class GenerateTask:
    task_id: str
    benchmark: str  # "gsm1k", "arc", "mbpp", etc.
    difficulty: float  # 0.0 - 1.0
    count: int
    template_hints: Optional[list[str]] = None

@dataclass
class GenerateResult:
    problems: list[dict]  # problem_id, text, answer, domain
    generation_notes: str
    confidence: float
```

### 2. `contaminate_check` - Firewall Verification

**Input**: Synthetic problem, canonical index reference
**Output**: Pass/fail with rejection reasons
**Workers**: Fast local (Python), Claude (review edge cases)

```python
@dataclass
class ContaminateCheckTask:
    task_id: str
    synthetic_problem: dict
    domain: str

@dataclass
class ContaminateCheckResult:
    passed: bool
    rejection_reasons: list[str]
    similarity_scores: dict[str, float]
    closest_canonical: Optional[str]
```

### 3. `tutor_inference` - Answer Key Generation

**Input**: Problem, tutor model config
**Output**: Reasoning trace + sparse logits
**Workers**: vLLM inference, API clients

```python
@dataclass
class TutorInferenceTask:
    task_id: str
    problem_text: str
    problem_id: str
    tutor_model: str  # "deepseek-r1", "kimi-k2", "gpt-4o"
    extract_logits: bool
    top_p: float = 0.99

@dataclass
class TutorInferenceResult:
    reasoning_trace: str
    final_answer: str
    tokens: list[int]
    sparse_logits: list[dict]  # position -> {token_id: log_prob}
    verified_correct: Optional[bool]
```

### 4. `verify_answer` - Answer Verification

**Input**: Problem + tutor answer
**Output**: Correctness judgment
**Workers**: Claude (math/logic), execution (code)

```python
@dataclass
class VerifyAnswerTask:
    task_id: str
    problem: dict
    tutor_answer: str
    domain: str  # "math", "code", "mcq"

@dataclass
class VerifyAnswerResult:
    correct: bool
    explanation: str
    ground_truth: Optional[str]
```

### 5. `icr_transform` - ICR Augmentation

**Input**: Canonical problem
**Output**: ICR-augmented version
**Workers**: Local (prepend library), Claude (validate)

```python
@dataclass
class ICRTransformTask:
    task_id: str
    problem: dict
    benchmark: str

@dataclass
class ICRTransformResult:
    icr_instance: dict
    method_library_hash: str
```

## Worker Concern Levels

Workers flag issues for orchestrator review:

| Level | Meaning | Action |
|-------|---------|--------|
| `info` | Just noting something | Log only |
| `review` | Needs human/orchestrator attention | Queue for review |
| `retry` | Transient failure, retry suggested | Re-queue task |
| `error` | Task failed, mark failed | Mark failed, log |
| `escalate` | Context inadequate for assignment | Pause queue, alert |

```python
@dataclass
class WorkerConcern:
    level: str  # info, review, retry, error, escalate
    message: str
    suggestion: Optional[str] = None
    context_sample: Optional[str] = None  # What the worker saw
```

## Claude Code Agent Definitions

### `.claude/agents/problem-generator.md`

```markdown
---
name: problem-generator
description: Generate synthetic problems for a benchmark domain. Pulls from
  generate queue, creates problems that are structurally similar but textually
  distinct from canonical problems.
model: sonnet
color: blue
---

You generate synthetic training problems. You receive:
1. A benchmark type (gsm1k, arc, mbpp, etc.)
2. A difficulty level (0.0-1.0)
3. Optional template hints

You output problems that:
- Match the benchmark's format exactly
- Have verifiable answers
- Are NOT copies of real benchmark problems
- Scale appropriately to difficulty level

## Ticket Workflow

1. **Claim task**:
```bash
phfe task claim generate --worker-type problem_generator
```

2. **Generate problems**
3. **Submit result**:
```bash
phfe task submit {task_id} --result problems.json --concerns concerns.json
```

[... detailed instructions, examples, vocabulary ...]
```

### `.claude/agents/answer-verifier.md`

```markdown
---
name: answer-verifier
description: Verify tutor-generated answers against ground truth or by
  independent reasoning. Catches tutor errors before they pollute training data.
model: opus
color: red
---

You verify answers. You receive:
1. A problem statement
2. A proposed answer (from a tutor model)
3. Domain type (math, code, multiple_choice)

You output:
- Whether the answer is correct
- Your reasoning
- The correct answer if different

## Verification Strategies by Domain

### Math
- Re-solve independently
- Check each step in tutor's reasoning
- Verify final numerical answer

### Code
- Execute against test cases (if available)
- Check logic correctness
- Verify edge cases

### Multiple Choice
- Eliminate wrong options
- Verify selected option matches criteria

[... detailed instructions ...]
```

### `.claude/agents/contamination-reviewer.md`

```markdown
---
name: contamination-reviewer
description: Review edge-case contamination flags. When the automated firewall
  is uncertain, this agent makes the final call on whether a synthetic problem
  is too similar to canonical benchmarks.
model: sonnet
color: yellow
---

You are a contamination judge. You receive:
1. A synthetic problem
2. The closest canonical problem(s)
3. Similarity scores from automated checks

You decide whether the synthetic is:
- PASS: Sufficiently distinct, safe for training
- FAIL: Too similar, would contaminate evaluation

## Decision Criteria

Similarity in STRUCTURE is okay:
- Same problem type (word problem, multiple choice)
- Same difficulty level
- Same domain

Similarity in CONTENT is not okay:
- Same numbers/entities with minor rewording
- Same story with name changes
- Paraphrase of canonical problem

[... detailed instructions, examples ...]
```

## Tool Interface (Python)

```python
# src/phfe/orchestrator/task_queue.py

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import uuid
import json
from pathlib import Path

class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class QueueType(str, Enum):
    GENERATE = "generate"
    CONTAMINATE_CHECK = "contaminate_check"
    TUTOR_INFERENCE = "tutor_inference"
    VERIFY_ANSWER = "verify_answer"
    ICR_TRANSFORM = "icr_transform"

@dataclass
class Task:
    task_id: str
    queue: QueueType
    input_data: dict
    status: TaskStatus = TaskStatus.PENDING
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    result: Optional[dict] = None
    concerns: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class TaskQueue:
    """
    Task queue with uniform interface for all worker types.

    Workers (Claude Code agents, scripts, vLLM) all use the same interface:
    - claim_task() -> get work
    - submit_result() -> deliver output
    - report_concern() -> flag issues
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._queues: dict[QueueType, list[Task]] = {q: [] for q in QueueType}

    def enqueue(self, queue: QueueType, input_data: dict) -> str:
        """Add a task to a queue. Returns task_id."""
        task = Task(
            task_id=str(uuid.uuid4())[:12],
            queue=queue,
            input_data=input_data,
        )
        self._queues[queue].append(task)
        return task.task_id

    def claim_task(
        self,
        queue: QueueType,
        worker_type: str,
        worker_id: Optional[str] = None,
    ) -> Optional[Task]:
        """
        Claim the next available task from a queue.

        Returns None if queue is empty.
        """
        for task in self._queues[queue]:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CLAIMED
                task.claimed_by = worker_type
                task.claimed_at = datetime.now().isoformat()
                return task
        return None

    def submit_result(
        self,
        task_id: str,
        result: dict,
        concerns: Optional[list[dict]] = None,
    ) -> bool:
        """
        Submit result for a claimed task.

        Returns True if successful.
        """
        task = self._find_task(task_id)
        if task is None:
            return False

        task.result = result
        task.concerns = concerns or []

        # Check concern levels
        has_error = any(c.get("level") == "error" for c in task.concerns)
        needs_review = any(c.get("level") in ("review", "escalate") for c in task.concerns)

        if has_error:
            task.status = TaskStatus.FAILED
        elif needs_review:
            task.status = TaskStatus.NEEDS_REVIEW
        else:
            task.status = TaskStatus.COMPLETED

        return True

    def get_queue_status(self, queue: QueueType) -> dict:
        """Get status summary for a queue."""
        tasks = self._queues[queue]
        return {
            "queue": queue.value,
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "claimed": sum(1 for t in tasks if t.status == TaskStatus.CLAIMED),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "needs_review": sum(1 for t in tasks if t.status == TaskStatus.NEEDS_REVIEW),
        }

    def get_compact_report(self) -> str:
        """
        Get compact summary for Claude Code orchestration.

        This is what Claude Code sees to understand queue health.
        """
        lines = ["QUEUE STATUS:"]
        for queue in QueueType:
            status = self.get_queue_status(queue)
            lines.append(
                f"  {queue.value}: {status['completed']}/{status['total']} done, "
                f"{status['pending']} pending, {status['failed']} failed"
            )
        return "\n".join(lines)
```

## CLI Interface

```bash
# Queue management
phfe task enqueue generate --benchmark gsm1k --difficulty 0.5 --count 100
phfe task status generate
phfe task list --queue generate --status pending

# Worker operations (used by Claude Code agents via tool)
phfe task claim generate --worker-type problem_generator
phfe task submit {task_id} --result result.json
phfe task concern {task_id} --level review --message "Unusual pattern"

# Orchestrator operations
phfe task report  # Compact summary for Claude Code
phfe task review  # List tasks needing review
phfe task retry {task_id}  # Re-queue failed task
```

## Integration with Claude Code

Claude Code agents use tools to interact with queues:

```python
# In Claude Code session, agent uses Bash tool:

# 1. Check what work is available
result = Bash("phfe task report")
# -> "QUEUE STATUS:
#      generate: 45/100 done, 50 pending, 5 failed
#      verify: 40/45 done, 5 pending, 0 failed"

# 2. Claim a task
result = Bash("phfe task claim generate --worker-type problem_generator")
# -> {"task_id": "abc123", "input_data": {...}}

# 3. Process the task (agent does work with local context)

# 4. Submit result
result = Bash("phfe task submit abc123 --result /tmp/result.json")
# -> {"success": true, "status": "completed"}
```

## Early Integration Debug Mode

For debugging context presentation issues:

```python
@dataclass
class DebugTaskResult:
    """Extended result with debug information."""
    result: dict

    # Debug fields for reviewing context adequacy
    context_received: str  # What the worker actually saw
    context_tokens: int
    processing_notes: list[str]
    decision_trace: str  # How the worker reasoned
    suggestions_for_context: Optional[list[str]]  # If context was inadequate
```

Enable with:
```bash
phfe task claim generate --debug-mode
```

This returns the full context that will be presented, allowing review before processing.

## Persistence

Tasks persisted as JSONL for crash recovery:

```
orchestrator_data/
├── queues/
│   ├── generate.jsonl
│   ├── contaminate_check.jsonl
│   ├── tutor_inference.jsonl
│   └── verify_answer.jsonl
├── completed/
│   └── 2025-12-22/
│       └── tasks.jsonl
└── concerns/
    └── needs_review.jsonl
```

## Example Workflow: Synthetic Problem Generation

1. **Orchestrator** enqueues generation tasks:
```python
for benchmark in ["gsm1k", "arc", "mbpp"]:
    for difficulty in [0.3, 0.5, 0.7]:
        queue.enqueue(QueueType.GENERATE, {
            "benchmark": benchmark,
            "difficulty": difficulty,
            "count": 100,
        })
```

2. **Claude Code problem-generator** claims and processes:
```bash
# Agent claims task
task = phfe task claim generate --worker-type problem_generator
# Agent generates problems (using its context, method libraries, etc.)
# Agent submits
phfe task submit {task.task_id} --result problems.json
```

3. **Contamination checker** (fast Python) processes:
```python
while task := queue.claim_task(QueueType.CONTAMINATE_CHECK, "firewall"):
    result = firewall.check(task.input_data["problem"])
    queue.submit_result(task.task_id, {"passed": result.passed, ...})
```

4. **Tutor inference** (vLLM) generates answer keys:
```python
while task := queue.claim_task(QueueType.TUTOR_INFERENCE, "vllm_deepseek"):
    output = vllm_server.generate(task.input_data["problem_text"], logprobs=True)
    queue.submit_result(task.task_id, {"trace": output.text, "logits": ...})
```

5. **Claude Code verifier** validates answers:
```bash
# Opus-tier agent for complex verification
task = phfe task claim verify_answer --worker-type answer_verifier
# ... reasoning ...
phfe task submit {task.task_id} --result verification.json
```

6. **Orchestrator** monitors and handles concerns:
```python
# In Claude Code orchestrator
report = queue.get_compact_report()
if "needs_review" in report:
    # Dispatch to appropriate reviewer
    reviews = queue.get_tasks_needing_review()
    for review in reviews:
        # Escalate or handle
```

## Benefits of This Design

1. **Debuggable**: Every task has full context trace
2. **Flexible**: Same interface for Claude, scripts, vLLM
3. **Scalable**: Queues can be processed in parallel
4. **Recoverable**: JSONL persistence allows crash recovery
5. **Observable**: Compact reports for orchestration
6. **Extensible**: New task types / workers just implement interface
