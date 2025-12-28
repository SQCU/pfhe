# Claudefile: Synthetic Reasoning Curriculum Generator

## Context for Implementing Claude

You're building the data generation infrastructure for a language model training project. The core thesis is that curriculum matters more than raw token count — a carefully staged curriculum can achieve in 36 GPU-hours what naive webscrape training achieves in 36,000.

The bottleneck for long training runs is **curriculum depth**: you need enough distinct, increasingly difficult data that the model keeps learning for the entire training window without saturating.

This module generates that data.

## The Curriculum Stages

### Stage 0: Syntax Acquisition (not your problem)

This is handled by TinyStories or similar existing datasets. The model learns basic grammar and token patterns. Saturates in ~30 GPU-minutes.

### Stage 1: Discourse and Structure (partially your problem)

The model learns to maintain coherence across paragraphs, track references, maintain topic. 

Data: Multi-paragraph documents with explicit structure. You may need to generate some of this synthetically.

### Stage 2: Reasoning Chains (your main problem)

The model learns to perform multi-step reasoning. This is where "depth" matters and where most training time is spent.

Data: Problems with verifiable solutions + reasoning traces from teacher models.

### Stage 3: Domain Depth (stretch goal)

Specialized knowledge in specific domains. Handled separately (e.g., the arxiv pipeline).

---

## What You're Building

A pipeline that generates reasoning problems with:
1. **Programmatically generated problems** with known solutions (for verification)
2. **Teacher model reasoning traces** showing how to solve them (for distillation)
3. **Difficulty scaling** so problems get harder as training progresses

### Problem Domains

#### Mathematics

**Elementary (saturates fast):**
- Arithmetic: "What is 347 + 892?"
- Fractions: "What is 3/4 + 2/5?"
- Percentages: "What is 15% of 240?"

**Intermediate:**
- Word problems: "Alice has 3 apples. Bob gives her 2 more. Charlie takes half. How many does Alice have?"
- Multi-step arithmetic: "Calculate (347 + 892) × 3 - 156"
- Basic algebra: "Solve for x: 3x + 7 = 22"

**Advanced:**
- Systems of equations
- Quadratics
- Probability/combinatorics
- Competition math (AMC/AIME style)

**Generation approach:**
- Procedurally generate problems by instantiating templates with random values
- Solve programmatically to get ground truth
- Get reasoning traces from teacher model (Kimi K2, DeepSeek, etc.)
- Verify teacher's answer matches ground truth before including in dataset

#### Logic Puzzles

**Simple:**
- "If all dogs are animals, and Rex is a dog, is Rex an animal?"
- "Alice is taller than Bob. Bob is taller than Charlie. Who is shortest?"

**Intermediate:**
- Knights and knaves: "A says 'B is a liar.' B says 'A and I are the same type.' What are they?"
- Constraint satisfaction: "Five people sit in a row. Alice won't sit next to Bob..."

**Advanced:**
- Zebra puzzles (Einstein's riddle variants)
- First-order logic problems
- Proof verification

**Generation approach:**
- Procedurally generate puzzle instances
- Solve with SAT solver / constraint solver to get ground truth
- Get teacher reasoning traces
- Verify

#### Code

**Simple:**
- "Write a function that returns the sum of a list"
- "Write a function that reverses a string"

**Intermediate:**
- "Write a function that finds the longest common subsequence"
- "Implement binary search"
- "Parse this JSON and extract field X"

**Advanced:**
- LeetCode medium/hard problems
- Multi-file tasks
- Debugging: "This code has a bug. Find and fix it."

**Generation approach:**
- Curate from existing datasets (MBPP, HumanEval, LeetCode)
- Generate variations by changing variable names, problem parameters
- Verify solutions by executing against test cases
- Get teacher traces for the reasoning/planning process

#### Science QA

**Factual (not our focus, but useful):**
- "What is the chemical formula for water?"
- "What is the speed of light?"

**Reasoning:**
- "If you double the pressure on a gas at constant temperature, what happens to the volume?"
- "A ball is thrown upward at 20 m/s. How high does it go?"
- "Why does ice float?"

**Generation approach:**
- Curate from existing QA datasets
- Generate variations
- This is harder to verify automatically — use teacher agreement as proxy

---

## Difficulty Curriculum

The key insight: don't just generate random problems. Generate problems in order of difficulty, so the model is always working at the edge of its capability.

**Difficulty proxies:**
- Number of reasoning steps required
- Number of entities/variables to track
- Numerical magnitude (larger numbers = harder arithmetic)
- Distractor information (irrelevant details that must be ignored)

**Implementation:**

```python
class CurriculumGenerator:
    def __init__(self, domain: str, difficulty_schedule: Callable[[int], float]):
        """
        difficulty_schedule: maps training step -> target difficulty (0.0 to 1.0)
        """
        self.domain = domain
        self.difficulty_schedule = difficulty_schedule
    
    def generate_batch(self, step: int, batch_size: int) -> list[Problem]:
        target_difficulty = self.difficulty_schedule(step)
        problems = []
        for _ in range(batch_size):
            problem = self.generate_problem(target_difficulty)
            problems.append(problem)
        return problems
```

**Difficulty schedules:**

```python
# Linear ramp
def linear_schedule(step, warmup=1000, max_steps=100000):
    if step < warmup:
        return 0.1
    return min(1.0, 0.1 + 0.9 * (step - warmup) / (max_steps - warmup))

# Staged (jump in difficulty at specific points)
def staged_schedule(step):
    if step < 5000: return 0.2
    if step < 20000: return 0.5
    if step < 50000: return 0.8
    return 1.0
```

---

## Teacher Model Integration

For each problem, we need a reasoning trace from a teacher model. This is the distillation signal.

**Requirements:**
- Teacher model running in vLLM or similar (Kimi K2 quantized, DeepSeek, Qwen)
- Prompt format that elicits chain-of-thought reasoning
- Verification that teacher's final answer is correct

**Pipeline:**

```
Problem → Teacher Prompt → Teacher Generation → Parse Answer → Verify → Include/Reject
```

**Prompt template (math example):**

```
Solve this problem step by step. Show your reasoning, then give your final answer.

Problem: {problem_text}

Solution:
```

**Verification:**
- Parse teacher's final answer (regex for numbers, code execution for code)
- Compare to ground truth
- If wrong: regenerate with different temperature/sampling, or discard

**Throughput consideration:**
- Teacher inference is slow
- Pre-generate a large pool of problems + traces
- Or: generate on-the-fly with caching

---

## Output Format

Each curriculum example should include:

```python
@dataclass
class CurriculumExample:
    problem_id: str
    domain: str  # "math", "logic", "code", etc.
    difficulty: float  # 0.0 to 1.0
    
    problem_text: str  # The problem statement
    ground_truth: str  # The correct answer (for verification)
    
    teacher_trace: str  # Full reasoning trace from teacher
    teacher_model: str  # Which teacher generated this
    
    # For code problems
    test_cases: list[tuple[str, str]] | None  # (input, expected_output)
    
    # Metadata
    generation_params: dict  # How the problem was generated
    verified: bool  # Did teacher answer match ground truth?
```

**Storage:**
- JSONL files, one example per line
- Organized by domain and difficulty: `curriculum/math/difficulty_0.3/batch_0001.jsonl`
- Index file mapping difficulty ranges to file paths

---

## Interface

### Python API

```python
from curriculum_generator import MathGenerator, LogicGenerator, CodeGenerator, MixedGenerator

# Generate a batch at specific difficulty
math_gen = MathGenerator(teacher_model="kimi-k2")
problems = math_gen.generate_batch(difficulty=0.5, batch_size=100)

# Generate with automatic difficulty scheduling
mixed_gen = MixedGenerator(
    domains=["math", "logic", "code"],
    weights=[0.5, 0.3, 0.2],
    teacher_model="kimi-k2",
)
for step in range(100000):
    batch = mixed_gen.generate_for_step(step, batch_size=32)
    # ... training ...

# Pre-generate entire curriculum
mixed_gen.generate_curriculum(
    total_examples=1_000_000,
    output_dir="./curriculum/",
    difficulty_schedule=linear_schedule,
)
```

### CLI

```bash
# Generate math curriculum
python -m curriculum_generator generate-math \
    --difficulty-min 0.1 --difficulty-max 1.0 \
    --num-examples 100000 \
    --teacher-model kimi-k2 \
    --output-dir ./curriculum/math/

# Generate mixed curriculum for full training run
python -m curriculum_generator generate-full \
    --domains math,logic,code \
    --weights 0.5,0.3,0.2 \
    --num-examples 1000000 \
    --output-dir ./curriculum/full/

# Verify existing curriculum (re-check answers)
python -m curriculum_generator verify ./curriculum/math/ --fix
```

---

## Scalability

For 24 hours of training on 32 GPUs with batch size 32 and ~1 step/second:
- ~86,400 steps
- ~2.7M examples consumed (with some repetition probably fine)

To avoid saturation, you want:
- At least 1M unique examples
- Difficulty spread across the full range
- Multiple domains to prevent overfitting to one problem type

**Teacher inference bottleneck:**
- If teacher generates 10 tokens/second and traces are ~200 tokens
- That's 1 trace per 20 seconds per GPU
- With 8 teacher GPUs: 1,440 traces/hour = 34K traces/day
- Not enough for 1M examples!

**Solutions:**
- Pre-generate curriculum before training
- Use smaller/faster teacher for easy problems
- Cache and reuse traces for similar problems
- Accept some repetition in training data

---

## Dependencies

```
# Problem generation
sympy              # symbolic math for verification
z3-solver          # constraint solving for logic puzzles

# Teacher inference
vllm               # fast inference
transformers       # fallback
openai             # if using API-based teachers

# Code execution
docker             # sandboxed code execution
restrictedpython   # lightweight sandboxing alternative

# Storage
orjsonl            # fast JSONL
```

---

## Implementation Priority

1. **Math generator (elementary + intermediate)** — easiest to verify, high value
2. **Teacher integration** — required for all domains
3. **Difficulty scheduling** — required for curriculum property
4. **Logic generator** — medium difficulty, adds diversity
5. **Code generator** — hardest (execution sandboxing), but high value

---

## Test Cases

Verify generators produce valid problems:

```python
def test_math_generator():
    gen = MathGenerator()
    for difficulty in [0.1, 0.3, 0.5, 0.7, 0.9]:
        problems = gen.generate_batch(difficulty, batch_size=10)
        for p in problems:
            # Problem is solvable
            assert p.ground_truth is not None
            # Difficulty is approximately correct
            assert abs(p.difficulty - difficulty) < 0.2
            # Problem text is well-formed
            assert len(p.problem_text) > 10
            assert "?" in p.problem_text or "=" in p.problem_text
```

---

## Open Questions

1. **How to measure problem difficulty objectively?** Current approach: proxy metrics (step count, entity count). Better: calibrate against model performance on held-out set.

2. **How much repetition is okay?** Training on the same problem twice is fine. Training on it 100 times is memorization. Where's the line?

3. **How to handle teacher errors?** Even good teachers make mistakes. Verification catches some but not all. Accept ~5% error rate? Filter more aggressively?

4. **Domain balance:** How to weight math vs. logic vs. code? Depends on what capabilities we're optimizing for. Make this configurable.

---

When done, this module should be able to generate millions of verified reasoning problems with difficulty-appropriate teacher traces, sufficient to keep a multi-GPU training run learning for 24+ hours.
