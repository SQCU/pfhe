# PHFE Project Progress Review

**Date**: 2025-12-22
**Status**: Phase 1 Infrastructure (Substantial Progress)

---

## Project Goals Summary

PHFE (Posthumanity's First Exam) is a benchmark suite measuring **in-context learning** vs **memorization**. The core scientific claim:

> "A model trained on synthetic problems (verified non-overlapping with canonical benchmarks) achieves X% on canonical eval and Y% on ICR-augmented eval. The gap (Y - X) represents pure in-context learning capability, not contamination."

---

## Specification Checklist

### Core Specifications

| Spec File | Purpose | Implementation Status |
|-----------|---------|----------------------|
| `pfe_benchmark_spec.md` | Benchmark suite, three-split architecture, contamination firewall | **Substantial** |
| `answer_key_corpus.md` | Multi-tutor answer key collection, style diversity | Stub only |
| `offline_distillation_protocol.md` | GKD training with cached tutor logits | **Cross-tokenizer core done** |

### Supporting Specifications

| Spec File | Purpose | Implementation Status |
|-----------|---------|----------------------|
| `benchmark_icr_transform.md` | ICR transformation methodology | **Complete** |
| `arxiv_tex_pipeline.md` | LaTeX → multimodal training data | Stub only |
| `curriculum_generator.md` | Procedural problem generation (superseded) | Stub only |
| `language_eval_suite.md` | Language competence metrics | Partial (repetition only) |

---

## Phase 1: Infrastructure — Detailed Status

### Completed ✅

#### 1. Project Structure & Tooling
- [x] `uv` environment with Python 3.10
- [x] `pyproject.toml` with all dependencies (ruff, mypy, pytest, sentence-transformers, etc.)
- [x] Source layout: `src/phfe/` with module packages
- [x] Test infrastructure: `tests/` with pytest

#### 2. Orchestrator Infrastructure (`src/phfe/orchestrator/`)
- [x] `observability.py`: `SubagentLog`, `WorkflowTrace`, `TraceStore`, `Timer`
- [x] `tutor.py`: `TutorCaller` with OpenAI/Anthropic/vLLM backends
- [x] Sparse logit storage format (`SparseLogits` dataclass)
- [x] Cost tracking per API call

#### 3. Contamination Firewall (`src/phfe/benchmark/contamination.py`)
- [x] **Token-level overlap**: n-gram Jaccard similarity (5-grams, 30% threshold)
- [x] **Semantic similarity**: Sentence-transformer embeddings (85% cosine threshold)
- [x] **Math structural**: Same numbers + operations + answer detection
- [x] **Code structural**: Same function name + test inputs detection
- [x] Combined `ContaminationFirewall` class
- [x] Batch indexing for efficiency
- [x] Statistics tracking (pass rate, rejection reasons)
- [x] **37/37 tests passing**

#### 4. CLI Framework (`src/phfe/cli.py`)
- [x] Typer-based CLI with subcommands
- [x] `phfe status` - show project status
- [x] `phfe benchmark list` - list benchmarks
- [x] Stub commands for all major operations

#### 5. Benchmark Loading (`src/phfe/benchmark/loader.py`) ✅ NEW
- [x] `BenchmarkLoader` class with HuggingFace datasets integration
- [x] `BenchmarkConfig` dataclass for per-benchmark configuration
- [x] Configs for: GSM8K, ARC-Easy/Challenge, RACE, BoolQ, HellaSwag, WinoGrande, MBPP
- [x] Field extraction, options parsing, answer type handling
- [x] Caching and limit support
- [x] **23/23 tests passing**

#### 6. ICR Transformation (`src/phfe/icr_transform/`) ✅ NEW
- [x] `method_libraries.py`: Method libraries for all 10+ benchmarks
  - GSM1K: Arithmetic word problem methods (6 methods)
  - ARC: Science fact library (5 categories + worked examples)
  - RACE: Reading comprehension strategies (6 strategies)
  - BoolQ: Yes/no question strategies (4 approaches)
  - HellaSwag: Commonsense completion principles (5 principles)
  - WinoGrande: Coreference resolution strategies (4 strategies)
  - MBPP: Programming pattern library (6 patterns)
  - FnCall: Function calling guide (6 example functions)
  - Format: Data transformation patterns (6 patterns)
- [x] `ICRTransformer` class for prepending method libraries
- [x] `ICRInstance` and `ICRBatch` dataclasses
- [x] Library versioning via content hashing
- [x] **39/39 tests passing**

#### 7. Cross-Tokenizer Distillation (`src/phfe/distillation/cross_tokenizer.py`) ✅ NEW
- [x] `SparseLogits`: Top-p sparse logit representation
- [x] `CrossTokenizerAligner`: Character-span-based token alignment
- [x] `LogitAggregator`: Multiple aggregation strategies (first, average, max)
- [x] `VocabularyMapper`: Exact/normalized token mapping between vocabs
- [x] `compute_gkd_loss`: Forward KL loss computation
- [x] **21/21 tests passing**

#### 8. Cross-Tokenizer GKD Testbed (`scripts/cross_tokenizer_testbed.py`) ✅ NEW
- [x] Tested with real models: Qwen2-0.5B (student) + Pythia-1B (teacher)
- [x] **100% alignment coverage** despite different tokenizers (151K vs 50K vocab)
- [x] **82.8% vocabulary mapping** between tokenizers
- [x] Real GKD loss computation verified:
  - "Hello world!": 8.85 nats
  - Complex text: 11-14 nats (expected divergence)
- [x] GPU inference working with device_map="auto"

#### 9. Task Queue Orchestrator (`src/phfe/orchestrator/task_queue.py`) ✅ NEW
- [x] `TaskQueue` class with uniform interface for all worker types
- [x] Supports: Claude Code subagents, external scripts, vLLM servers
- [x] Queue types: generate, contaminate_check, tutor_inference, verify_answer, icr_transform
- [x] Worker concern levels: info, review, retry, error, escalate
- [x] JSONL persistence for crash recovery
- [x] Compact reporting for Claude Code orchestration
- [x] **42/42 tests passing**

#### 10. Claude Code Agent Definitions (`.claude/agents/`) ✅ NEW
- [x] `problem-generator.md` — Generate synthetic problems via task queue
- [x] `answer-verifier.md` — Verify tutor answers (Opus-tier)
- [x] `contamination-reviewer.md` — Review edge-case contamination flags

### In Progress 🔄

#### 9. Tutor Inference Setup
- [ ] vLLM server configuration for open-weight models
- [ ] API client configuration (OpenAI, Anthropic)
- [ ] Logit capture during generation

#### 9. Answer Key Logging
- [ ] Token + logit storage in Parquet format
- [ ] Multi-tutor collection pipeline
- [ ] Style diversity enforcement (<40% per tutor)

---

## Phase 2: Corpus Generation — Not Started

| Task | Status |
|------|--------|
| Generate synthetic problems | ❌ |
| Contamination checking at generation time | ❌ |
| Collect answer keys from tutor ensemble | ❌ |
| Quality filtering and verification | ❌ |
| Assemble final corpus (39K problems, 100K+ answer keys) | ❌ |

---

## Phase 3: Training — Partial

| Task | Status |
|------|--------|
| Implement GKD training loop | ❌ |
| Cross-tokenizer alignment | ✅ Core logic done |
| Regularization mixing | ❌ |
| Train student model | ❌ |

---

## Phase 4: Evaluation — Not Started

| Task | Status |
|------|--------|
| Canonical benchmark evaluation | ❌ |
| ICR-augmented evaluation | ❌ |
| Baseline comparisons | ❌ |
| Report generation | ❌ |

---

## Benchmark Coverage

### Target Benchmarks

| Benchmark | Canonical Size | Synthetic Target | Domain | Status |
|-----------|---------------|------------------|--------|--------|
| GSM1K | 1,250 | 10,000 | math | Loader + ICR ready |
| ARC-Easy | 2,376 | 2,500 | science | Loader + ICR ready |
| ARC-Challenge | 1,172 | 2,500 | science | Loader + ICR ready |
| RACE | 4,934 | 5,000 | reading | Loader + ICR ready |
| BoolQ | 3,270 | 5,000 | boolean | Loader + ICR ready |
| HellaSwag | 10,042 | 5,000 | commonsense | Loader + ICR ready |
| WinoGrande | 1,267 | 3,000 | coreference | Loader + ICR ready |
| MBPP | 500 | 2,000 | code | Loader + ICR ready |
| FnCall | 500 | 2,000 | code | ICR ready (no HF dataset) |
| Format | 500 | 2,000 | code | ICR ready (no HF dataset) |

**Total Canonical**: ~25,000 problems
**Total Synthetic Target**: ~39,000 problems

---

## Tutor Ensemble

| Model | Access | Role | Integration Status |
|-------|--------|------|-------------------|
| DeepSeek-R1 | Weights | Reasoning traces | TutorCaller stub |
| Kimi K2 | Weights | Math strength | TutorCaller stub |
| Qwen-72B | Weights | General capability | TutorCaller stub |
| GPT-4o | API | High accuracy baseline | TutorCaller ready |
| Claude Sonnet | API | Clear explanations | TutorCaller ready |

---

## Test Coverage

```
Total: 162 tests passing

tests/test_contamination.py ........... 37 passed
tests/test_benchmark_loader.py ........ 23 passed
tests/test_icr_transform.py ........... 39 passed
tests/test_cross_tokenizer.py ......... 21 passed
tests/test_task_queue.py .............. 42 passed
```

| Component | Tests | Status |
|-----------|-------|--------|
| Tokenization & n-grams | 6 | ✅ |
| Token overlap checker | 4 | ✅ |
| Math structural checker | 4 | ✅ |
| Code structural checker | 4 | ✅ |
| Semantic similarity | 2 | ✅ |
| Combined firewall | 6 | ✅ |
| Convenience functions | 2 | ✅ |
| Edge cases | 4 | ✅ |
| Benchmark configs | 5 | ✅ |
| Loader init | 2 | ✅ |
| Mock loading | 8 | ✅ |
| Error handling | 2 | ✅ |
| Method libraries | 8 | ✅ |
| ICR transformer | 5 | ✅ |
| Transform single | 5 | ✅ |
| Transform batch | 5 | ✅ |
| All benchmarks | 11 | ✅ |
| Sparse logits | 5 | ✅ |
| Cross-tokenizer aligner | 4 | ✅ |
| Logit aggregator | 4 | ✅ |
| Vocabulary mapper | 3 | ✅ |
| GKD loss | 4 | ✅ |
| Task queue basics | 5 | ✅ |
| Claim and submit | 6 | ✅ |
| Worker concerns | 5 | ✅ |
| Queue status | 6 | ✅ |
| Retry and progress | 3 | ✅ |
| Debug mode | 1 | ✅ |
| Persistence | 3 | ✅ |
| Serialization | 2 | ✅ |
| Convenience functions | 1 | ✅ |
| Queue types | 5 | ✅ |
| List tasks | 4 | ✅ |
| Concern summary | 1 | ✅ |

---

## Files Created

```
src/phfe/
├── __init__.py                      # Package root
├── cli.py                           # Typer CLI
├── orchestrator/
│   ├── __init__.py
│   ├── observability.py             # Tracing infrastructure
│   ├── tutor.py                     # Multi-model caller
│   └── task_queue.py                # 🆕 Task queue orchestrator
├── benchmark/
│   ├── __init__.py                  # Benchmark types, exports
│   ├── contamination.py             # Contamination firewall
│   └── loader.py                    # HuggingFace loader
├── icr_transform/
│   ├── __init__.py                  # ICR transformer
│   └── method_libraries.py          # All benchmark method libraries
├── distillation/
│   ├── __init__.py                  # GKD training stubs
│   └── cross_tokenizer.py           # Cross-tokenizer alignment
├── answer_key_corpus/
│   └── __init__.py                  # Answer key collection stubs
├── arxiv_pipeline/
│   └── __init__.py                  # LaTeX pipeline stubs
├── curriculum_generator/
│   └── __init__.py                  # Problem generation stubs
└── language_evals/
    └── __init__.py                  # Language competence stubs

.claude/agents/                       # 🆕 Claude Code agent definitions
├── problem-generator.md             # Synthetic problem generation
├── answer-verifier.md               # Answer verification (Opus)
└── contamination-reviewer.md        # Edge-case contamination review

tests/
├── __init__.py
├── test_contamination.py            # 37 tests
├── test_benchmark_loader.py         # 23 tests
├── test_icr_transform.py            # 39 tests
├── test_cross_tokenizer.py          # 21 tests
└── test_task_queue.py               # 🆕 42 tests

scripts/
└── cross_tokenizer_testbed.py       # GKD testbed with real models

claudefiles/
└── task_orchestrator_spec.md        # 🆕 Task orchestrator specification
```

---

## Next Steps — Suggested Priorities

### Option A: Complete Tutor Inference (vLLM)
1. **vLLM Setup** — Configure vLLM for logprob extraction
2. **Logit Capture** — Extract top-p sparse logits during generation
3. **Multi-tutor Pipeline** — Route to different teachers based on domain

### Option B: Cross-Tokenizer GKD Testbed
1. **Load Qwen-1B and Gemma-4B** — Download and configure
2. **Alignment Test** — Verify cross-tokenizer alignment works
3. **Loss Measurement** — Compute GKD loss without gradients

### Option C: Synthetic Generation Pipeline
1. **Problem Generator** — Use tutors to generate synthetic problems
2. **Generation + Firewall Loop** — Generate, check, accept/reject
3. **Answer Key Collection** — Multi-tutor answer keys with logits

---

## Reference Implementation

The `reference_dialogue_yoinker/` directory contains patterns from a similar subagent orchestrator project:
- `subagent_orchestrator/subagent.py` — API caller pattern
- `subagent_orchestrator/observability.py` — Trace storage pattern
- `api_server.py` — FastAPI dispatch example

---

## Commands

```bash
# Sync dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_cross_tokenizer.py -v

# Check project status
uv run phfe status

# List benchmarks
uv run phfe benchmark list

# Load a benchmark (example)
uv run python -c "from phfe.benchmark import load_benchmark, BenchmarkType; print(load_benchmark(BenchmarkType.GSM1K, limit=3))"
```
