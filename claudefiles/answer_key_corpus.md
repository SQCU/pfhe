# Claudefile: Answer Key Corpus — Multi-Tutor Collection Strategy

## Purpose

This spec describes how to collect the answer key corpus for PFE synthetic training. The goals are:

1. **High yield**: >90% of generated answer keys are correct and usable
2. **Style diversity**: No single tutor's voice dominates the corpus
3. **Contamination-free**: All synthetic problems pass the firewall (see `pfe_benchmark_spec.md`)
4. **Full provenance**: Every answer key is traceable to its tutor, with tokens and logits preserved

## The Tutor Ensemble

We collect answer keys from multiple tutors to avoid stylistic monoculture.

### Primary Tutors (Use for All Benchmarks)

| Model | Access | Strengths | Style Notes |
|-------|--------|-----------|-------------|
| **DeepSeek-R1** | Weights | Strong reasoning, verbose traces | Distinctive "thinking out loud" style |
| **Kimi K2** | Weights | Deep (61 layers), good math | More concise than DeepSeek |
| **Qwen-72B-Chat** | Weights | Strong all-around | Moderate verbosity |
| **GPT-4o** | API | High accuracy, good style | The "default" LLM voice |
| **Claude-3.5-Sonnet** | API | Clear explanations | Structured, sometimes verbose |

### Backup Tutors (Use if Primary Unavailable)

| Model | Access | Notes |
|-------|--------|-------|
| **Gemma-2-27B-IT** | Weights | Google lineage, different training |
| **Mistral-Large** | API | European, different tics |
| **Llama-3.1-70B-Instruct** | Weights | Meta baseline |

### Allocation Strategy

For each synthetic problem, collect answer keys from **at least 3 tutors**:
- 1 from DeepSeek/Kimi/Qwen (open weights, reasoning-focused)
- 1 from GPT-4o or Claude (API, high quality)
- 1 from any other tutor (diversity)

This ensures:
- No single tutor > 40% of any benchmark's answer keys
- Mix of open-weights and API models
- Mix of verbose and concise styles

---

## Collection Pipeline

### Phase 0: Canonical Benchmark Ingestion

Before generating anything, load and index all canonical benchmark problems.

```python
class CanonicalIndex:
    """Index of all canonical problems for contamination checking."""
    
    def __init__(self, benchmarks: list[str]):
        self.problems = {}  # benchmark -> list of problems
        self.embeddings = {}  # benchmark -> embedding matrix
        self.ngrams = {}  # benchmark -> set of n-grams
        
        for bench in benchmarks:
            problems = load_canonical_benchmark(bench)
            self.problems[bench] = problems
            self.embeddings[bench] = embed_all([p.text for p in problems])
            self.ngrams[bench] = extract_all_ngrams([p.text for p in problems], n=5)
    
    def check_contamination(self, synthetic: Problem, benchmark: str) -> tuple[bool, str]:
        """Returns (is_safe, reason)."""
        # Implementation in pfe_benchmark_spec.md
        ...
```

**Critical**: This index is built once and never modified. It's the reference for all contamination checks.

### Phase 1: Synthetic Problem Generation

Generate candidate synthetic problems using tutor models.

```python
def generate_synthetic_problems(
    benchmark: str,
    target_count: int,
    canonical_index: CanonicalIndex,
    tutor_model: str,
) -> list[SyntheticProblem]:
    """
    Generate synthetic problems similar to a benchmark.
    
    Returns only problems that pass contamination firewall.
    """
    
    prompt_template = get_generation_prompt(benchmark)
    # e.g., "Generate a grade school math word problem similar to GSM8K.
    #        It should require 2-5 steps and use only basic arithmetic.
    #        Do not copy any existing problem."
    
    accepted = []
    rejected_counts = {"token_overlap": 0, "semantic": 0, "structural": 0}
    
    # Over-generate to account for rejections
    candidates_needed = int(target_count * 1.5)
    
    for _ in range(candidates_needed):
        if len(accepted) >= target_count:
            break
        
        # Generate candidate
        candidate = tutor_generate(tutor_model, prompt_template)
        
        # Parse into structured format
        problem = parse_problem(candidate, benchmark)
        
        # Contamination check
        is_safe, reason = canonical_index.check_contamination(problem, benchmark)
        
        if is_safe:
            accepted.append(problem)
        else:
            rejected_counts[reason] += 1
    
    # Log rejection stats (this is the contamination meta-eval!)
    log_rejection_stats(tutor_model, benchmark, rejected_counts, len(accepted))
    
    return accepted
```

**Key output**: Rejection statistics per tutor. High rejection rate suggests tutor has memorized the canonical benchmark.

### Phase 2: Answer Key Generation

For each accepted synthetic problem, generate answer keys from multiple tutors.

```python
def generate_answer_keys(
    problem: SyntheticProblem,
    tutor_ensemble: list[str],
    tutors_per_problem: int = 3,
) -> list[AnswerKey]:
    """
    Generate answer keys from multiple tutors for a single problem.
    """
    
    answer_keys = []
    
    # Select tutors for this problem (rotate to ensure diversity)
    selected_tutors = select_tutors(tutor_ensemble, tutors_per_problem)
    
    for tutor in selected_tutors:
        # Generate with logit logging
        prompt = format_problem_for_tutor(problem, tutor)
        
        response = tutor_generate_with_logits(
            model=tutor,
            prompt=prompt,
            max_tokens=1024,
            top_p_storage=0.95,  # Store top-95% probability mass
        )
        
        # Parse and verify
        parsed = parse_answer_key(response.text, problem)
        
        if parsed.answer is not None:
            # Verify correctness (domain-specific)
            is_correct = verify_answer(parsed.answer, problem)
            
            answer_key = AnswerKey(
                problem_id=problem.id,
                tutor_model=tutor,
                
                # The answer key content
                text=response.text,
                tokens=response.token_ids,
                logits=response.sparse_logits,
                
                # Parsed structure
                reasoning_trace=parsed.reasoning,
                final_answer=parsed.answer,
                
                # Verification
                verified_correct=is_correct,
                
                # Provenance
                generation_timestamp=now(),
                generation_config=response.config,
            )
            
            answer_keys.append(answer_key)
    
    return answer_keys
```

### Phase 3: Quality Filtering

Filter answer keys for correctness and diversity.

```python
def filter_answer_keys(
    answer_keys: list[AnswerKey],
    problem: SyntheticProblem,
) -> list[AnswerKey]:
    """
    Filter to keep only correct, diverse answer keys.
    """
    
    # Keep only verified correct
    correct = [ak for ak in answer_keys if ak.verified_correct]
    
    if len(correct) == 0:
        # All tutors got it wrong — flag problem as potentially bad
        flag_problem_for_review(problem)
        return []
    
    if len(correct) == 1:
        # Only one correct — keep it but note low confidence
        correct[0].confidence = "low"
        return correct
    
    # Multiple correct — check for agreement
    answers = [ak.final_answer for ak in correct]
    if len(set(answers)) > 1:
        # Tutors disagree on answer — flag for review
        flag_problem_for_review(problem, reason="tutor_disagreement")
        # Keep majority answer
        majority_answer = most_common(answers)
        correct = [ak for ak in correct if ak.final_answer == majority_answer]
    
    # All remaining are correct and agree
    for ak in correct:
        ak.confidence = "high"
    
    return correct
```

### Phase 4: Corpus Assembly

Assemble the final corpus with balanced tutor representation.

```python
def assemble_corpus(
    all_answer_keys: dict[str, list[AnswerKey]],  # problem_id -> answer keys
    target_per_benchmark: int,
    tutor_balance_threshold: float = 0.4,  # No tutor > 40%
) -> AnswerKeyCorpus:
    """
    Assemble final corpus with tutor balance constraints.
    """
    
    corpus = AnswerKeyCorpus()
    
    for problem_id, answer_keys in all_answer_keys.items():
        if not answer_keys:
            continue
        
        # Select answer keys to include (balance tutors)
        selected = select_balanced(
            answer_keys,
            corpus.tutor_counts,
            tutor_balance_threshold,
        )
        
        for ak in selected:
            corpus.add(ak)
    
    # Final stats
    corpus.compute_statistics()
    
    return corpus

def select_balanced(
    candidates: list[AnswerKey],
    current_counts: dict[str, int],
    threshold: float,
) -> list[AnswerKey]:
    """
    Select answer keys while maintaining tutor balance.
    """
    
    total = sum(current_counts.values())
    
    selected = []
    for ak in candidates:
        tutor = ak.tutor_model
        
        # Would adding this exceed threshold?
        new_count = current_counts.get(tutor, 0) + 1
        new_total = total + 1
        
        if new_count / new_total <= threshold:
            selected.append(ak)
            # Don't actually update counts here — let caller do it
    
    # If all would exceed threshold, take the one from least-represented tutor
    if not selected and candidates:
        least_represented = min(candidates, key=lambda ak: current_counts.get(ak.tutor_model, 0))
        selected = [least_represented]
    
    return selected
```

---

## Data Format

### AnswerKey Schema

```python
@dataclass
class AnswerKey:
    # === Identity ===
    id: str                          # Unique ID
    problem_id: str                  # Links to SyntheticProblem
    
    # === Content ===
    text: str                        # Full text of answer key
    tokens: list[int]                # Token IDs (tutor's tokenizer)
    logits: list[SparseLogits]       # Top-p logits per position
    
    # === Parsed Structure ===
    reasoning_trace: str             # The reasoning/explanation part
    final_answer: str                # The final answer
    
    # === Verification ===
    verified_correct: bool           # Did answer match ground truth?
    confidence: str                  # "high", "medium", "low"
    
    # === Provenance ===
    tutor_model: str                 # Which tutor generated this
    tutor_tokenizer: str             # Tokenizer identifier
    generation_timestamp: datetime
    generation_config: dict          # Temperature, top_p, etc.
    
    # === For GKD ===
    def get_distillation_target(self, student_tokenizer) -> DistillationTarget:
        """Convert to format needed for GKD training."""
        if self.tutor_tokenizer == student_tokenizer:
            # Same tokenizer — direct use
            return DistillationTarget(
                tokens=self.tokens,
                logits=self.logits,
            )
        else:
            # Cross-tokenizer alignment needed
            return align_tokenizations(
                text=self.text,
                tutor_tokenizer=self.tutor_tokenizer,
                student_tokenizer=student_tokenizer,
                tutor_logits=self.logits,
            )
```

### SyntheticProblem Schema

```python
@dataclass
class SyntheticProblem:
    # === Identity ===
    id: str
    benchmark: str                   # "gsm1k", "arc", etc.
    
    # === Content ===
    text: str                        # Problem statement
    
    # === Ground Truth ===
    answer: str                      # Correct answer
    answer_type: str                 # "number", "multiple_choice", "code", etc.
    
    # === ICR Context ===
    icr_context: str                 # Method library / worked examples
    
    # === Contamination Check ===
    contamination_check_passed: bool
    check_timestamp: datetime
    
    # === Generation Provenance ===
    generator_model: str             # Which tutor generated the problem
    generation_timestamp: datetime
    
    # === Metadata ===
    difficulty_estimate: float       # 0-1
    num_steps: int                   # For math problems
    tags: list[str]                  # e.g., ["addition", "multi-step"]
```

### Corpus Statistics

```python
@dataclass
class CorpusStatistics:
    # === Size ===
    total_problems: int
    total_answer_keys: int
    answer_keys_per_problem: float   # Average
    
    # === By Benchmark ===
    problems_per_benchmark: dict[str, int]
    
    # === Tutor Distribution ===
    answer_keys_per_tutor: dict[str, int]
    tutor_percentages: dict[str, float]
    max_tutor_percentage: float      # Should be < 40%
    
    # === Quality ===
    verification_rate: float         # % of answer keys verified correct
    tutor_agreement_rate: float      # % of problems where tutors agree
    flagged_for_review: int          # Problems with issues
    
    # === Contamination ===
    rejection_rate_by_tutor: dict[str, float]  # Meta-eval of tutors
    total_rejected: int
    rejection_reasons: dict[str, int]
```

---

## Storage Format

### Directory Structure

```
answer_key_corpus/
├── metadata.json                    # Corpus statistics, generation config
├── canonical_index/                 # Cached canonical benchmark data
│   ├── gsm1k_index.pkl
│   ├── arc_index.pkl
│   └── ...
├── synthetic_problems/
│   ├── gsm1k_synth.parquet
│   ├── arc_synth.parquet
│   └── ...
├── answer_keys/
│   ├── gsm1k_keys.parquet
│   ├── arc_keys.parquet
│   └── ...
├── icr_contexts/
│   ├── gsm1k_methods.txt
│   ├── arc_facts.txt
│   └── ...
└── logs/
    ├── generation_log.jsonl         # Full generation trace
    ├── rejection_log.jsonl          # Contamination rejections
    └── review_queue.jsonl           # Problems flagged for review
```

### Parquet Schema for Answer Keys

```python
answer_keys_schema = pa.schema([
    ("id", pa.string()),
    ("problem_id", pa.string()),
    
    # Content
    ("text", pa.string()),
    ("tokens", pa.list_(pa.int32())),
    ("logits", pa.list_(pa.struct([
        ("token_ids", pa.list_(pa.int32())),
        ("logit_values", pa.list_(pa.float16())),
        ("coverage", pa.float32()),
    ]))),
    
    # Parsed
    ("reasoning_trace", pa.string()),
    ("final_answer", pa.string()),
    
    # Verification
    ("verified_correct", pa.bool_()),
    ("confidence", pa.string()),
    
    # Provenance
    ("tutor_model", pa.string()),
    ("tutor_tokenizer", pa.string()),
    ("generation_timestamp", pa.timestamp("us")),
    ("generation_config", pa.string()),  # JSON
])
```

---

## Tutor Contamination Meta-Eval

The rejection log provides a free measurement of tutor contamination.

### Metrics

For each tutor T and benchmark B:

```
rejection_rate(T, B) = rejected_problems(T, B) / attempted_problems(T, B)
```

### Interpretation

| Rejection Rate | Interpretation |
|----------------|----------------|
| < 5% | Tutor generalizes well, minimal contamination |
| 5-15% | Some contamination or limited creativity |
| 15-30% | Significant contamination concerns |
| > 30% | Tutor likely memorized benchmark |

### Reporting

```
Contamination Meta-Eval: GSM1K Synthetic Generation
───────────────────────────────────────────────────
Tutor              Attempted   Rejected   Rate    Reason Breakdown
───────────────────────────────────────────────────
DeepSeek-R1           2,500       125    5.0%    token: 2%, semantic: 2%, struct: 1%
Kimi K2               2,500       175    7.0%    token: 3%, semantic: 3%, struct: 1%
GPT-4o                2,500       225    9.0%    token: 4%, semantic: 4%, struct: 1%
Claude-3.5-Sonnet     2,500       200    8.0%    token: 3%, semantic: 4%, struct: 1%
Qwen-72B              2,500       150    6.0%    token: 2%, semantic: 3%, struct: 1%
───────────────────────────────────────────────────
```

This is valuable data about tutor models that we get "for free" from the corpus generation process.

---

## CLI Interface

```bash
# Build canonical index (do this first, once)
python -m answer_key_corpus build-index \
    --benchmarks gsm1k,arc,race,boolq,mbpp \
    --output-dir ./corpus/canonical_index/

# Generate synthetic problems for one benchmark
python -m answer_key_corpus generate-problems \
    --benchmark gsm1k \
    --target-count 10000 \
    --tutor-model deepseek-r1 \
    --canonical-index ./corpus/canonical_index/ \
    --output-dir ./corpus/synthetic_problems/

# Generate answer keys for synthetic problems
python -m answer_key_corpus generate-keys \
    --problems ./corpus/synthetic_problems/gsm1k_synth.parquet \
    --tutors deepseek-r1,kimi-k2,gpt-4o,claude-3.5-sonnet,qwen-72b \
    --keys-per-problem 3 \
    --output-dir ./corpus/answer_keys/

# Assemble final corpus with balance constraints
python -m answer_key_corpus assemble \
    --answer-keys ./corpus/answer_keys/ \
    --tutor-balance-threshold 0.4 \
    --output-dir ./corpus/final/

# Generate corpus statistics report
python -m answer_key_corpus stats \
    --corpus ./corpus/final/ \
    --output ./corpus/report.md
```

---

## Quality Assurance

### Automated Checks

1. **Correctness verification**: Domain-specific answer checking
2. **Tutor agreement**: Flag problems where tutors disagree
3. **Balance constraints**: Ensure no tutor dominates
4. **Contamination firewall**: Reject overlapping problems

### Manual Review Queue

Problems flagged for human review:
- All tutors got wrong answer
- Tutors disagree on answer
- Unusually high similarity to canonical (near threshold)
- Malformed generation (parsing failed)

### Periodic Audits

- Sample 100 random answer keys per benchmark
- Human verification of correctness
- Human evaluation of style diversity
- Check for subtle contamination patterns

---

## Success Criteria

1. **Yield**: >90% of generated problems pass contamination firewall
2. **Correctness**: >95% of answer keys verified correct
3. **Agreement**: >90% of problems have tutor agreement on answer
4. **Balance**: No tutor > 40% of any benchmark's answer keys
5. **Size**: Target counts met for all benchmarks

---

## Dependencies

```
# Tutor inference
vllm                   # Local inference
openai                 # GPT-4o API
anthropic              # Claude API

# Embeddings (for semantic similarity)
sentence-transformers

# Data handling
pyarrow
pandas
numpy

# Verification
sympy                  # Math verification
docker                 # Code execution sandboxing
```

---

When complete, this corpus provides:
- ~39,000 synthetic training problems (verified non-overlapping with canonical)
- ~100,000+ answer keys from diverse tutors (with logits for GKD)
- Contamination meta-eval data on all tutor models
- Full provenance for every piece of data
