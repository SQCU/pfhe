# Claudefile: PROJECT INDEX — PFE (Posthumanity's First Exam)

## One-Line Summary

Train a language model that demonstrates **in-context learning** rather than memorization, measured by a benchmark suite with **provably separate** training and evaluation data.

## The Core Claim

> "A model trained on synthetic problems (verified non-overlapping with canonical benchmarks) achieves X% on canonical eval and Y% on ICR-augmented eval. The gap (Y - X) represents pure in-context learning capability, not contamination or memorization."

This is a clean scientific claim because:
1. **Training data is synthetic**: Generated fresh, not scraped
2. **Contamination firewall**: Every training example is verified distinct from eval
3. **ICR augmentation**: Same eval problems, with/without method library context
4. **Multi-tutor diversity**: No single model's style dominates training

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CANONICAL BENCHMARKS                         │
│                   (GSM1K, ARC, RACE, MBPP, etc.)                    │
│                           NEVER TOUCHED                             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
         ▼                                                   ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│  SYNTHETIC GENERATION   │                   │   ICR AUGMENTATION      │
│                         │                   │                         │
│  • Generate problems    │                   │  • Add method libraries │
│  • Contamination check  │                   │  • Add worked examples  │
│  • Multi-tutor keys     │                   │  • Same eval problems   │
└───────────┬─────────────┘                   └───────────┬─────────────┘
            │                                             │
            ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│    TRAINING CORPUS      │                   │    EVALUATION SUITE     │
│                         │                   │                         │
│  • Synthetic problems   │                   │  • Canonical (no ctx)   │
│  • Answer keys + logits │                   │  • ICR (with ctx)       │
│  • ICR contexts         │                   │  • Compare the gap      │
└───────────┬─────────────┘                   └───────────┬─────────────┘
            │                                             │
            ▼                                             │
┌─────────────────────────┐                              │
│    GKD TRAINING         │                              │
│                         │                              │
│  • Offline distillation │                              │
│  • Multi-tutor logits   │                              │
│  • Style averaging      │                              │
└───────────┬─────────────┘                              │
            │                                             │
            └──────────────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   TRAINED MODEL     │
                    │                     │
                    │  Evaluate on both   │
                    │  canonical and ICR  │
                    └─────────────────────┘
```

---

## Claudefile Index

### Core Specifications (The Main Loop)

| File | Purpose | Status |
|------|---------|--------|
| **`pfe_benchmark_spec.md`** | The benchmark suite: what's in, what's out, three-split architecture, contamination firewall | ✅ Complete |
| **`answer_key_corpus.md`** | Multi-tutor answer key collection, rejection filtering, style diversity | ✅ Complete |
| **`offline_distillation_protocol.md`** | Data format (tokens + sparse logits), GKD training loop, cross-tokenizer alignment | ✅ Complete |

### Supporting Specifications

| File | Purpose | Status |
|------|---------|--------|
| **`benchmark_icr_transform.md`** | General methodology for ICR transformation (referenced by pfe_benchmark_spec) | ✅ Complete |
| **`arxiv_tex_pipeline.md`** | LaTeX → multimodal training data (Stage 3 / domain depth) | ✅ Complete |

### Deprecated / Reference

| File | Purpose | Status |
|------|---------|--------|
| **`curriculum_generator.md`** | Procedural problem generation (superseded by answer_key_corpus for most uses) | Superseded |
| **`language_eval_suite.md`** | Syntax/discourse metrics (partially superseded, still useful for sanity checks) | Partial |

---

## PFE Benchmark Suite

### Included Benchmarks

| Benchmark | Type | Canonical Size | ICR Augmentation |
|-----------|------|----------------|------------------|
| **GSM1K** | Math | 1,250 | 5-method arithmetic library |
| **ARC-Easy** | Science | 2,376 | Science fact library |
| **ARC-Challenge** | Science | 1,172 | Science fact library |
| **RACE** | Reading | 4,934 | Strategy hints |
| **BoolQ** | Boolean QA | 3,270 | (passage already provided) |
| **HellaSwag** | Commonsense | 10,042 | Setup paragraphs |
| **WinoGrande** | Coreference | 1,267 | Explicit backstory |
| **MBPP** | Code | 500 | Pattern library |
| **FnCall** | Function calling | 500 | API docs + examples |
| **Format** | Data transform | 500 | Input/output examples |

**Total canonical eval: ~25,000 problems**

### Excluded Benchmarks (Complexity Not Proportionate)

| Benchmark | Reason |
|-----------|--------|
| MATH-500 | Requires creative insight, not method application |
| GPQA | PhD-level, context construction prohibitively expensive |
| Codeforces | Algorithmic creativity beyond pattern application |
| MMLU-professional | Domain expertise required for context construction |

### Synthetic Training Corpus

| Synthetic Set | Target Size | Source |
|---------------|-------------|--------|
| GSM1K-Synth | 10,000 | Multi-tutor generation |
| ARC-Synth | 5,000 | Multi-tutor generation |
| RACE-Synth | 5,000 | Multi-tutor generation |
| BoolQ-Synth | 5,000 | Multi-tutor generation |
| HellaSwag-Synth | 5,000 | Multi-tutor generation |
| WinoGrande-Synth | 3,000 | Multi-tutor generation |
| MBPP-Synth | 2,000 | Multi-tutor generation |
| FnCall-Synth | 2,000 | Multi-tutor generation |
| Format-Synth | 2,000 | Multi-tutor generation |

**Total synthetic training: ~39,000 problems**
**With multi-tutor answer keys: ~100,000+ distillation targets**

---

## Tutor Ensemble

| Model | Access | Role |
|-------|--------|------|
| DeepSeek-R1 | Weights | Reasoning traces, distinctive style |
| Kimi K2 | Weights | Deep model, math strength |
| Qwen-72B-Chat | Weights | General capability |
| GPT-4o | API | High accuracy baseline |
| Claude-3.5-Sonnet | API | Clear explanations |

**Constraint**: No single tutor > 40% of answer keys for any benchmark.

---

## Key Metrics

### Per-Benchmark

| Metric | Definition |
|--------|------------|
| **Canonical Accuracy** | % correct on benchmark, no context |
| **ICR Accuracy** | % correct on benchmark, with method library |
| **ICR Lift** | ICR Accuracy - Canonical Accuracy |
| **Relative ICR Lift** | ICR Lift / (100 - Canonical Accuracy) |

### Aggregate (PFE Scores)

| Metric | Definition |
|--------|------------|
| **PFE-Canonical** | Average canonical accuracy across benchmarks |
| **PFE-ICR** | Average ICR accuracy across benchmarks |
| **PFE-Lift** | Average ICR lift across benchmarks |

### Comparison Points

- Our trained model vs. base (random init)
- Our trained model vs. existing models (Llama, Mistral, Phi)
- Our trained model vs. tutors (ceiling)
- ICR vs. canonical for all models (who benefits from context?)

---

## Success Criteria

### Scientific Success

1. **Clean separation**: All training data passes contamination firewall
2. **Measurable ICR lift**: Trained model shows >10% average ICR lift
3. **Transfer demonstrated**: Model solves canonical problems it was never trained on
4. **Tutor diversity held**: No stylistic monoculture in training data

### Practical Success

1. **Corpus generated**: ~39K synthetic problems, ~100K answer keys
2. **Training completed**: Model trained via GKD on corpus
3. **Evaluation completed**: Full PFE benchmark suite evaluated
4. **Report generated**: Results, comparisons, analysis

---

## Implementation Phases

### Phase 1: Infrastructure (Weeks 1-2)

- [ ] Build canonical benchmark index
- [ ] Implement contamination firewall
- [ ] Set up tutor inference (vLLM for weights, API clients)
- [ ] Implement answer key logging (tokens + logits)

### Phase 2: Corpus Generation (Weeks 3-4)

- [ ] Generate synthetic problems (with rejection logging)
- [ ] Collect answer keys from tutor ensemble
- [ ] Quality filtering and verification
- [ ] Assemble final corpus

### Phase 3: Training (Week 5)

- [ ] Implement GKD training loop
- [ ] Train student model on corpus
- [ ] Checkpoint and log

### Phase 4: Evaluation (Week 6)

- [ ] Evaluate on canonical benchmarks
- [ ] Evaluate on ICR-augmented benchmarks
- [ ] Compare to baselines
- [ ] Generate report

---

## Bonus Outputs

Beyond the trained model, this project produces:

1. **Contamination meta-eval**: Which tutors have memorized which benchmarks?
2. **ICR methodology**: Reusable approach for transforming benchmarks
3. **Synthetic corpus**: Can be released for others to use
4. **Style diversity data**: How different are the tutors, really?

---

## For the Implementing Claude

If you're picking up a component:

1. **Read the relevant spec** (pfe_benchmark_spec, answer_key_corpus, offline_distillation_protocol)
2. **Respect the contamination firewall** — this is the scientific credibility
3. **Log everything** — provenance matters
4. **Test on small scale first** — generate 100 before 10,000
5. **Flag blockers** — some specs may have gaps

The goal is a **clean scientific result**, not a SOTA model. The interesting question is "does ICR help?" not "how high can we score?"

---

## Contact / Context

These specs were generated in conversation on 2024-12-22. Context:
- https://sqcu.dev/texts/items/fall_of_ml2025.html
- https://github.com/SQCU/logsnrcat
- https://arxiv.org/abs/2405.00332 (GSM1K paper)
- https://arxiv.org/abs/2306.13649 (GKD paper)

For questions not answered in specs: do the simple thing, document your choice, move on.
