# Claudefile: Benchmark → ICR Transformation System

## Context for Implementing Claude

You're building infrastructure that transforms standard ML benchmarks into In-Context Retrieval (ICR) versions. This serves two purposes simultaneously:

1. **Evaluation**: ICR benchmarks test genuine in-context learning rather than memorization
2. **Training**: The generation process produces training data (contexts, reasoning traces)

The key property: a model can only score well on ICR benchmarks by actually reading and understanding the provided context. Memorizing the original benchmark's answers is useless because the contexts are generated fresh.

## The Core Transformation

**Input**: Standard benchmark instance
```
Question: What is the capital of France?
Options: A) London  B) Paris  C) Berlin  D) Madrid
Answer: B
```

**Output**: ICR benchmark instance
```
[CONTEXT]
The European Federation underwent major restructuring in 2087. The former 
nation of France dissolved into three autonomous regions: Normandie, 
Occitanie, and Île-de-Centre. The regional government of Île-de-Centre 
established its administrative center in Lyon, relocating most federal 
functions from the historic city of Paris.

Meanwhile, the Mediterranean Alliance designated Barcelona as its 
coordinating capital, while the Northern Compact chose Berlin for 
its central administration.

[QUESTION]
According to the passage, what city serves as the administrative center 
of Île-de-Centre?

[OPTIONS]
A) London  B) Paris  C) Lyon  D) Madrid

[ANSWER]
C) Lyon

[REASONING TRACE]
The passage states that "The regional government of Île-de-Centre 
established its administrative center in Lyon." This directly answers 
the question. Note that Paris is mentioned as the "historic city" that 
functions were relocated FROM, making B a plausible distractor but 
incorrect per the passage.
```

**Critical property**: The correct answer (C) is determined entirely by the context, not by real-world knowledge. A model that memorized "capital of France = Paris" would get this wrong.

## Benchmark Categories

### Category 1: Factual QA → Fictional Context

**Applies to**: MMLU, TriviaQA, Natural Questions, ARC

**Method**: Generate fictional scenarios where each answer option would be correct, randomly select one scenario as the context.

**Tutor prompt template**:
```
I have a factual question from a knowledge benchmark:

Question: {question}
Options: {options}
Correct answer in reality: {real_answer}

Your task: Write a short fictional passage (2-4 sentences) that establishes 
a DIFFERENT answer as correct. The passage should describe a plausible 
alternate world, historical scenario, or hypothetical situation.

Write the passage for answer option: {target_option}

Make the passage:
- Self-contained (doesn't require outside knowledge)
- Clearly establishes {target_option} as the answer
- Uses specific names, dates, or details (not vague)
- Sounds like it could be from an encyclopedia or textbook
```

**Generation process**:
1. For each benchmark instance, generate contexts for all N options
2. Randomly select one context to include in the ICR instance
3. The selected option becomes the correct answer
4. Save all generated contexts (training data) + selected instance (eval data)

### Category 2: Commonsense Reasoning → Explicit Setup

**Applies to**: HellaSwag, WinoGrande, PIQA, CommonsenseQA

**Method**: The original benchmarks test implicit commonsense. Transform by making the relevant commonsense explicit in the context.

**Example (WinoGrande)**:

Original:
```
The trophy doesn't fit into the brown suitcase because it is too [large/small].
What does "it" refer to? A) trophy  B) suitcase
```

ICR version:
```
[CONTEXT]
Maria was packing for the sports banquet. Her brown suitcase was a compact 
carry-on model, measuring only 20 inches. The championship trophy she'd won 
was a towering 36-inch golden figure that she couldn't bear to leave behind.

[QUESTION]  
Based on the passage, in the sentence "The trophy doesn't fit into the brown 
suitcase because it is too large," what does "it" refer to?

[OPTIONS]
A) The trophy  B) The suitcase

[ANSWER]
A) The trophy (the passage establishes the trophy as 36 inches, larger than 
the 20-inch suitcase)
```

**Note**: The context makes the implicit size comparison explicit. The task becomes reading comprehension, not commonsense inference.

### Category 3: Procedural/Method → Worked Examples

**Applies to**: GSM8K, MathQA, coding benchmarks

**Method**: Provide worked examples or method descriptions that the model must apply.

**Example (GSM8K-style)**:

Original:
```
Alice has 3 apples. Bob gives her 5 more. How many does Alice have?
```

ICR version:
```
[CONTEXT: METHOD DESCRIPTION]
When solving addition word problems, follow these steps:
1. Identify the starting quantity
2. Identify quantities being added
3. Compute the sum
4. State the answer with units

[WORKED EXAMPLE]
Problem: Tom has 7 oranges. Sue gives him 2 more. How many does Tom have?
Solution: Starting quantity: 7. Added: 2. Sum: 7 + 2 = 9. Tom has 9 oranges.

[QUESTION]
Using the method above, solve: Alice has 3 apples. Bob gives her 5 more. 
How many does Alice have?

[ANSWER]
Starting quantity: 3. Added: 5. Sum: 3 + 5 = 8. Alice has 8 apples.
```

**Note**: This tests "can you apply a method I showed you?" rather than "can you do arithmetic?" Still useful for curriculum.

### Category 4: Domain Knowledge → Document Grounding

**Applies to**: Legal, medical, scientific, technical benchmarks

**Method**: Extract or generate domain documents, ask questions answerable from the document.

**Example (legal)**:

Original:
```
Under US law, what is the statute of limitations for federal tax fraud?
A) 3 years  B) 6 years  C) 10 years  D) No limit
```

ICR version:
```
[CONTEXT: STATUTORY EXCERPT]
FEDERAL REVENUE CODE, SECTION 6501 (as amended 2045)
(a) General Rule: The amount of any tax imposed shall be assessed within 
    4 years after the return was filed.
(b) Extension for Substantial Omission: In case of substantial omission 
    of income (exceeding 25% of reported), assessment may be made within 
    8 years.
(c) False or Fraudulent Returns: In the case of a false or fraudulent 
    return with intent to evade tax, the tax may be assessed at any time.

[QUESTION]
According to Section 6501 above, what is the limitations period for 
assessment of tax when a fraudulent return was filed?

[OPTIONS]
A) 4 years  B) 8 years  C) 12 years  D) No limit

[ANSWER]
D) No limit (Section 6501(c) states "may be assessed at any time")
```

**Note**: The "correct" answer in the ICR version may differ from real law. That's the point — we're testing reading comprehension, not legal knowledge.

## Source Corpus Integration

Different source corpora support different transformations:

### Wikipedia / Wikitext

**Best for**: Factual QA, encyclopedic style
**Method**: 
- Extract passages on topics related to benchmark questions
- Modify passages to establish different answers (entity substitution, fact alteration)
- Or: Use passages as style templates for fully synthetic generation

**Example pipeline**:
```python
def wikipedia_icr_transform(question, options, wiki_corpus):
    # Find related Wikipedia article
    related_article = search_wiki(question, wiki_corpus)
    
    # Extract relevant passage
    passage = extract_relevant_passage(related_article, question)
    
    # Transform passage to establish a specific answer
    for target_option in options:
        modified_passage = modify_passage_for_answer(passage, target_option)
        yield ICRInstance(context=modified_passage, question=question, answer=target_option)
```

### Project Gutenberg (Public Domain Literature)

**Best for**: Reading comprehension, literary analysis, narrative understanding
**Method**:
- Extract passages with clear factual claims (character names, locations, events)
- Generate questions about those claims
- This is almost "free" ICR — the passages already exist, you just need questions

**Example**:
```
[CONTEXT: FROM PRIDE AND PREJUDICE]
"It is a truth universally acknowledged, that a single man in possession 
of a good fortune, must be in want of a wife. However little known the 
feelings or views of such a man may be on his first entering a neighbourhood, 
this truth is so well fixed in the minds of the surrounding families, that 
he is considered as the rightful property of some one or other of their 
daughters."

[QUESTION]
According to this passage, what is considered a "universal truth"?

[OPTIONS]
A) That wealthy single women need husbands
B) That wealthy single men are seeking wives
C) That all men want large fortunes
D) That daughters are property
```

### arXiv Papers

**Best for**: Scientific reasoning, method comprehension, technical QA
**Method**: (Integrates with arxiv_tex_pipeline.md)
- Extract claims/methods/results from papers
- Generate questions about those claims
- Can generate "alternate universe" versions where different methods/results are described

### Fanfiction Archives

**Best for**: Narrative comprehension, diverse register/style, character consistency
**Method**:
- Fanfic has explicit character names, settings, plot events
- Extract passages, generate questions about stated facts
- Rich source of non-encyclopedic prose

**Why fanfic is underrated**:
- Human-written (not synthetic)
- Narrative structure (beginning/middle/end)
- Enormous topical diversity (spans every fandom)
- Register diversity (literary to casual)
- Clear entity tracking (characters have names)

## Output Format

Each ICR instance should include:

```python
@dataclass
class ICRInstance:
    # Identifiers
    instance_id: str
    source_benchmark: str  # "mmlu", "triviaqa", etc.
    source_instance_id: str  # Original benchmark instance ID
    
    # The ICR instance itself
    context: str  # The generated/extracted context
    question: str  # May be modified from original
    options: list[str] | None  # For multiple choice
    correct_answer: str
    reasoning_trace: str  # Tutor's explanation
    
    # Training data
    context_token_ids: list[int]  # Preserved from tutor generation
    reasoning_token_ids: list[int]  # Preserved from tutor generation
    tutor_model: str  # Which model generated this
    
    # Metadata
    transformation_type: str  # "fictional_context", "explicit_setup", etc.
    source_corpus: str | None  # If extracted from a corpus
    difficulty_estimate: float  # 0-1, based on context length, complexity
    
    # Verification
    verified: bool  # Did a verifier confirm the answer is correct given context?
    verification_method: str  # "automatic", "judge_model", "human"
```

## Generation Pipeline

### Phase 1: Benchmark Ingestion

```python
def ingest_benchmark(benchmark_name: str) -> list[BenchmarkInstance]:
    """Load standard benchmark, return list of instances."""
    if benchmark_name == "mmlu":
        return load_mmlu()
    elif benchmark_name == "triviaqa":
        return load_triviaqa()
    # ... etc
```

### Phase 2: Context Generation

```python
def generate_icr_contexts(
    instance: BenchmarkInstance,
    tutor_model: TutorModel,
    transformation_type: str,
    source_corpus: Corpus | None = None,
) -> list[ICRInstance]:
    """Generate ICR versions of a benchmark instance."""
    
    if transformation_type == "fictional_context":
        # Generate synthetic contexts for each answer option
        contexts = []
        for option in instance.options:
            prompt = build_fictional_context_prompt(instance, option)
            context, tokens = tutor_model.generate_with_tokens(prompt)
            contexts.append((option, context, tokens))
        
        # Select one randomly for the ICR instance
        selected = random.choice(contexts)
        return [build_icr_instance(instance, selected, tutor_model)]
    
    elif transformation_type == "corpus_extraction":
        # Extract and modify passages from source corpus
        passage = find_relevant_passage(source_corpus, instance)
        modified = modify_for_icr(passage, instance)
        return [build_icr_instance(instance, modified, tutor_model)]
    
    # ... other transformation types
```

### Phase 3: Verification

```python
def verify_icr_instance(instance: ICRInstance, verifier: Verifier) -> ICRInstance:
    """Verify that the correct answer is actually correct given the context."""
    
    # For multiple choice: check that context supports the answer
    if instance.options:
        prompt = f"""
        Context: {instance.context}
        Question: {instance.question}
        Options: {instance.options}
        
        Based ONLY on the context provided, which option is correct?
        """
        verifier_answer = verifier.generate(prompt)
        instance.verified = (verifier_answer == instance.correct_answer)
    
    return instance
```

### Phase 4: Output

```python
def export_icr_dataset(instances: list[ICRInstance], output_dir: str):
    """Export ICR dataset for training and evaluation."""
    
    # Split into train (context generation) and eval (ICR instances)
    train_data = []
    eval_data = []
    
    for inst in instances:
        # Training data: the contexts and reasoning traces
        train_data.append({
            "text": inst.context + "\n" + inst.reasoning_trace,
            "token_ids": inst.context_token_ids + inst.reasoning_token_ids,
            "source": inst.tutor_model,
        })
        
        # Eval data: the ICR instances
        eval_data.append({
            "context": inst.context,
            "question": inst.question,
            "options": inst.options,
            "answer": inst.correct_answer,
        })
    
    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(eval_data, f"{output_dir}/eval.jsonl")
```

## CLI Interface

```bash
# Transform MMLU to MMLU-ICR
python -m icr_transform transform mmlu \
    --transformation fictional_context \
    --tutor-model kimi-k2 \
    --output-dir ./icr_benchmarks/mmlu_icr/

# Transform with corpus grounding
python -m icr_transform transform triviaqa \
    --transformation corpus_extraction \
    --source-corpus ./corpora/wikipedia/ \
    --tutor-model kimi-k2 \
    --output-dir ./icr_benchmarks/triviaqa_icr/

# Generate ICR versions of multiple benchmarks
python -m icr_transform transform-all \
    --benchmarks mmlu,triviaqa,arc,hellaswag \
    --tutor-model kimi-k2 \
    --output-dir ./icr_benchmarks/

# Verify existing ICR dataset
python -m icr_transform verify ./icr_benchmarks/mmlu_icr/ \
    --verifier-model gpt-4 \
    --fix-invalid
```

## Success Criteria

1. **Transformation coverage**: Successfully transform >90% of instances from each benchmark
2. **Verification rate**: >95% of generated ICR instances pass verification
3. **Discrimination**: Models score significantly different on canonical vs ICR versions (proving ICR tests different skills)
4. **Training utility**: Models trained on ICR contexts show improved in-context learning

## Evaluation Protocol

To demonstrate the value of ICR benchmarks:

```python
def evaluate_icr_discrimination(model, benchmark_name):
    """Show that ICR and canonical versions measure different things."""
    
    canonical = load_benchmark(benchmark_name)
    icr = load_icr_benchmark(benchmark_name)
    
    canonical_score = evaluate(model, canonical)
    icr_score = evaluate(model, icr)
    
    print(f"Canonical {benchmark_name}: {canonical_score:.1%}")
    print(f"ICR {benchmark_name}: {icr_score:.1%}")
    
    # The interesting models are those with:
    # - High ICR, low canonical: learned in-context retrieval, not facts
    # - High canonical, low ICR: memorized facts, poor in-context learning
    # - High both: genuinely capable
    # - Low both: not capable
```

## Dependencies

```
# Benchmark loading
datasets           # HuggingFace datasets
lm-eval            # EleutherAI eval harness (for benchmark access)

# Tutor inference
vllm               # Fast inference
transformers       # Fallback

# Corpus handling
wikipedia-api      # Wikipedia access
gutenberg          # Project Gutenberg access
```

## Non-Goals

- Perfect transformation of every benchmark (some are genuinely hard to ICR-ify)
- Replacing canonical benchmarks (ICR is complementary, not a replacement)
- Fully automatic pipeline with no human review (verification is important)

---

When done, this system should be able to take any major ML benchmark and produce:
1. An ICR version for evaluation
2. Training data from the generation process
3. Evidence that ICR measures in-context learning rather than memorization
