# Claudefile: Language Acquisition Eval Suite

## Context for Implementing Claude

You're building evaluation infrastructure for a language model training project. The central claim of this project is that a model trained with efficient curriculum + distillation can match the *language competence* of much deeper/larger models — even if it doesn't match them on knowledge retrieval benchmarks like MMLU.

To make this claim defensible, we need evals that measure **language competence** specifically, not factual knowledge. These evals should:

1. Run on model *rollouts* (generated text), not just next-token prediction
2. Be computable during training (on "hot" checkpoints) to show acquisition curves
3. Distinguish "real" language understanding from surface pattern matching
4. Place known models in the expected order (GPT-4 > Llama-70B > Llama-7B > random)

This is partially original eval work. Existing benchmarks don't measure what we care about.

## What You're Building

A Python module that takes model generations and returns scores on several language competence dimensions.

### Eval Dimensions

#### 1. Syntactic Validity

**What it measures:** Does the model produce grammatically valid English?

**Method:** 
- Run a constituency parser (e.g., `benepar`, `stanza`) on generated text
- Score = fraction of sentences that parse successfully without errors
- Bonus: parse tree depth distribution (deeper = more complex syntax)

**Baseline behavior:**
- Random token sequences: ~0% valid
- Char-level model at step 100: ~10% valid
- Char-level model at step 1000: ~90% valid
- Trained LM: ~99% valid

**Edge cases:**
- Creative/poetic syntax might fail parsing but be "valid" — accept this noise
- Very long sentences often fail parsers — truncate or handle gracefully

#### 2. Lexical Validity (for sub-BPE tokenizations only)

**What it measures:** Does the model produce real words vs. character salad?

**Method:**
- Tokenize generated text into words
- Check each word against a dictionary (e.g., `enchant`, `wordfreq`, or just a word list)
- Score = fraction of tokens that are valid words OR valid punctuation OR valid numbers

**Note:** This only matters for character-level or byte-level models. BPE models can't produce invalid words by construction. Skip this eval for BPE models.

#### 3. Reference Tracking / Entity Consistency

**What it measures:** Does the model maintain consistent references across a passage?

**Method:**
- Run coreference resolution (e.g., `coreferee`, `neuralcoref`, or a dedicated model)
- Identify entity chains
- Check for inconsistencies:
  - Gender switches: "Alice went to the store. He bought milk."
  - Number switches: "The dogs ran. It was fast."
  - Impossible references: pronouns with no antecedent

**Score:** Fraction of passages with zero reference errors (or: average errors per 1K tokens)

**This is important:** Reference tracking is a key marker of "understanding" vs. "pattern matching." Models that fail here are doing surface-level generation.

#### 4. Discourse Coherence

**What it measures:** Do paragraphs follow from each other? Is there a consistent topic?

**Method (embedding-based):**
- Split generation into chunks (sentences or paragraphs)
- Embed each chunk (sentence-transformers or similar)
- Compute cosine similarity between adjacent chunks
- Score = mean adjacent similarity (higher = more coherent)

**Method (LM-based):**
- Use a critic model to score "does sentence B follow from sentence A?"
- This is more expensive but more accurate

**Method (entropy-based):**
- Measure perplexity of each sentence conditioned on previous sentences (using a reference model)
- Coherent text should have lower perplexity than random sentence orderings

**Baseline behavior:**
- Shuffled sentences: low coherence score
- Real human text: high coherence score
- Trained LM should approach human text

#### 5. Narrative Structure (for story generation)

**What it measures:** Does the model produce text with beginning/middle/end structure?

**Method:**
- Prompt model with story beginnings
- Check for structural elements:
  - Does it introduce characters/setting early?
  - Does it have a conflict/event/change in the middle?
  - Does it conclude (vs. trailing off)?

**This is hard to automate.** Options:
- Keyword heuristics (names in first paragraph, "finally"/"the end" in last paragraph)
- Critic model trained on story structure
- Human eval (expensive, not for training-time use)

For now: implement a simple heuristic version, flag it as approximate.

#### 6. Repetition / Degeneration

**What it measures:** Does the model fall into repetitive loops?

**Method:**
- Compute n-gram repetition rates in generated text
- Flag if any n-gram (n >= 4) repeats more than k times
- Score = 1 - (repetition rate)

**This is a sanity check.** Degenerate models fail here catastrophically. It's not a measure of "understanding" but it's necessary to detect failure modes.

### Aggregate Scoring

Combine the above into a single "Language Competence Score" (LCS):

```
LCS = w1*syntactic + w2*lexical + w3*reference + w4*discourse + w5*narrative + w6*repetition
```

Default weights TBD, but reference tracking and discourse coherence should be weighted heavily — those are the dimensions that distinguish "real" understanding.

## Interface

### Python API

```python
from language_evals import evaluate_generations, LanguageScores

# Single generation
scores: LanguageScores = evaluate_generations(
    texts=["Once upon a time, there was a ..."],
    eval_dimensions=["syntactic", "reference", "discourse"],
)
print(scores.syntactic)  # 0.95
print(scores.reference)  # 0.88
print(scores.aggregate)  # 0.91

# Batch evaluation
batch_scores = evaluate_generations(
    texts=[gen1, gen2, gen3, ...],
    eval_dimensions="all",
    return_per_text=True,  # vs. aggregate only
)
```

### CLI

```bash
# Evaluate a file of generations (one per line)
python -m language_evals eval generations.txt --output scores.json

# Evaluate with a specific model (for LM-based metrics)
python -m language_evals eval generations.txt --critic-model gpt2-large

# Run on model outputs during training
python -m language_evals eval-live --model-path /path/to/checkpoint --prompts prompts.txt
```

### Integration with Training

The key use case is evaluating "hot" models during training:

```python
# In training loop
if step % eval_interval == 0:
    prompts = load_eval_prompts()
    generations = model.generate(prompts, max_length=512)
    scores = evaluate_generations(generations)
    log_metrics({
        "language/syntactic": scores.syntactic,
        "language/reference": scores.reference,
        "language/discourse": scores.discourse,
        "language/aggregate": scores.aggregate,
    })
```

This produces acquisition curves showing when each capability emerges.

## Dependencies

```
# Parsing
benepar            # constituency parsing
spacy              # tokenization, base NLP
stanza             # alternative parser

# Coreference
coreferee          # lightweight coref
# or: neuralcoref (deprecated but works)
# or: a dedicated coref model via transformers

# Embeddings
sentence-transformers  # for discourse coherence

# Utilities
numpy
torch              # for model-based metrics
```

Optional:
```
# For critic-model-based evals
transformers
accelerate
```

## Implementation Priority

1. **Syntactic validity** — easiest, most important baseline
2. **Repetition/degeneration** — easy, catches failure modes
3. **Reference tracking** — medium difficulty, high signal
4. **Discourse coherence (embedding-based)** — medium difficulty, high signal
5. **Lexical validity** — easy but only relevant for char/byte models
6. **Narrative structure** — hard to automate, lower priority

## Test Cases

Generate test inputs that should produce known scores:

```python
# Should score high on everything
good_text = """
Alice walked into the coffee shop and ordered a latte. She sat by the window, 
watching the rain. The barista called her name, and she picked up her drink. 
It was too hot, so she waited. After a few minutes, she took a sip and smiled.
"""

# Should fail reference tracking
bad_reference = """
Alice walked into the coffee shop. He ordered a latte. The barista called 
their name. She picked up his drink.
"""

# Should fail discourse coherence
incoherent = """
The mitochondria is the powerhouse of the cell. Purple elephants dance on 
Tuesdays. Financial derivatives require careful hedging. She never liked 
the taste of cilantro.
"""

# Should fail repetition
degenerate = """
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. 
The cat sat on the mat. The cat sat on the mat.
"""
```

## Calibration

Before using these evals to make claims, calibrate against known models:

1. Run on generations from GPT-4, Claude, Llama-70B, Llama-7B, GPT-2, random
2. Verify the ranking matches expectations on each dimension
3. Adjust weights/thresholds if needed

If the evals don't rank known models correctly, they're not measuring what we think they're measuring.

## Non-Goals

- Factual accuracy (that's knowledge, not language)
- Task performance (MMLU, HumanEval, etc.)
- Style/tone evaluation
- Safety/toxicity (separate concern)

## Open Questions

1. **How to handle different generation lengths?** Longer generations have more opportunity for errors. Normalize somehow?

2. **What prompts to use for eval generations?** Need a standard prompt set that elicits extended generation. Story prompts? Essay prompts? Open-ended questions?

3. **How expensive is this to run?** Parsing and coref are not free. Need to profile and possibly subsample for training-time eval.

---

When done, this module should let us make claims like: "Our model achieves 0.92 LCS after 10K training steps, matching Llama-70B's score of 0.91 on these language competence measures."
