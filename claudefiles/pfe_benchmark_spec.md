# Claudefile: PFE (Posthumanity's First Exam) — Benchmark Specification

## What This Is

PFE is a benchmark suite designed to measure **in-context learning** rather than **memorization** or **retrieval from weights**. It transforms existing benchmarks into ICR (In-Context Retrieval) format and enforces strict separation between training and evaluation data.

The core scientific claim PFE enables:

> "A model trained on synthetic problems (verified non-overlapping with canonical benchmarks) achieves X% on canonical eval and Y% on ICR-augmented eval. The gap (Y - X) represents pure in-context learning capability, not contamination."

## The Three-Split Architecture

### Split 1: Canonical Eval (UNTOUCHED)

These are the original benchmark test sets. We never train on them. We never generate answer keys for them. We only evaluate on them.

| Benchmark | Size | What It Tests |
|-----------|------|---------------|
| GSM1K | 1,250 | Arithmetic reasoning |
| ARC-Easy | 2,376 | Grade school science |
| ARC-Challenge | 1,172 | Harder science reasoning |
| RACE (test) | 4,934 | Reading comprehension |
| BoolQ (test) | 3,270 | Boolean QA from passages |
| HellaSwag (val) | 10,042 | Commonsense completion |
| WinoGrande (test) | 1,267 | Coreference resolution |
| MBPP (test) | 500 | Simple Python programming |
| TriviaQA (test) | 11,313 | Factual QA |

**Total canonical eval: ~36,000 problems**

These remain sealed. The model sees them only at evaluation time.

### Split 2: Synthetic Training Corpus (OUR CREATION)

Problems we generate that are **similar in structure and difficulty** to canonical benchmarks, but **verified non-overlapping**.

For each canonical benchmark, we generate synthetic problems that:
- Test the same skills
- Use the same format
- Have similar difficulty distribution
- Are **provably distinct** from canonical problems

| Synthetic Set | Target Size | Source of Generation |
|---------------|-------------|---------------------|
| GSM1K-Synth | 10,000 | Tutor-generated word problems |
| ARC-Synth | 5,000 | Tutor-generated science QA |
| RACE-Synth | 5,000 | Tutor-generated reading comp |
| BoolQ-Synth | 5,000 | Tutor-generated boolean QA |
| HellaSwag-Synth | 5,000 | Tutor-generated completions |
| WinoGrande-Synth | 3,000 | Tutor-generated coref problems |
| MBPP-Synth | 2,000 | Tutor-generated coding tasks |
| FnCall-Synth | 2,000 | Tutor-generated function calling |
| Format-Synth | 2,000 | Tutor-generated format tasks |

**Total synthetic training: ~39,000 problems**

Each synthetic problem comes with:
- The problem itself
- Answer keys from multiple tutors (for GKD)
- ICR context (method library, worked examples)
- Verification that it passes the contamination firewall

### Split 3: ICR-Augmented Eval (FOR DIAGNOSIS)

The canonical eval problems, augmented with method libraries / worked examples prepended as context.

This is NOT for training. This is for measuring: "Does providing context help the model solve problems it couldn't solve from weights alone?"

| ICR Eval Set | Base | ICR Augmentation |
|--------------|------|------------------|
| GSM1K-ICR | GSM1K canonical | 5-method arithmetic library + worked examples |
| ARC-ICR | ARC canonical | Science fact library + reasoning examples |
| RACE-ICR | RACE canonical | (already has passage — add strategy hints) |
| MBPP-ICR | MBPP canonical | Pattern examples + similar solved problems |
| FnCall-ICR | FnCall canonical | API docs + example calls |

**Same size as canonical: ~36,000 problems**

No training on these. Evaluation only.

---

## The Contamination Firewall

Every synthetic training problem must pass three checks before inclusion:

### Check 1: Token-Level Overlap

```python
def check_token_overlap(synthetic: str, canonical_set: list[str], n: int = 5, threshold: float = 0.3) -> bool:
    """
    Reject if synthetic shares too many n-grams with any canonical problem.
    
    Returns True if SAFE (low overlap), False if CONTAMINATED.
    """
    synthetic_ngrams = set(ngrams(tokenize(synthetic), n))
    
    for canonical in canonical_set:
        canonical_ngrams = set(ngrams(tokenize(canonical), n))
        
        if len(synthetic_ngrams) == 0:
            continue
            
        overlap = len(synthetic_ngrams & canonical_ngrams) / len(synthetic_ngrams)
        
        if overlap > threshold:
            return False  # CONTAMINATED
    
    return True  # SAFE
```

**Parameters:**
- n = 5 (5-grams)
- threshold = 0.3 (reject if >30% of synthetic n-grams appear in any canonical)

### Check 2: Semantic Similarity

```python
def check_semantic_similarity(synthetic: str, canonical_set: list[str], threshold: float = 0.85) -> bool:
    """
    Reject if synthetic is too semantically similar to any canonical problem.
    
    Returns True if SAFE, False if CONTAMINATED.
    """
    synthetic_emb = embed(synthetic)  # sentence-transformers or similar
    
    for canonical in canonical_set:
        canonical_emb = embed(canonical)
        similarity = cosine_similarity(synthetic_emb, canonical_emb)
        
        if similarity > threshold:
            return False  # CONTAMINATED
    
    return True  # SAFE
```

**Parameters:**
- threshold = 0.85 (reject if cosine similarity > 0.85)
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 or similar

### Check 3: Structural Similarity (Domain-Specific)

For math problems:
```python
def check_math_structure(synthetic: MathProblem, canonical_set: list[MathProblem]) -> bool:
    """
    Reject if synthetic has same numbers AND same operations as any canonical.
    """
    synthetic_sig = (
        frozenset(extract_numbers(synthetic)),
        frozenset(extract_operations(synthetic)),
        synthetic.answer
    )
    
    for canonical in canonical_set:
        canonical_sig = (
            frozenset(extract_numbers(canonical)),
            frozenset(extract_operations(canonical)),
            canonical.answer
        )
        
        if synthetic_sig == canonical_sig:
            return False  # CONTAMINATED
    
    return True  # SAFE
```

For code problems:
```python
def check_code_structure(synthetic: CodeProblem, canonical_set: list[CodeProblem]) -> bool:
    """
    Reject if synthetic has same function signature AND same test cases.
    """
    synthetic_sig = (synthetic.function_name, frozenset(synthetic.test_inputs))
    
    for canonical in canonical_set:
        canonical_sig = (canonical.function_name, frozenset(canonical.test_inputs))
        
        if synthetic_sig == canonical_sig:
            return False  # CONTAMINATED
    
    return True  # SAFE
```

### Combined Firewall

```python
def contamination_firewall(synthetic: Problem, canonical_set: list[Problem], domain: str) -> tuple[bool, str]:
    """
    Run all contamination checks. Return (is_safe, rejection_reason).
    """
    if not check_token_overlap(synthetic.text, [c.text for c in canonical_set]):
        return False, "token_overlap"
    
    if not check_semantic_similarity(synthetic.text, [c.text for c in canonical_set]):
        return False, "semantic_similarity"
    
    if domain == "math":
        if not check_math_structure(synthetic, canonical_set):
            return False, "math_structure"
    
    elif domain == "code":
        if not check_code_structure(synthetic, canonical_set):
            return False, "code_structure"
    
    return True, "passed"
```

---

## Benchmark Specifications

### GSM1K / GSM1K-ICR

**Canonical (GSM1K):**
- 1,250 grade school math problems
- 2-8 steps each
- Basic arithmetic only

**ICR Context (prepended to each problem):**
```
[METHOD LIBRARY]

METHOD 1: COMBINING QUANTITIES (Addition)
When: Someone gets more, or finding a total
Pattern: "altogether", "in total", "combined", "plus", "more"
Example: Sam has 12 apples. He buys 8 more. How many now?
→ Combining: 12 + 8 = 20 apples.

METHOD 2: FINDING WHAT'S LEFT (Subtraction)
When: Something removed, used, given away, or finding difference
Pattern: "left", "remaining", "gave away", "fewer", "less"
Example: 45 cookies, kids eat 17. How many left?
→ What's left: 45 - 17 = 28 cookies.

METHOD 3: REPEATED GROUPS (Multiplication)
When: Equal groups, "each", rates
Pattern: "each", "per", "every", "times"
Example: 6 boxes × 8 toys each. How many toys?
→ Repeated groups: 6 × 8 = 48 toys.

METHOD 4: SPLITTING EVENLY (Division)
When: Sharing equally, finding group size
Pattern: "split", "share", "each gets", "divided"
Example: 36 stickers ÷ 4 kids. How many each?
→ Splitting: 36 ÷ 4 = 9 stickers each.

METHOD 5: MULTI-STEP
When: Problem has multiple parts
Strategy: Solve steps in order, use earlier results.
Example: 5 bags × 6 apples, give away 10. How many left?
→ Step 1: 5 × 6 = 30. Step 2: 30 - 10 = 20 apples.

[PROBLEM]
{canonical problem here}
```

**Synthetic Training (GSM1K-Synth):**
- 10,000 generated problems
- Same format and difficulty as GSM1K
- All pass contamination firewall
- Answer keys from 5+ tutors

**Evaluation Metrics:**
- GSM1K canonical accuracy (no context)
- GSM1K-ICR accuracy (with method library)
- Gap: ICR - canonical (measures in-context learning benefit)

### ARC / ARC-ICR

**Canonical:**
- ARC-Easy: 2,376 problems
- ARC-Challenge: 1,172 problems
- Multiple choice science questions

**ICR Context:**
```
[SCIENCE FACT LIBRARY]

CATEGORY: States of Matter
- Solids have fixed shape and volume
- Liquids have fixed volume but take container's shape
- Gases expand to fill any container
- Heating generally causes expansion
- Cooling generally causes contraction

CATEGORY: Forces and Motion
- Objects at rest stay at rest unless acted on by force
- Friction opposes motion
- Gravity pulls objects toward Earth
- Heavier objects don't fall faster (ignoring air resistance)

CATEGORY: Energy
- Energy cannot be created or destroyed, only transformed
- Heat flows from hot to cold
- Light travels in straight lines
- Sound needs a medium to travel

[WORKED EXAMPLES]
Q: What happens to water when heated to 100°C at sea level?
A: It boils and becomes steam (liquid → gas transition).

Q: Why does a ball roll farther on smooth floor than rough carpet?
A: Less friction on smooth surface allows more motion.

[PROBLEM]
{canonical problem here}
```

### MBPP / MBPP-ICR (Code)

**Canonical:**
- 500 Python programming problems
- Function implementation from docstring

**ICR Context:**
```
[PATTERN LIBRARY]

PATTERN 1: List Iteration
When: Process each element, transform, filter, or accumulate
Template:
    result = []
    for item in items:
        if condition(item):  # optional filter
            result.append(transform(item))
    return result

PATTERN 2: Dictionary Building
When: Group items, count occurrences, create mapping
Template:
    result = {}
    for item in items:
        key = get_key(item)
        result[key] = result.get(key, default) + value
    return result

PATTERN 3: Two-Pointer / Sliding Window
When: Find pairs, subarrays, or compare positions
Template:
    left, right = 0, len(arr) - 1
    while left < right:
        # process arr[left], arr[right]
        # move pointers based on condition

PATTERN 4: Recursion with Base Case
When: Problem has self-similar subproblems
Template:
    def solve(problem):
        if is_base_case(problem):
            return base_solution
        return combine(solve(smaller_problem))

[WORKED EXAMPLE]
Task: Write a function to find the sum of squares of a list.
Solution:
    def sum_of_squares(lst):
        result = 0
        for x in lst:
            result += x * x
        return result

[PROBLEM]
{canonical problem here}
```

### Function Calling / FnCall-ICR

**Canonical (custom):**
- 500 function calling problems
- Given API docs, generate correct call

**ICR Context:**
```
[AVAILABLE FUNCTIONS]

get_weather(location: str, units: str) -> dict
    Returns current weather for location.
    units: "celsius" or "fahrenheit"
    Example: get_weather("Tokyo", "celsius") → {"temp": 22, "conditions": "cloudy"}

search_web(query: str, num_results: int) -> list[dict]
    Searches web, returns top results.
    Example: search_web("python tutorials", 3) → [{"title": ..., "url": ..., "snippet": ...}, ...]

send_email(to: str, subject: str, body: str) -> dict
    Sends email, returns status.
    Example: send_email("bob@example.com", "Hello", "Hi Bob!") → {"success": true, "id": "msg_123"}

calculate(expression: str) -> float
    Evaluates math expression.
    Example: calculate("(15 + 27) * 3") → 126.0

get_stock_price(symbol: str) -> dict
    Returns current stock price.
    Example: get_stock_price("AAPL") → {"price": 182.52, "currency": "USD"}

[REQUEST]
{user request here}

[TASK]
Generate the correct function call(s) to fulfill this request.
```

---

## Evaluation Protocol

### Primary Metrics

For each benchmark B:

1. **Canonical Accuracy**: % correct on B with no context
2. **ICR Accuracy**: % correct on B-ICR with method library
3. **ICR Lift**: (ICR Accuracy - Canonical Accuracy)
4. **Relative ICR Lift**: ICR Lift / (100 - Canonical Accuracy)

The **Relative ICR Lift** measures: "Of the problems the model couldn't solve without context, what fraction could it solve with context?"

### Comparison Points

| Model | Description |
|-------|-------------|
| **Ours (trained)** | Student model trained on synthetic corpus via GKD |
| **Ours (base)** | Same architecture, no training (random init baseline) |
| **Tutors** | Each tutor model in the ensemble |
| **Existing models** | Llama-3-8B, Mistral-7B, Phi-3, etc. |

### Reporting Format

```
Benchmark: GSM1K
──────────────────────────────────────────────────────
Model                 Canonical    ICR      Lift   Rel.Lift
──────────────────────────────────────────────────────
Ours (trained)           72.3%    86.1%   +13.8%     49.8%
Ours (base)               8.2%    31.4%   +23.2%     25.3%
GPT-4o (tutor)           95.2%    96.8%    +1.6%     33.3%
Llama-3-8B               58.1%    71.3%   +13.2%     31.5%
──────────────────────────────────────────────────────
```

### Aggregate Scores

**PFE-Canonical**: Average canonical accuracy across all benchmarks
**PFE-ICR**: Average ICR accuracy across all benchmarks
**PFE-Lift**: Average ICR lift across all benchmarks

---

## What Success Looks Like

### Strong Success

- Trained model shows **large ICR lift** (>10% average)
- Trained model approaches **tutor performance** on ICR benchmarks
- Trained model **outperforms same-size baselines** on both canonical and ICR
- **Contamination firewall held**: model didn't see canonical problems

Interpretation: Model learned generalizable in-context learning from synthetic training.

### Moderate Success

- Trained model shows **moderate ICR lift** (5-10%)
- Trained model **matches baselines** on canonical, **exceeds on ICR**
- Some benchmarks show lift, others don't

Interpretation: ICR helps for some task types. Worth investigating which.

### Weak/Negative Results

- Trained model shows **minimal ICR lift** (<5%)
- Or: **ICR hurts** (negative lift)
- Or: Model performs **worse than baselines**

Interpretation: Either our synthetic training failed, or ICR doesn't help for these tasks, or our method libraries are bad. Each is interesting to diagnose.

---

## Implementation Checklist

- [ ] Obtain canonical benchmark test sets (do not modify)
- [ ] Build contamination firewall infrastructure
- [ ] Generate synthetic training problems (with rejection logging)
- [ ] Collect answer keys from tutor ensemble
- [ ] Build ICR context templates for each benchmark
- [ ] Implement evaluation harness (canonical + ICR modes)
- [ ] Run baseline evaluations on existing models
- [ ] Train student model on synthetic corpus
- [ ] Evaluate trained model on canonical + ICR
- [ ] Compute all metrics, generate report

---

## Non-Goals

- Achieving SOTA on canonical benchmarks (that's not the point)
- Replacing existing benchmarks (PFE is complementary)
- Testing every possible benchmark (we chose tractable ones)
- Perfect synthetic data (some noise is fine if firewall holds)

The goal is a **clean scientific measurement** of in-context learning capability, with **provable separation** between training and evaluation data.
