# Text Templating as ICR: Breaking Free from Format Wars

## The Problem We Discovered

Cross-tutor GKD testing revealed that models from different families disagree violently on text formatting at sequence start:
- Gemma-3 predicts `import`, `package`, `<h2>` (code/HTML prior)
- Qwen3 predicts `Instructions`, `Answer` (instruction prior)

This isn't a bug - it's a discovered property. Models trained on different corpora have different priors about what text "should look like". The 2022-2024 era "solved" this with rigid chat templates (`<|im_start|>user`, `<start_of_turn>`, etc.) that became API provider lock-in artifacts.

## The Insight

Text templating is a **policy optimization problem**, not an SFT problem. You can't SFT your way out of format disagreement - you have to train the model to *anticipate* and *follow* formatting schemes explained in context.

This is exactly what ICR already does for domain knowledge. We extend it to formatting.

## Text Templating as In-Context Retrieval

Just as we prepend method libraries for math problems:
```
# Method Library: Arithmetic
To add numbers, combine their values...

Q: What is 5 + 3?
```

We can prepend **format specifications** for text structure:
```
# Format Specification
In this document:
- Queries are prefixed with "Q:" on their own line
- Responses are prefixed with "A:" on their own line
- Calculations are shown inline with <<expr=result>> notation
- Final answers are marked with #### followed by the number

Q: Janet has 16 eggs...
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs...
#### 18
```

The model learns to anticipate that format specs exist and to follow them.

## Fictional but Verifiable Formats

The power of this approach: we can define *arbitrary* formats that are:
1. Described in natural language
2. Demonstrated with examples
3. Mechanically verifiable

### Example: Markerdown

Markerdown is like Markdown but every division (heading, horizontal rule) includes a chromatic accent marker:

```
# Format: Markerdown
Markerdown extends Markdown with chromatic division markers.
After each heading or horizontal rule, add a marker: 🖍{color}
Colors indicate emphasis: 🔴=critical, 🟠=warning, 🟡=note, 🟢=success, 🔵=info

---

# Problem Statement 🖍🔵

Janet's ducks lay 16 eggs per day...

# Solution 🖍🟢

Step 1: Calculate eggs remaining after breakfast...

# Final Answer 🖍🔴

#### 18
```

This format:
- Is completely fictional (not a real standard)
- Is trivially parseable (regex: `🖍[🔴🟠🟡🟢🔵🟣]`)
- Can generate true positives (correctly formatted) and false positives (violations)
- Breaks any model's formatting prior (no pretraining on this)

### Example: Bracket-Speaker Attribution

```
# Format: Bracket-Speaker
Speaker attribution uses [SPEAKER: content] blocks.
Speakers in this document: STUDENT (asking), TUTOR (explaining)

[STUDENT: I need help with this problem. Janet has 16 eggs...]

[TUTOR: Let's break this down step by step.
First, calculate how many eggs remain after Janet eats breakfast:
16 - 3 = 13 eggs remaining
...]
```

### Example: Indentation-Scope

```
# Format: Indentation-Scope
Logical scope is indicated by indentation (2 spaces per level).
Scope 0: Problem statement
Scope 1: Solution steps
Scope 2: Sub-calculations
Scope 3: Verification

Janet has 16 eggs...
  We need to find her daily earnings.
    Eggs remaining: 16 - 3 - 4 = 9
      Verify: 3 + 4 + 9 = 16 ✓
    Earnings: 9 × $2 = $18
  Therefore, Janet earns $18 daily.
```

## Training Implications

### What This Means for GKD

When doing cross-tokenizer distillation:
1. **Prefix tokens (format spec) should be loss-masked** - we're not training the model to predict the format spec, we're training it to follow it
2. **Response tokens get full GKD loss** - this is where the model learns to generate content that follows the specified format
3. **Format disagreement becomes a feature** - models that "fight" on format in their priors will learn to defer to in-context format specs

### Generating Training Data

For each benchmark problem, we can generate multiple format variations:
```python
FORMATS = [
    "qa_simple",      # Q: ... A: ...
    "bracket_speaker", # [STUDENT: ...] [TUTOR: ...]
    "markerdown",     # Headings with 🖍{color}
    "indentation",    # Scope via indentation
    "xml_tags",       # <question>...</question><answer>...</answer>
    "numbered_steps", # 1. Problem 2. Given 3. Solution 4. Answer
]

for problem in benchmark:
    for fmt in FORMATS:
        yield format_problem(problem, fmt, include_spec=True)
```

### Verification Pipeline

Each format has a verifier:
```python
def verify_markerdown(text: str) -> bool:
    """Check that all headings/rules have valid color markers."""
    headings = re.findall(r'^#+\s+.+$', text, re.MULTILINE)
    for h in headings:
        if not re.search(r'🖍[🔴🟠🟡🟢🔵🟣]', h):
            return False
    return True

def verify_bracket_speaker(text: str, speakers: list[str]) -> bool:
    """Check that all content is in valid speaker blocks."""
    blocks = re.findall(r'\[([A-Z]+):', text)
    return all(s in speakers for s in blocks)
```

## Why This Works

### Natural Language is Homoiconic

Just as Lisp code is Lisp data, natural language descriptions of formats are themselves formatted text. We can:
- Describe a format in English
- Demonstrate the format with examples
- Use the format in the same document
- All without escaping or meta-levels

### Models Already Do This

Instruction-tuned models already follow format specs when explicitly stated:
> "Respond in JSON format with keys 'answer' and 'reasoning'"

We're just making this systematic and extending it to arbitrary formats that:
- Break pretraining priors (no model has seen markerdown)
- Are verifiable (we can check compliance)
- Are diverse (many formats per problem)

### ICR Already Solves the Hard Part

The ICR transformation already prepends method libraries. Adding format specs is the same operation:
```
[FORMAT_SPEC]
[METHOD_LIBRARY]
[PROBLEM]
[RESPONSE]
```

The model learns:
1. Read the format spec
2. Read the method library
3. Read the problem
4. Generate a response that follows both

## Integration with Existing Infrastructure

### ICRTransformer Extension

```python
class ICRTransformer:
    def transform(
        self,
        problem_id: str,
        benchmark: str,
        question_text: str,
        answer: str,
        options: Optional[list[str]] = None,
        format_spec: Optional[str] = None,  # NEW
        transformation_type: Optional[TransformationType] = None,
    ) -> ICRInstance:
        ...
```

### Format Registry

```python
FORMAT_SPECS = {
    "qa_simple": """
Queries are prefixed with "Q:" and responses with "A:".
Calculations use <<expr=result>> notation.
Final answers use #### followed by the number.
""",
    "markerdown": """
This document uses Markerdown format.
Headings and horizontal rules include chromatic markers: 🖍{color}
Colors: 🔴=critical, 🟠=warning, 🟡=note, 🟢=success, 🔵=info
""",
    # ... more formats
}
```

### Contamination Checking

Format specs don't affect contamination - they're metadata, not problem content. The firewall checks the underlying problem text, not the formatting wrapper.

## Summary

1. **Text templating disagreement is a training signal**, not a bug
2. **Format specs are just another form of ICR** - prepend, demonstrate, follow
3. **Fictional formats break priors** - markerdown, bracket-speaker, etc.
4. **Verification is mechanical** - regex, parsing, structural checks
5. **This sidesteps format wars** - we don't pick a winner, we train format-following

The goal: models that *anticipate* format specs and *defer* to in-context formatting rules, rather than fighting with their pretraining priors.
