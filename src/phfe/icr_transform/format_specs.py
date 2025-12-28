"""
Natural Language Format Specifications for ICR Text Templating

These are human-readable explanations of text formats, written from the perspective
of "how to write data so it parses correctly". Used as ICR context to teach models
format-following behavior.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import re


@dataclass
class FormatSpec:
    """A text format specification with explanation, examples, and verifier."""

    name: str
    explanation: str  # Natural language description
    example: str  # Demonstration of the format
    verifier: Optional[Callable[[str], bool]] = None  # Mechanical verification

    def full_spec(self) -> str:
        """Return the complete format specification for ICR prepending."""
        return f"{self.explanation}\n\n{self.example}"


# =============================================================================
# Core Format Specifications
# =============================================================================

MARKDOWN_SPEC = FormatSpec(
    name="markdown",
    explanation="""## How to Write Markdown

Markdown is a lightweight way to format text. Here's what you need to know:

**Headings**: Start a line with # symbols. More #'s = smaller heading.
- # Big heading
- ## Medium heading
- ### Small heading

**Emphasis**: Wrap text in asterisks or underscores.
- *italic* or _italic_
- **bold** or __bold__
- ***bold italic***

**Lists**: Start lines with - or * for bullets, or 1. 2. 3. for numbered lists.
Indent with 2-4 spaces for nested items.

**Code**: Use `backticks` for inline code. Use triple backticks for code blocks:
```python
def example():
    return "code here"
```

**Links and Images**:
- [link text](url)
- ![alt text](image-url)

**Blockquotes**: Start lines with >
> This is a quote

**Horizontal rules**: Three or more dashes, asterisks, or underscores on a line.
---

**Paragraphs**: Separate with blank lines. Single newlines within a paragraph
are treated as spaces.""",

    example="""### Example Document

Here's a **bold** statement with some *emphasis*.

A list of items:
- First item
- Second item
  - Nested item
- Third item

Some `inline code` and a block:
```
code block here
```

> A wise quote

---

That's the basics of Markdown.""",

    verifier=lambda text: (
        # Basic check: has some markdown-like structures
        bool(re.search(r'^#+\s', text, re.MULTILINE)) or  # headings
        bool(re.search(r'\*\*.+\*\*', text)) or  # bold
        bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE))  # lists
    )
)


JSON_SPEC = FormatSpec(
    name="json",
    explanation="""## How to Write JSON

JSON (JavaScript Object Notation) stores structured data as text. The rules are strict:

**Objects**: Curly braces containing key-value pairs.
- Keys MUST be strings in double quotes
- Separate key from value with colon
- Separate pairs with commas
- No trailing commas allowed

**Arrays**: Square brackets containing values separated by commas.

**Values can be**:
- Strings: "text in double quotes" (escape " as \\", newlines as \\n)
- Numbers: 42, 3.14, -17, 1.5e10 (no quotes, no leading zeros like 007)
- Booleans: true or false (lowercase, no quotes)
- Null: null (lowercase, no quotes)
- Objects: {...}
- Arrays: [...]

**Whitespace**: Spaces, tabs, newlines between elements are ignored (for readability).

**Common mistakes to avoid**:
- Single quotes (wrong: 'text', right: "text")
- Trailing commas (wrong: [1, 2, 3,], right: [1, 2, 3])
- Unquoted keys (wrong: {name: "x"}, right: {"name": "x"})
- Comments (JSON has no comments - remove them)""",

    example="""{
  "name": "Example",
  "count": 42,
  "active": true,
  "tags": ["important", "example"],
  "metadata": {
    "created": "2024-01-15",
    "version": 1.0
  },
  "notes": null
}""",

    verifier=lambda text: _verify_json(text)
)


def _verify_json(text: str) -> bool:
    """Check if text is valid JSON."""
    import json
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


YAML_SPEC = FormatSpec(
    name="yaml",
    explanation="""## How to Write YAML

YAML is a human-friendly data format. It's like JSON but uses indentation instead of braces.

**Key-value pairs**: Key, colon, space, value. No quotes needed for simple strings.
```
name: Example
count: 42
```

**Indentation**: Use spaces (not tabs!) to show nesting. Be consistent (usually 2 spaces).
```
person:
  name: Alice
  age: 30
```

**Lists**: Dash, space, then the item. Each item on its own line.
```
colors:
  - red
  - green
  - blue
```

**Strings**: Usually no quotes needed. Use quotes for:
- Strings starting with special chars: "- not a list"
- Strings with colons: "key: looks like yaml"
- Multi-line strings use | (literal) or > (folded)

**Numbers and booleans**: Just write them plain.
- Numbers: 42, 3.14, 1e10
- Booleans: true, false, yes, no (all work)
- Null: null or ~

**Comments**: # starts a comment (unlike JSON, YAML has comments!)

**Common mistakes**:
- Mixing tabs and spaces (use only spaces)
- Inconsistent indentation
- Forgetting the space after colons
- Forgetting the space after list dashes""",

    example="""# Example YAML document
name: Example Configuration
version: 1.0
enabled: true

settings:
  timeout: 30
  retries: 3

tags:
  - important
  - example
  - demo

description: |
  This is a multi-line
  string that preserves
  line breaks.""",

    verifier=lambda text: _verify_yaml(text)
)


def _verify_yaml(text: str) -> bool:
    """Check if text is valid YAML."""
    try:
        import yaml
        yaml.safe_load(text)
        return True
    except Exception:
        return False


# =============================================================================
# Experimental/Fictional Format Specifications
# =============================================================================

MARKERDOWN_SPEC = FormatSpec(
    name="markerdown",
    explanation="""## How to Write Markerdown

Markerdown extends Markdown with chromatic division markers. Every heading or
horizontal rule gets a color marker to indicate its semantic emphasis.

**Markers**: After headings or rules, add a crayon emoji followed by a color:
- 🖍🔴 = Critical/Error - for warnings, breaking changes, must-read sections
- 🖍🟠 = Warning/Caution - for deprecations, gotchas, edge cases
- 🖍🟡 = Note/Attention - for tips, clarifications, asides
- 🖍🟢 = Success/Good - for solutions, correct examples, achievements
- 🖍🔵 = Info/Neutral - for general information, context, background

**Usage**: Place the marker at the end of the heading line or after a rule.

```
# Important Warning 🖍🔴

This section contains critical information.

## Solution 🖍🟢

Here's how to fix it.

---
🖍🟡

A brief note between sections.
```

**Rules**:
- Every # heading MUST have a marker
- Horizontal rules (---) SHOULD have a marker on the next line
- Choose colors based on the semantic meaning, not aesthetics
- When in doubt, use 🖍🔵 (info/neutral)""",

    example="""# Document Title 🖍🔵

This document demonstrates Markerdown formatting.

## Problem Statement 🖍🟠

We need to calculate Janet's earnings from selling eggs.

## Given Information 🖍🔵

- Janet's ducks lay 16 eggs per day
- She eats 3 for breakfast
- She bakes with 4 for muffins
- She sells the rest at $2 each

## Solution 🖍🟢

Eggs remaining: 16 - 3 - 4 = 9
Daily earnings: 9 × $2 = $18

---
🖍🟡

Note: This assumes all remaining eggs are sold.

## Final Answer 🖍🔴

Janet earns **$18** per day at the farmers' market.""",

    verifier=lambda text: _verify_markerdown(text)
)


def _verify_markerdown(text: str) -> bool:
    """Verify markerdown format - all headings must have color markers."""
    lines = text.split('\n')
    for line in lines:
        # Check if line is a heading
        if re.match(r'^#+\s+', line):
            # Must end with a marker
            if not re.search(r'🖍[🔴🟠🟡🟢🔵🟣]$', line.strip()):
                return False
    return True


BRACKET_SPEAKER_SPEC = FormatSpec(
    name="bracket_speaker",
    explanation="""## How to Write Bracket-Speaker Format

Bracket-Speaker format explicitly attributes all content to named speakers using
square bracket tags. This eliminates ambiguity about who is saying what.

**Basic syntax**: [SPEAKER: content]

**Rules**:
- Speaker names are UPPERCASE
- Colon and space after the speaker name
- Content can span multiple lines within the brackets
- Close bracket ] ends that speaker's turn

**Common speakers**:
- [SYSTEM: ...] - Instructions, context, metadata
- [USER: ...] - Questions, requests, input
- [ASSISTANT: ...] - Responses, answers, output
- [STUDENT: ...] - Questions from a learner
- [TUTOR: ...] - Explanations from a teacher

**Multi-line content**:
```
[TUTOR: Let me explain this step by step.

First, we identify what we know.
Then, we apply the relevant formula.
Finally, we calculate the answer.]
```

**Nesting is NOT allowed** - each speaker block is independent.

**Transitioning**: Just close one bracket and open the next.
```
[USER: What is 2+2?]

[ASSISTANT: 2+2 equals 4.]
```""",

    example="""[SYSTEM: This is a math tutoring session. The tutor will help the student
solve word problems step by step.]

[STUDENT: I need help with this problem: Janet's ducks lay 16 eggs per day.
She eats 3 for breakfast and bakes with 4. She sells the rest at $2 each.
How much does she make?]

[TUTOR: Great question! Let's break this down.

Step 1: Find eggs remaining after Janet uses some.
- Total eggs: 16
- Eaten for breakfast: 3
- Used for baking: 4
- Remaining: 16 - 3 - 4 = 9 eggs

Step 2: Calculate earnings from selling.
- Eggs sold: 9
- Price per egg: $2
- Earnings: 9 × $2 = $18

Janet makes $18 per day at the farmers' market.]

[STUDENT: That makes sense! Thank you.]""",

    verifier=lambda text: _verify_bracket_speaker(text)
)


def _verify_bracket_speaker(text: str) -> bool:
    """Verify bracket-speaker format - balanced brackets, valid speakers."""
    # Check for balanced brackets with speaker pattern
    pattern = r'\[([A-Z_]+):'
    opens = re.findall(pattern, text)
    closes = text.count(']')
    # Should have roughly equal opens and closes
    return len(opens) > 0 and abs(len(opens) - closes) <= 1


QA_SIMPLE_SPEC = FormatSpec(
    name="qa_simple",
    explanation="""## How to Write Q&A Format

Q&A format is a simple, clear way to structure questions and answers.

**Questions**: Start with "Q:" followed by a space, then the question.
```
Q: What is the capital of France?
```

**Answers**: Start with "A:" followed by a space, then the answer.
```
A: The capital of France is Paris.
```

**Multi-line answers**: Continue on subsequent lines, optionally indented.
```
A: Let me explain step by step.
   First, we calculate the total.
   Then, we subtract the used amount.
   Finally, we multiply by the price.
```

**Calculations**: Show work inline using <<expression=result>> notation.
```
A: Janet has 16 - 3 - 4 = <<16-3-4=9>>9 eggs to sell.
```

**Final answers**: Mark with #### followed by the answer.
```
#### 18
```

**Multiple Q&A pairs**: Separate with blank lines.
```
Q: First question?
A: First answer.

Q: Second question?
A: Second answer.
```""",

    example="""Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes
muffins with 4. She sells the rest at $2 per egg. How much does she make daily?

A: Let's solve this step by step.

First, find how many eggs Janet has left to sell:
Total eggs: 16
Minus breakfast: 16 - 3 = <<16-3=13>>13
Minus baking: 13 - 4 = <<13-4=9>>9 eggs remaining

Next, calculate her earnings:
Eggs to sell: 9
Price per egg: $2
Daily earnings: 9 × 2 = <<9*2=18>>$18

#### 18""",

    verifier=lambda text: (
        bool(re.search(r'^Q:', text, re.MULTILINE)) and
        bool(re.search(r'^A:', text, re.MULTILINE))
    )
)


INDENTATION_SCOPE_SPEC = FormatSpec(
    name="indentation_scope",
    explanation="""## How to Write Indentation-Scope Format

Indentation-Scope uses whitespace indentation to show logical hierarchy and scope.
Each indentation level (2 spaces) represents a deeper level of detail or sub-step.

**Scope levels**:
- Level 0 (no indent): Main statements, problem description, final answers
- Level 1 (2 spaces): Major steps, primary reasoning
- Level 2 (4 spaces): Sub-steps, calculations, details
- Level 3 (6 spaces): Verification, notes, fine details

**Rules**:
- Use exactly 2 spaces per indentation level
- Don't skip levels (go 0→1→2, not 0→2)
- Deeper = more specific/detailed
- Same level = same importance/type

**Transitions**:
- Indent to go deeper into detail
- Dedent to return to higher-level reasoning
- Stay at same level for parallel items

**Example structure**:
```
Problem statement at level 0
  Major step 1
    Detail of step 1
    Another detail
  Major step 2
    Sub-step 2a
      Calculation for 2a
    Sub-step 2b
Final answer at level 0
```""",

    example="""Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes with 4.
She sells the rest at $2 each. How much does she earn?

  Step 1: Calculate eggs remaining
    Start with total: 16 eggs
    Subtract breakfast: 16 - 3 = 13
    Subtract baking: 13 - 4 = 9
      Verify: 3 + 4 + 9 = 16 ✓
    Eggs to sell: 9

  Step 2: Calculate earnings
    Eggs available: 9
    Price per egg: $2
    Earnings: 9 × $2 = $18
      Verify: $18 ÷ $2 = 9 eggs ✓

Janet earns $18 per day at the farmers' market.""",

    verifier=lambda text: bool(re.search(r'^  \S', text, re.MULTILINE))
)


# =============================================================================
# Format Registry
# =============================================================================

FORMAT_REGISTRY: dict[str, FormatSpec] = {
    # Core formats (real, widely used)
    "markdown": MARKDOWN_SPEC,
    "json": JSON_SPEC,
    "yaml": YAML_SPEC,

    # Task-specific formats
    "qa_simple": QA_SIMPLE_SPEC,

    # Experimental/fictional formats
    "markerdown": MARKERDOWN_SPEC,
    "bracket_speaker": BRACKET_SPEAKER_SPEC,
    "indentation_scope": INDENTATION_SCOPE_SPEC,
}


def get_format_spec(name: str) -> Optional[FormatSpec]:
    """Get a format specification by name."""
    return FORMAT_REGISTRY.get(name)


def list_formats() -> list[str]:
    """List all available format names."""
    return list(FORMAT_REGISTRY.keys())


def verify_format(text: str, format_name: str) -> bool:
    """Verify text adheres to a format specification."""
    spec = FORMAT_REGISTRY.get(format_name)
    if spec is None or spec.verifier is None:
        return True  # No verifier = assume valid
    return spec.verifier(text)
