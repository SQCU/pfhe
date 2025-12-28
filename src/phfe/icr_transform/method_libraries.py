"""
Method Libraries for ICR Transformation

Each library provides:
1. Named methods/patterns with clear "when to use" signals
2. Worked examples showing the method in action
3. A consistent format for prepending to problems

These are the "contexts" in In-Context Retrieval - they make
implicit knowledge explicit so the model can apply it.
"""

# =============================================================================
# GSM1K - Grade School Math
# =============================================================================

GSM1K_METHOD_LIBRARY = """[METHOD LIBRARY: Arithmetic Word Problems]

METHOD 1: COMBINING QUANTITIES (Addition)
When to use: Someone gets more, receives, buys, or finding a total
Signal words: "altogether", "in total", "combined", "plus", "more", "and"
Example: Sam has 12 apples. He buys 8 more. How many now?
→ Combining: 12 + 8 = 20 apples.

METHOD 2: FINDING WHAT'S LEFT (Subtraction)
When to use: Something removed, used, given away, or finding a difference
Signal words: "left", "remaining", "gave away", "fewer", "less", "difference"
Example: 45 cookies on a plate. Kids eat 17. How many left?
→ What's left: 45 - 17 = 28 cookies.

METHOD 3: REPEATED GROUPS (Multiplication)
When to use: Equal groups, "each" quantities, rates
Signal words: "each", "per", "every", "times", "groups of"
Example: 6 boxes with 8 toys each. How many toys total?
→ Repeated groups: 6 × 8 = 48 toys.

METHOD 4: SPLITTING EVENLY (Division)
When to use: Sharing equally, finding group size, finding how many groups
Signal words: "split", "share", "each gets", "divided", "equally"
Example: 36 stickers shared among 4 kids. How many each?
→ Splitting: 36 ÷ 4 = 9 stickers each.

METHOD 5: MULTI-STEP PROBLEMS
When to use: Problem has multiple parts or stages
Strategy: Identify each step, solve in order, use earlier results
Example: 5 bags with 6 apples each. Give away 10 apples. How many left?
→ Step 1: Total apples: 5 × 6 = 30
→ Step 2: After giving: 30 - 10 = 20 apples.

METHOD 6: FRACTIONS AND PARTS
When to use: Portions, parts of a whole, percentages
Signal words: "half", "third", "quarter", "percent", "fraction", "part of"
Example: A pizza has 8 slices. Tom eats 3/8. How many slices left?
→ Parts: 8 × (3/8) = 3 eaten, 8 - 3 = 5 slices left.

[PROBLEM]
"""

# =============================================================================
# ARC - Science Reasoning
# =============================================================================

ARC_SCIENCE_LIBRARY = """[SCIENCE FACT LIBRARY]

CATEGORY: States of Matter
• Solids have fixed shape and volume
• Liquids have fixed volume but take container's shape
• Gases expand to fill any container
• Heating causes expansion, cooling causes contraction
• Phase changes: solid ↔ liquid ↔ gas (melting, freezing, boiling, condensation)

CATEGORY: Forces and Motion
• Objects at rest stay at rest unless acted on by a force (inertia)
• Friction opposes motion and generates heat
• Gravity pulls objects toward Earth (9.8 m/s²)
• Heavier objects don't fall faster (ignoring air resistance)
• Action-reaction: every force has an equal and opposite force

CATEGORY: Energy
• Energy cannot be created or destroyed, only transformed
• Forms: kinetic (motion), potential (stored), thermal (heat), chemical, electrical
• Heat flows from hot to cold
• Light travels in straight lines at 300,000 km/s
• Sound needs a medium (air, water, solid) to travel

CATEGORY: Living Things
• All living things are made of cells
• Plants make food via photosynthesis (sunlight + CO₂ + water → sugar + O₂)
• Animals get energy by consuming other organisms
• Ecosystems have producers, consumers, and decomposers
• Adaptations help organisms survive in their environment

CATEGORY: Earth and Space
• Earth rotates (day/night) and revolves around Sun (seasons)
• Moon phases caused by Moon's orbit around Earth
• Weathering and erosion shape landforms
• Water cycle: evaporation → condensation → precipitation
• Fossils show evidence of past life

[WORKED EXAMPLES]
Q: What happens to water when heated to 100°C at sea level?
A: It boils and becomes steam (liquid → gas phase transition).

Q: Why does a ball roll farther on smooth floor than rough carpet?
A: Less friction on smooth surface allows more motion before stopping.

Q: Why do plants grow toward light?
A: Phototropism - plants bend toward light for photosynthesis.

[PROBLEM]
"""

# =============================================================================
# RACE - Reading Comprehension
# =============================================================================

RACE_STRATEGY_LIBRARY = """[READING COMPREHENSION STRATEGIES]

STRATEGY 1: Main Idea Questions
Signal words: "mainly about", "best title", "central theme"
Approach: Look at first/last paragraphs, topic sentences
The answer covers the WHOLE passage, not just one detail

STRATEGY 2: Detail Questions
Signal words: "according to", "the author states", "which of the following"
Approach: Find the specific sentence, read context around it
The answer is directly stated or closely paraphrased

STRATEGY 3: Inference Questions
Signal words: "implies", "suggests", "can be inferred", "most likely"
Approach: Combine stated facts to reach unstated conclusion
The answer goes ONE step beyond what's written

STRATEGY 4: Vocabulary in Context
Signal words: "the word X means", "as used in line Y"
Approach: Read the sentence, substitute answer choices
The meaning fits THIS context, not necessarily dictionary definition

STRATEGY 5: Author's Purpose/Tone
Signal words: "purpose", "attitude", "tone", "point of view"
Approach: Consider word choice, what's emphasized, what's left out
Common tones: objective, critical, enthusiastic, skeptical, humorous

STRATEGY 6: Sequence/Cause-Effect
Signal words: "led to", "resulted in", "before/after", "because"
Approach: Map out the chain of events or reasoning
Look for transition words: therefore, consequently, as a result

[PASSAGE]
"""

# =============================================================================
# BoolQ - Boolean Question Answering
# =============================================================================

BOOLQ_STRATEGY_LIBRARY = """[YES/NO QUESTION STRATEGIES]

APPROACH 1: Find the Claim
The question asks if something is TRUE or FALSE
Locate the specific claim in the passage

APPROACH 2: Check Exact Wording
"Always/never" claims need NO exceptions to be true
"Some/can/may" claims need just ONE example to be true
Qualifiers matter: "usually", "often", "sometimes"

APPROACH 3: Beware of Partial Matches
The passage might discuss the topic but not confirm the claim
"X is mentioned" ≠ "X is true"

APPROACH 4: Negation Traps
Double negatives: "not uncommon" = "common"
Watch for "except", "unless", "without"

[PASSAGE]
"""

# =============================================================================
# HellaSwag - Commonsense Completion
# =============================================================================

HELLASWAG_STRATEGY_LIBRARY = """[COMMONSENSE COMPLETION STRATEGIES]

PRINCIPLE 1: Physical Plausibility
Actions must obey physics (gravity, momentum, etc.)
Objects can't appear/disappear without explanation
People have normal human capabilities

PRINCIPLE 2: Temporal Coherence
Events happen in logical order
Causes precede effects
Actions take appropriate time

PRINCIPLE 3: Goal-Directed Behavior
People act to achieve goals
Actions should make progress toward stated objective
Random or counterproductive actions are unlikely

PRINCIPLE 4: Social Norms
People usually follow social conventions
Extremely rude/bizarre behavior needs context
Consider the setting (formal vs. casual)

PRINCIPLE 5: Narrative Continuity
Characters maintain their identity
Objects/settings persist
Tone remains consistent

[COMMON TRAPS]
• Non-sequiturs that suddenly change topic
• Physically impossible actions
• Illogical reversals of progress
• Uncharacteristic behavior
• Anachronisms (mixing time periods)

[CONTEXT]
"""

# =============================================================================
# WinoGrande - Coreference Resolution
# =============================================================================

WINOGRANDE_STRATEGY_LIBRARY = """[COREFERENCE RESOLUTION STRATEGIES]

The Task: Determine what "they/it/he/she" refers to

STRATEGY 1: Semantic Fit
Which referent makes the sentence meaningful?
Test by substituting each option

STRATEGY 2: World Knowledge
Use common sense about how things work
Consider typical properties of objects/people

STRATEGY 3: Contrast Clues
Often the sentence contrasts two things
The pronoun refers to whichever fits the predicate

STRATEGY 4: Cause-Effect Logic
If there's a because/so/therefore:
- The pronoun is usually the cause or effect

EXAMPLES:
"The trophy doesn't fit in the suitcase because it is too big."
→ "it" = trophy (trophies can be big; suitcases being big would help)

"The trophy doesn't fit in the suitcase because it is too small."
→ "it" = suitcase (suitcase being small prevents fitting)

[SENTENCE]
"""

# =============================================================================
# MBPP / Code - Programming Patterns
# =============================================================================

MBPP_PATTERN_LIBRARY = """[PROGRAMMING PATTERN LIBRARY]

PATTERN 1: List Iteration & Transformation
When: Process each element, filter, or accumulate
```python
def process(items):
    result = []
    for item in items:
        if condition(item):       # optional filter
            result.append(transform(item))
    return result
# Or: [transform(x) for x in items if condition(x)]
```

PATTERN 2: Dictionary Building
When: Group items, count occurrences, create mapping
```python
def build_dict(items):
    result = {}
    for item in items:
        key = get_key(item)
        result[key] = result.get(key, 0) + 1  # counting
        # or: result.setdefault(key, []).append(item)  # grouping
    return result
```

PATTERN 3: Two-Pointer / Sliding Window
When: Find pairs, palindromes, or subarrays
```python
def two_pointer(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # compare arr[left], arr[right]
        # move pointers based on condition
        left += 1
        right -= 1
```

PATTERN 4: Recursion with Base Case
When: Self-similar subproblems (trees, math sequences)
```python
def solve(n):
    if n <= base_case:
        return base_value
    return combine(solve(smaller_n))
```

PATTERN 5: String Processing
When: Parse, validate, or transform strings
```python
def process_string(s):
    result = []
    for char in s:
        if char.isalpha():
            result.append(char.lower())
    return ''.join(result)
```

PATTERN 6: Mathematical Formulas
When: Number theory, geometry, statistics
- Factorial: n! = n × (n-1) × ... × 1
- Fibonacci: fib(n) = fib(n-1) + fib(n-2)
- GCD: use Euclidean algorithm
- Prime check: test divisibility up to √n

[PROBLEM]
"""

# =============================================================================
# Function Calling
# =============================================================================

FNCALL_API_LIBRARY = """[FUNCTION CALLING GUIDE]

STEP 1: Identify the User Intent
What does the user want to accomplish?
Which function(s) can achieve this?

STEP 2: Extract Parameters
Find required parameters in the user request
Infer reasonable defaults for optional parameters
Ask for clarification if required info is missing

STEP 3: Format the Call
Use exact parameter names from the function signature
Match types (string, int, list, etc.)
Handle nested objects if required

[AVAILABLE FUNCTIONS]

get_weather(location: str, units: str = "celsius") -> dict
    Returns current weather for location.
    Example: get_weather("Tokyo", "celsius")
    → {"temp": 22, "conditions": "cloudy", "humidity": 65}

search_web(query: str, num_results: int = 5) -> list[dict]
    Searches web, returns top results.
    Example: search_web("python tutorials", 3)
    → [{"title": ..., "url": ..., "snippet": ...}, ...]

send_email(to: str, subject: str, body: str, cc: list[str] = None) -> dict
    Sends email, returns status.
    Example: send_email("bob@example.com", "Hello", "Hi Bob!")
    → {"success": true, "message_id": "msg_123"}

calculate(expression: str) -> float
    Evaluates mathematical expression.
    Example: calculate("(15 + 27) * 3") → 126.0

get_stock_price(symbol: str) -> dict
    Returns current stock price.
    Example: get_stock_price("AAPL")
    → {"price": 182.52, "currency": "USD", "change": +1.23}

create_calendar_event(title: str, start: str, end: str, attendees: list[str] = None) -> dict
    Creates calendar event.
    Example: create_calendar_event("Meeting", "2024-01-15T10:00", "2024-01-15T11:00")
    → {"event_id": "evt_456", "link": "https://..."}

[REQUEST]
"""

# =============================================================================
# Format Transformation
# =============================================================================

FORMAT_TRANSFORM_LIBRARY = r"""[DATA FORMAT TRANSFORMATION GUIDE]

PATTERN 1: JSON ↔ Other Formats
```python
import json
data = json.loads(json_string)  # parse
json_string = json.dumps(data, indent=2)  # serialize
```

PATTERN 2: CSV Processing
```python
import csv
with open('file.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        process(row['column_name'])
```

PATTERN 3: XML/HTML Parsing
```python
from xml.etree import ElementTree as ET
root = ET.fromstring(xml_string)
for elem in root.findall('.//tag'):
    value = elem.text
```

PATTERN 4: Date/Time Formatting
```python
from datetime import datetime
dt = datetime.strptime("2024-01-15", "%Y-%m-%d")
formatted = dt.strftime("%B %d, %Y")  # "January 15, 2024"
```

PATTERN 5: Nested Structure Flattening
```python
def flatten(nested, prefix=''):
    result = {}
    for key, value in nested.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, new_key))
        else:
            result[new_key] = value
    return result
```

PATTERN 6: Regex Extraction
```python
import re
matches = re.findall(r'pattern', text)
groups = re.search(r'(\d+)-(\d+)', text).groups()
cleaned = re.sub(r'\s+', ' ', text)
```

[TASK]
"""

# =============================================================================
# Registry
# =============================================================================

METHOD_LIBRARIES = {
    "gsm1k": GSM1K_METHOD_LIBRARY,
    "arc": ARC_SCIENCE_LIBRARY,
    "arc_easy": ARC_SCIENCE_LIBRARY,
    "arc_challenge": ARC_SCIENCE_LIBRARY,
    "race": RACE_STRATEGY_LIBRARY,
    "boolq": BOOLQ_STRATEGY_LIBRARY,
    "hellaswag": HELLASWAG_STRATEGY_LIBRARY,
    "winogrande": WINOGRANDE_STRATEGY_LIBRARY,
    "mbpp": MBPP_PATTERN_LIBRARY,
    "fncall": FNCALL_API_LIBRARY,
    "format": FORMAT_TRANSFORM_LIBRARY,
}


def get_method_library(benchmark: str) -> str:
    """Get the method library for a benchmark."""
    return METHOD_LIBRARIES.get(benchmark.lower(), "")


def list_available_libraries() -> list[str]:
    """List benchmarks with method libraries."""
    return list(METHOD_LIBRARIES.keys())
