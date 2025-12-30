#!/usr/bin/env python3
"""
Format Self-Evaluation with Structured Output

Demonstrates edge models both:
1. USING a format (generating markerdown)
2. PARSING format success (self-evaluating via structured JSON output)

Key insight: LLM judgment of qualitative features requires parseable output fields.
We use JSON-demarcated response format for mechanical extraction of verdicts.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phfe.icr_transform.format_specs import (
    get_format_spec,
    verify_format,
    MARKERDOWN_SPEC,
    FORMAT_REGISTRY,
)


@dataclass
class FormatEvalResult:
    """Result of format use + self-evaluation."""

    format_name: str
    problem: str
    generated_text: str

    # Mechanical verification
    mechanical_pass: bool

    # Self-evaluation (model judges its own output)
    self_eval_verdict: Optional[str] = None  # PASS/FAIL/UNCLEAR
    self_eval_issues: Optional[list[str]] = None
    self_eval_raw: Optional[str] = None

    # Cross-evaluation (another model judges)
    cross_eval_verdict: Optional[str] = None
    cross_eval_issues: Optional[list[str]] = None


# =============================================================================
# Structured Output Prompts
# =============================================================================

GENERATION_PROMPT = """# Format Specification

{format_spec}

# Example

{example}

---

# Task

Write a solution to the following problem using the {format_name} format EXACTLY as shown above.

Problem: {problem}

# Solution (in {format_name} format)

"""

SELF_EVAL_PROMPT = """You are evaluating whether a piece of text follows a specific format.
You MUST respond with a JSON object only. No other text.

## Format Specification

{format_spec}

## Text to Evaluate

{text}

## Response Format

Respond with ONLY a JSON object in this exact format:
```json
{{
  "verdict": "PASS" or "FAIL",
  "issues": ["list of specific format violations, empty if PASS"],
  "format_elements_found": ["list of format elements correctly used"]
}}
```

## Evaluation

```json
"""

CROSS_EVAL_PROMPT = """You are a format adherence evaluator. Another model generated the following text
attempting to follow a specific format. Judge whether it succeeded.

You MUST respond with a JSON object only. No other text.

## Target Format: {format_name}

### Format Rules
{format_spec}

## Generated Text to Evaluate

{text}

## Response Format

```json
{{
  "verdict": "PASS" or "FAIL",
  "issues": ["list of specific violations"],
  "score": 0.0 to 1.0
}}
```

## Evaluation

```json
"""


def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from model response."""
    # Try to find JSON block
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown code block
        r'```\s*(.*?)\s*```',       # Generic code block
        r'\{[^{}]*\}',              # Raw JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # Last resort: try parsing the whole thing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# =============================================================================
# Inference Backend - HuggingFace Transformers
# =============================================================================

# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def resolve_model_id(model_path: str) -> str:
    """Map GGUF paths or shortcuts to HuggingFace model IDs."""
    # Direct HF model IDs
    if model_path.startswith("google/") or model_path.startswith("meta-llama/"):
        return model_path

    # Map GGUF paths to HF IDs
    path_lower = model_path.lower()
    if "gemma-3-27b" in path_lower:
        return "google/gemma-3-27b-it"
    elif "gemma-3-4b" in path_lower:
        return "google/gemma-3-4b-it"
    elif "gemma-2-9b" in path_lower:
        return "google/gemma-2-9b-it"
    else:
        # Assume it's already a HF model ID
        return model_path


def get_tokenizer(model_path: str):
    """Get or create tokenizer for model."""
    model_id = resolve_model_id(model_path)

    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {model_id}")
        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)

    return _TOKENIZER_CACHE[model_id]


def get_model(model_path: str, device: str = "cuda"):
    """Get or create HuggingFace model instance."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_id = resolve_model_id(model_path)

    if model_id not in _MODEL_CACHE:
        print(f"Loading model: {model_id}")

        # Use 4-bit quantization for large models (27B+)
        if "27b" in model_id.lower():
            print("Using 4-bit quantization for 27B model...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            _MODEL_CACHE[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            _MODEL_CACHE[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )

        print(f"Model loaded: {sum(p.numel() for p in _MODEL_CACHE[model_id].parameters())/1e9:.1f}B params")

    return _MODEL_CACHE[model_id]


def generate_with_gguf_primitives(
    prompt: str,
    model_path: str,
    max_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    """
    Generate using HuggingFace transformers.

    Note: Named 'gguf_primitives' for API compatibility, but uses HF backend
    for faster loading. Our GGUF dequantization is correct but slow;
    for production, add GPU-accelerated dequant kernels.
    """
    import torch

    model = get_model(model_path, device)
    tokenizer = get_tokenizer(model_path)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text


# =============================================================================
# Format Self-Evaluation Loop
# =============================================================================

def run_format_self_eval(
    model_path: str,
    format_name: str = "markerdown",
    problems: Optional[list[str]] = None,
    do_cross_eval: bool = False,
    cross_eval_model: Optional[str] = None,
) -> list[FormatEvalResult]:
    """
    Run the format use + self-evaluation loop.

    1. Generate text in the target format
    2. Mechanically verify format adherence
    3. Have the model self-evaluate its output
    4. Optionally have another model cross-evaluate
    """
    spec = get_format_spec(format_name)
    if not spec:
        raise ValueError(f"Unknown format: {format_name}")

    if problems is None:
        problems = [
            "A store sells apples for $2 each. If Maria buys 5 apples, how much does she pay?",
            "A train travels 60 miles per hour. How far does it travel in 3 hours?",
            "A rectangle has length 8 and width 5. What is its area?",
        ]

    results = []

    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem[:50]}...")
        print(f"{'='*60}")

        # === Step 1: Generate in format ===
        gen_prompt = GENERATION_PROMPT.format(
            format_spec=spec.explanation,
            example=spec.example,
            format_name=format_name,
            problem=problem,
        )

        print(f"\n[1] Generating in {format_name} format...")
        generated = generate_with_gguf_primitives(gen_prompt, model_path)
        print(f"Generated ({len(generated)} chars):")
        print("-" * 40)
        print(generated[:600])
        if len(generated) > 600:
            print("... [truncated]")
        print("-" * 40)

        # === Step 2: Mechanical verification ===
        mechanical_pass = verify_format(generated, format_name)
        print(f"\n[2] Mechanical verification: {'PASS' if mechanical_pass else 'FAIL'}")

        # === Step 3: Self-evaluation with structured output ===
        print(f"\n[3] Self-evaluation (structured JSON output)...")
        self_eval_prompt = SELF_EVAL_PROMPT.format(
            format_spec=spec.explanation,
            text=generated,
        )

        self_eval_response = generate_with_gguf_primitives(
            self_eval_prompt,
            model_path,
        )
        print(f"Raw self-eval response:")
        print(self_eval_response[:400])

        self_eval_parsed = parse_json_response(self_eval_response)

        if self_eval_parsed:
            self_verdict = self_eval_parsed.get("verdict", "UNCLEAR")
            self_issues = self_eval_parsed.get("issues", [])
            print(f"\nParsed self-eval: verdict={self_verdict}, issues={self_issues}")
        else:
            self_verdict = "PARSE_ERROR"
            self_issues = ["Failed to parse JSON response"]
            print(f"\nFailed to parse JSON from self-eval response")

        # Build result
        result = FormatEvalResult(
            format_name=format_name,
            problem=problem,
            generated_text=generated,
            mechanical_pass=mechanical_pass,
            self_eval_verdict=self_verdict,
            self_eval_issues=self_issues,
            self_eval_raw=self_eval_response,
        )

        # === Step 4: Cross-evaluation (optional) ===
        if do_cross_eval and cross_eval_model:
            print(f"\n[4] Cross-evaluation by different model...")
            cross_prompt = CROSS_EVAL_PROMPT.format(
                format_name=format_name,
                format_spec=spec.explanation,
                text=generated,
            )

            cross_response = generate_with_gguf_primitives(cross_prompt, cross_eval_model)
            cross_parsed = parse_json_response(cross_response)

            if cross_parsed:
                result.cross_eval_verdict = cross_parsed.get("verdict", "UNCLEAR")
                result.cross_eval_issues = cross_parsed.get("issues", [])
                print(f"Cross-eval: verdict={result.cross_eval_verdict}")

        results.append(result)

    return results


def print_summary(results: list[FormatEvalResult]):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    mechanical_pass = sum(1 for r in results if r.mechanical_pass)
    self_pass = sum(1 for r in results if r.self_eval_verdict == "PASS")

    print(f"Total samples: {len(results)}")
    print(f"Mechanical verification: {mechanical_pass}/{len(results)} PASS")
    print(f"Self-evaluation: {self_pass}/{len(results)} PASS")

    # Agreement analysis
    agree = sum(
        1 for r in results
        if (r.mechanical_pass and r.self_eval_verdict == "PASS") or
           (not r.mechanical_pass and r.self_eval_verdict == "FAIL")
    )
    print(f"Mechanical/Self agreement: {agree}/{len(results)}")

    # Common issues
    all_issues = []
    for r in results:
        if r.self_eval_issues:
            all_issues.extend(r.self_eval_issues)

    if all_issues:
        print(f"\nCommon issues identified:")
        for issue in set(all_issues):
            count = all_issues.count(issue)
            print(f"  - {issue} ({count}x)")


def main():
    parser = argparse.ArgumentParser(description="Format self-evaluation demo")
    parser.add_argument(
        "--model",
        default="/mnt/f/dox/ai/text/models/gemma-3-4b-it-Q8_0.gguf",
        help="Path to GGUF model"
    )
    parser.add_argument(
        "--format",
        default="markerdown",
        choices=list(FORMAT_REGISTRY.keys()),
        help="Format to test"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="Custom problems to test (optional)"
    )
    parser.add_argument(
        "--cross-eval",
        help="Second model path for cross-evaluation"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    print("="*60)
    print("FORMAT SELF-EVALUATION DEMO")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Cross-eval: {args.cross_eval or 'None'}")

    results = run_format_self_eval(
        model_path=args.model,
        format_name=args.format,
        problems=args.problems,
        do_cross_eval=bool(args.cross_eval),
        cross_eval_model=args.cross_eval,
    )

    print_summary(results)

    if args.output:
        # Serialize results
        output_data = []
        for r in results:
            output_data.append({
                "format": r.format_name,
                "problem": r.problem,
                "generated": r.generated_text,
                "mechanical_pass": r.mechanical_pass,
                "self_eval_verdict": r.self_eval_verdict,
                "self_eval_issues": r.self_eval_issues,
                "cross_eval_verdict": r.cross_eval_verdict,
                "cross_eval_issues": r.cross_eval_issues,
            })

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
