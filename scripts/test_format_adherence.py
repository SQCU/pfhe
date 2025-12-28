#!/usr/bin/env python3
"""
Test Format Adherence with Local Models

Tests whether models can follow text format specifications described in-context.
Uses the self-vendored vLLM inferencer with local GGUF models.

Usage:
    # Test with gemma-3-27b-it (local GGUF)
    python scripts/test_format_adherence.py

    # Test specific format
    python scripts/test_format_adherence.py --format markerdown

    # Use as judge
    python scripts/test_format_adherence.py --judge
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_generation_adherence(
    model: str,
    formats: list[str],
    n_samples: int = 2,
):
    """Test if model can generate text following format specs."""
    from phfe.inference.engine import VLLMEngine
    from phfe.icr_transform.format_specs import get_format_spec, list_formats
    from phfe.icr_transform.template_generator import generate_format_adherence_test

    logger.info(f"Loading model: {model}")
    engine = VLLMEngine(model_path=model, max_tokens=1024)

    results = []

    for fmt in formats:
        spec = get_format_spec(fmt)
        if not spec:
            logger.warning(f"Unknown format: {fmt}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing format: {fmt}")
        logger.info(f"{'='*60}")

        for i in range(n_samples):
            # Build test prompt
            prompt = f"""# Format Specification

{spec.explanation}

# Example

{spec.example}

---

# Task

Continue this document by adding a new section about a math problem:
"A bakery sells cookies for $3 each. If someone buys 7 cookies, how much do they pay?"

Write the problem and solution following the {fmt} format EXACTLY as shown above.

# New Section

"""
            logger.info(f"\nSample {i+1}/{n_samples}:")
            logger.info(f"Generating with format spec for {fmt}...")

            result = engine.generate(prompt, temperature=0.7)

            # Verify
            adheres = False
            if spec.verifier:
                try:
                    adheres = spec.verifier(result.generated_text)
                except Exception as e:
                    logger.warning(f"Verifier error: {e}")

            logger.info(f"Generated ({result.tokens_generated} tokens):")
            logger.info("-" * 40)
            logger.info(result.generated_text[:800])
            if len(result.generated_text) > 800:
                logger.info("... [truncated]")
            logger.info("-" * 40)
            logger.info(f"Format adherence (mechanical check): {'PASS' if adheres else 'FAIL'}")

            results.append({
                "format": fmt,
                "sample": i + 1,
                "adheres": adheres,
                "tokens": result.tokens_generated,
                "generated": result.generated_text,
            })

    return results


def test_judge_capability(
    model: str,
    formats: list[str],
):
    """Test if model can judge format adherence."""
    from phfe.inference.engine import VLLMEngine
    from phfe.icr_transform.format_specs import get_format_spec

    logger.info(f"Loading model for judging: {model}")
    engine = VLLMEngine(model_path=model, max_tokens=1024)

    results = []

    for fmt in formats:
        spec = get_format_spec(fmt)
        if not spec:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing judge capability for: {fmt}")
        logger.info(f"{'='*60}")

        # Test with a correct example
        logger.info("\nJudging CORRECT example:")
        correct_result = engine.judge_format(
            text=spec.example,
            format_description=spec.explanation,
        )
        logger.info(f"Verdict: {correct_result['verdict']}")
        logger.info(f"Reasoning excerpt: {correct_result['reasoning'][:500]}...")

        # Test with an incorrect example (plain text)
        incorrect_text = """
This is just plain text without any special formatting.
It doesn't follow the rules at all.
No markers, no brackets, no nothing.
"""
        logger.info("\nJudging INCORRECT example:")
        incorrect_result = engine.judge_format(
            text=incorrect_text,
            format_description=spec.explanation,
        )
        logger.info(f"Verdict: {incorrect_result['verdict']}")
        logger.info(f"Reasoning excerpt: {incorrect_result['reasoning'][:500]}...")

        results.append({
            "format": fmt,
            "correct_example_verdict": correct_result['verdict'],
            "incorrect_example_verdict": incorrect_result['verdict'],
            "judge_accuracy": (
                correct_result['verdict'] == "PASS" and
                incorrect_result['verdict'] == "FAIL"
            ),
        })

    return results


def test_multi_format_selection(model: str):
    """
    Test if model can select and follow the correct format when given multiple options.

    This is the key test: present 4 formats, say "this document uses format X",
    and see if the model continues with format X.
    """
    from phfe.inference.engine import VLLMEngine
    from phfe.icr_transform.format_specs import get_format_spec

    logger.info(f"Loading model: {model}")
    engine = VLLMEngine(model_path=model, max_tokens=1024)

    formats = ["qa_simple", "markerdown", "bracket_speaker", "indentation_scope"]
    results = []

    for target_format in formats:
        logger.info(f"\n{'='*60}")
        logger.info(f"Target format: {target_format}")
        logger.info(f"{'='*60}")

        # Build prompt with ALL format descriptions but only ONE being "active"
        format_descriptions = []
        for i, fmt in enumerate(formats, 1):
            spec = get_format_spec(fmt)
            format_descriptions.append(
                f"## Format {i}: {fmt}\n\n{spec.explanation[:500]}..."
            )

        prompt = f"""# Available Text Formats

This document describes 4 different text formatting styles.

{chr(10).join(format_descriptions)}

---

# Document Content

**This document uses the {target_format} format.**

Here is an example problem and solution in {target_format} format:

Problem: Calculate 15% tip on a $40 meal.

"""
        logger.info(f"Prompt asks for {target_format} format...")
        result = engine.generate(prompt, temperature=0.5)

        # Check which format the output most resembles
        target_spec = get_format_spec(target_format)
        adheres_to_target = False
        if target_spec and target_spec.verifier:
            try:
                adheres_to_target = target_spec.verifier(result.generated_text)
            except Exception:
                pass

        logger.info(f"Generated:")
        logger.info("-" * 40)
        logger.info(result.generated_text[:600])
        logger.info("-" * 40)
        logger.info(f"Adheres to {target_format}: {'YES' if adheres_to_target else 'NO'}")

        results.append({
            "target_format": target_format,
            "adheres": adheres_to_target,
            "generated": result.generated_text[:500],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test format adherence")
    parser.add_argument(
        "--model",
        default="gemma-3-27b-it-q4",
        help="Model to use (shortcut or path)"
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["markerdown", "bracket_speaker", "qa_simple"],
        help="Formats to test"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Samples per format"
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Test judging capability instead of generation"
    )
    parser.add_argument(
        "--multi-format",
        action="store_true",
        help="Test multi-format selection"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("FORMAT ADHERENCE TEST")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Formats: {args.format}")

    if args.judge:
        results = test_judge_capability(args.model, args.format)
    elif args.multi_format:
        results = test_multi_format_selection(args.model)
    else:
        results = test_generation_adherence(
            args.model, args.format, args.samples
        )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("="*60)

    if args.judge:
        accurate = sum(1 for r in results if r.get('judge_accuracy', False))
        logger.info(f"Judge accuracy: {accurate}/{len(results)}")
    else:
        passed = sum(1 for r in results if r.get('adheres', False))
        logger.info(f"Format adherence: {passed}/{len(results)}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
