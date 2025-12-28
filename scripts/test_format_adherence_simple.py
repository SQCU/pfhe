#!/usr/bin/env python3
"""
Simple Format Adherence Test (transformers backend)

Tests format adherence without vLLM dependency issues.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_format_generation(model_name: str = "Qwen/Qwen3-0.6B"):
    """Test if model can follow format specs."""

    # Import format specs
    import sys
    sys.path.insert(0, 'src')
    from phfe.icr_transform.format_specs import get_format_spec

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    formats_to_test = ["markerdown", "bracket_speaker", "qa_simple"]

    for fmt_name in formats_to_test:
        spec = get_format_spec(fmt_name)
        if not spec:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing format: {fmt_name}")
        logger.info(f"{'='*60}")

        # Build prompt
        prompt = f"""# Format Specification

{spec.explanation}

# Example

{spec.example}

---

# Task

Write a solution to this problem using the {fmt_name} format:
"A store sells apples for $2 each. How much for 5 apples?"

# Solution

"""
        logger.info(f"Prompt length: {len(prompt)} chars")

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Verify
        adheres = False
        if spec.verifier:
            try:
                adheres = spec.verifier(generated)
            except Exception as e:
                logger.warning(f"Verifier error: {e}")

        logger.info(f"\nGenerated:")
        logger.info("-" * 40)
        logger.info(generated[:500])
        logger.info("-" * 40)
        logger.info(f"Format adherence: {'PASS' if adheres else 'FAIL'}")


if __name__ == "__main__":
    test_format_generation()
