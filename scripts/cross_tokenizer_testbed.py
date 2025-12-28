#!/usr/bin/env python3
"""
Cross-Tokenizer GKD Testbed

Tests cross-tokenizer alignment and GKD loss computation with real models:
- Student: Qwen2-0.5B (small, fast)
- Teacher: Gemma-2-2B (different tokenizer family)

This validates that our cross-tokenizer distillation approach works
with real BPE tokenizers that have different vocabularies.

Usage:
    python scripts/cross_tokenizer_testbed.py --tokenizers-only
    python scripts/cross_tokenizer_testbed.py --full
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Lazy imports for optional dependencies
def get_transformers():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        return AutoTokenizer, AutoModelForCausalLM, torch
    except ImportError:
        print("Please install transformers: pip install transformers torch")
        raise


@dataclass
class TokenizerStats:
    """Statistics about a tokenizer."""
    name: str
    vocab_size: int
    sample_tokens: list[str]
    encode_time_ms: float


@dataclass
class AlignmentStats:
    """Statistics about cross-tokenizer alignment."""
    text: str
    teacher_tokens: int
    student_tokens: int
    aligned_positions: int
    coverage_ratio: float  # What fraction of student positions have alignment


def load_tokenizers(
    student_model: str = "Qwen/Qwen2-0.5B",
    teacher_model: str = "EleutherAI/pythia-1b",  # Ungated, GPT-NeoX tokenizer
):
    """Load tokenizers for student and teacher models."""
    AutoTokenizer, _, _ = get_transformers()

    print(f"Loading student tokenizer: {student_model}")
    student_tok = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)

    print(f"Loading teacher tokenizer: {teacher_model}")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)

    return student_tok, teacher_tok


def analyze_tokenizer(tokenizer, name: str, sample_text: str) -> TokenizerStats:
    """Analyze a tokenizer's behavior."""
    start = time.perf_counter()
    tokens = tokenizer.encode(sample_text)
    elapsed = (time.perf_counter() - start) * 1000

    # Decode individual tokens to see vocabulary
    sample_tokens = []
    for tid in tokens[:10]:
        try:
            decoded = tokenizer.decode([tid])
            sample_tokens.append(repr(decoded))
        except:
            sample_tokens.append(f"<{tid}>")

    return TokenizerStats(
        name=name,
        vocab_size=tokenizer.vocab_size,
        sample_tokens=sample_tokens,
        encode_time_ms=elapsed,
    )


def test_alignment(
    text: str,
    teacher_tok,
    student_tok,
) -> AlignmentStats:
    """Test cross-tokenizer alignment on sample text."""
    from phfe.distillation.cross_tokenizer import CrossTokenizerAligner

    aligner = CrossTokenizerAligner(teacher_tok, student_tok)

    teacher_tokens = teacher_tok.encode(text)
    student_tokens = student_tok.encode(text)

    alignment = aligner.align(text, teacher_tokens, student_tokens)

    # Count aligned positions
    aligned = sum(1 for s in range(len(student_tokens))
                  if s in alignment.student_to_teacher and alignment.student_to_teacher[s])

    return AlignmentStats(
        text=text[:50] + "..." if len(text) > 50 else text,
        teacher_tokens=len(teacher_tokens),
        student_tokens=len(student_tokens),
        aligned_positions=aligned,
        coverage_ratio=aligned / len(student_tokens) if student_tokens else 0,
    )


def test_vocabulary_mapping(teacher_tok, student_tok) -> dict:
    """Test vocabulary mapping between tokenizers."""
    from phfe.distillation.cross_tokenizer import VocabularyMapper

    mapper = VocabularyMapper(teacher_tok, student_tok)
    mapping = mapper.build_mapping()

    # Statistics
    mapped_count = sum(1 for v in mapping.values() if v)
    total_teacher = teacher_tok.vocab_size

    # Sample some mappings
    samples = []
    for tid, sids in list(mapping.items())[:20]:
        if sids:
            teacher_text = teacher_tok.decode([tid])
            student_texts = [student_tok.decode([sid]) for sid in sids[:3]]
            samples.append((repr(teacher_text), [repr(s) for s in student_texts]))

    return {
        "mapped_tokens": mapped_count,
        "total_teacher_tokens": total_teacher,
        "mapping_rate": mapped_count / total_teacher if total_teacher else 0,
        "samples": samples,
    }


def compute_mock_gkd_loss(
    text: str,
    teacher_tok,
    student_tok,
    teacher_model=None,
    student_model=None,
) -> dict:
    """
    Compute GKD loss (mock or real depending on model availability).

    If models are None, uses random logits for testing the pipeline.
    """
    from phfe.distillation.cross_tokenizer import (
        CrossTokenizerAligner,
        LogitAggregator,
        SparseLogits,
        compute_gkd_loss,
    )
    _, _, torch = get_transformers()

    # Tokenize
    teacher_tokens = teacher_tok.encode(text)
    student_tokens = student_tok.encode(text)

    # Align
    aligner = CrossTokenizerAligner(teacher_tok, student_tok)
    alignment = aligner.align(text, teacher_tokens, student_tokens)

    # Generate teacher logits (mock or real)
    if teacher_model is not None:
        # Real forward pass
        with torch.no_grad():
            inputs = teacher_tok(text, return_tensors="pt")
            # Move to same device as model
            device = next(teacher_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = teacher_model(**inputs)
            logits = outputs.logits[0].cpu()  # [seq_len, vocab_size]

            # Convert to sparse logits (top-p 99%)
            teacher_logits = []
            for pos in range(logits.shape[0]):
                probs = torch.softmax(logits[pos], dim=-1)
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)

                # Find top-p cutoff
                cumsum = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumsum >= 0.99).nonzero(as_tuple=True)[0]
                k = cutoff[0].item() + 1 if len(cutoff) > 0 else len(probs)
                k = min(k, 100)  # Cap at 100 tokens

                teacher_logits.append(SparseLogits(
                    token_ids=sorted_ids[:k].tolist(),
                    log_probs=torch.log(sorted_probs[:k]).tolist(),
                    total_prob=sorted_probs[:k].sum().item(),
                    position=pos,
                ))
    else:
        # Mock logits
        teacher_logits = []
        for pos in range(len(teacher_tokens)):
            # Random sparse distribution
            k = 50
            token_ids = np.random.choice(teacher_tok.vocab_size, k, replace=False).tolist()
            log_probs = np.log(np.random.dirichlet(np.ones(k))).tolist()
            teacher_logits.append(SparseLogits(
                token_ids=token_ids,
                log_probs=log_probs,
                total_prob=0.95,
                position=pos,
            ))

    # Aggregate to student positions
    aggregator = LogitAggregator(strategy="first")
    student_targets = aggregator.aggregate(
        teacher_logits, alignment, len(student_tokens)
    )

    # Generate student logits (mock or real)
    if student_model is not None:
        with torch.no_grad():
            inputs = student_tok(text, return_tensors="pt")
            device = next(student_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = student_model(**inputs)
            student_logits = outputs.logits[0].cpu()  # [seq_len, vocab_size]
    else:
        # Mock student logits
        student_logits = torch.randn(len(student_tokens), student_tok.vocab_size)

    # Compute loss
    loss = compute_gkd_loss(student_logits, student_targets, reduction="mean")

    return {
        "text_length": len(text),
        "teacher_tokens": len(teacher_tokens),
        "student_tokens": len(student_tokens),
        "aligned_targets": len(student_targets),
        "gkd_loss": loss,
        "using_real_models": teacher_model is not None,
    }


def run_tokenizer_tests(student_tok, teacher_tok):
    """Run tokenizer-only tests."""
    print("\n" + "="*60)
    print("TOKENIZER ANALYSIS")
    print("="*60)

    sample = "The quick brown fox jumps over the lazy dog. Mathematics is beautiful: 2 + 2 = 4."

    student_stats = analyze_tokenizer(student_tok, "Student (Qwen)", sample)
    teacher_stats = analyze_tokenizer(teacher_tok, "Teacher (Gemma)", sample)

    print(f"\nStudent: {student_stats.name}")
    print(f"  Vocab size: {student_stats.vocab_size:,}")
    print(f"  Encode time: {student_stats.encode_time_ms:.2f}ms")
    print(f"  Sample tokens: {student_stats.sample_tokens[:5]}")

    print(f"\nTeacher: {teacher_stats.name}")
    print(f"  Vocab size: {teacher_stats.vocab_size:,}")
    print(f"  Encode time: {teacher_stats.encode_time_ms:.2f}ms")
    print(f"  Sample tokens: {teacher_stats.sample_tokens[:5]}")

    # Test alignment on various texts
    print("\n" + "="*60)
    print("ALIGNMENT TESTS")
    print("="*60)

    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Alice has 5 apples. Bob gives her 3 more. How many apples does Alice have now?",
        "In a groundbreaking study, researchers discovered that quantum entanglement can be maintained at room temperature.",
    ]

    for text in test_texts:
        stats = test_alignment(text, teacher_tok, student_tok)
        print(f"\nText: {stats.text}")
        print(f"  Teacher tokens: {stats.teacher_tokens}")
        print(f"  Student tokens: {stats.student_tokens}")
        print(f"  Aligned: {stats.aligned_positions}/{stats.student_tokens} ({stats.coverage_ratio:.1%})")

    # Test vocabulary mapping
    print("\n" + "="*60)
    print("VOCABULARY MAPPING")
    print("="*60)

    mapping_stats = test_vocabulary_mapping(teacher_tok, student_tok)
    print(f"\nMapped tokens: {mapping_stats['mapped_tokens']:,} / {mapping_stats['total_teacher_tokens']:,}")
    print(f"Mapping rate: {mapping_stats['mapping_rate']:.1%}")
    print("\nSample mappings (teacher -> student):")
    for teacher, students in mapping_stats['samples'][:10]:
        print(f"  {teacher} -> {students}")


def run_full_tests(student_tok, teacher_tok, student_model, teacher_model):
    """Run full tests including model inference."""
    print("\n" + "="*60)
    print("GKD LOSS COMPUTATION")
    print("="*60)

    test_texts = [
        "Hello world!",
        "The answer to life, the universe, and everything is 42.",
        "def add(a, b): return a + b",
    ]

    for text in test_texts:
        print(f"\nText: {text[:50]}...")

        # Mock loss (no models)
        mock_result = compute_mock_gkd_loss(text, teacher_tok, student_tok)
        print(f"  Mock GKD loss: {mock_result['gkd_loss']:.4f}")
        print(f"  Aligned targets: {mock_result['aligned_targets']}/{mock_result['student_tokens']}")

        if student_model and teacher_model:
            real_result = compute_mock_gkd_loss(
                text, teacher_tok, student_tok, teacher_model, student_model
            )
            print(f"  Real GKD loss: {real_result['gkd_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-tokenizer GKD testbed")
    parser.add_argument(
        "--tokenizers-only",
        action="store_true",
        help="Only test tokenizers (no model download)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full test with model inference"
    )
    parser.add_argument(
        "--student-model",
        default="Qwen/Qwen2-0.5B",
        help="Student model name"
    )
    parser.add_argument(
        "--teacher-model",
        default="EleutherAI/pythia-1b",
        help="Teacher model name (default: pythia-1b, ungated)"
    )
    args = parser.parse_args()

    print("="*60)
    print("CROSS-TOKENIZER GKD TESTBED")
    print("="*60)
    print(f"Student: {args.student_model}")
    print(f"Teacher: {args.teacher_model}")

    # Load tokenizers
    student_tok, teacher_tok = load_tokenizers(
        args.student_model, args.teacher_model
    )

    # Run tokenizer tests
    run_tokenizer_tests(student_tok, teacher_tok)

    # Optionally load models and run full tests
    if args.full:
        print("\n" + "="*60)
        print("LOADING MODELS (this may take a while)")
        print("="*60)

        AutoTokenizer, AutoModelForCausalLM, torch = get_transformers()

        print(f"Loading student model: {args.student_model}")
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        run_full_tests(student_tok, teacher_tok, student_model, teacher_model)
    else:
        # Just run mock GKD loss computation
        print("\n" + "="*60)
        print("MOCK GKD LOSS (no models loaded)")
        print("="*60)

        test_text = "Alice has 5 apples. Bob gives her 3 more. How many apples does Alice have?"
        result = compute_mock_gkd_loss(test_text, teacher_tok, student_tok)

        print(f"\nText: {test_text}")
        print(f"Teacher tokens: {result['teacher_tokens']}")
        print(f"Student tokens: {result['student_tokens']}")
        print(f"Aligned targets: {result['aligned_targets']}")
        print(f"Mock GKD loss: {result['gkd_loss']:.4f}")

    print("\n" + "="*60)
    print("TESTBED COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
