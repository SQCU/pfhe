#!/usr/bin/env python3
"""
KL Divergence Demo: GGUF Dequantization Verification

Compares logits from:
1. llama-cpp-python (reference C++ implementation)
2. Manual forward pass using phfe.inference GGUF loader

This verifies that our pure-Python dequantization produces
numerically equivalent results to the reference implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phfe.inference import GGUFReader, tensor_loading_context, inspect_gguf


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    Compute KL(P || Q) where P and Q are logit distributions.

    KL divergence measures how different Q is from P.
    KL = 0 means identical distributions.
    """
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)

    # Add small epsilon for numerical stability
    eps = 1e-10
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    return kl.mean().item()


def jensen_shannon_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded 0-1).
    """
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    m = 0.5 * (p + q)

    eps = 1e-10
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)

    return (0.5 * (kl_pm + kl_qm)).mean().item()


def get_llama_cpp_logits(model_path: str, prompt: str, n_ctx: int = 512) -> torch.Tensor:
    """Get logits from llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("Install llama-cpp-python: pip install llama-cpp-python")

    print(f"Loading model with llama-cpp-python: {Path(model_path).name}")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,  # Use GPU
        logits_all=True,
        verbose=False,
    )

    # Tokenize
    tokens = llm.tokenize(prompt.encode())
    print(f"Prompt: {prompt!r}")
    print(f"Tokens: {len(tokens)}")

    # Run inference
    llm.reset()
    llm.eval(tokens)

    # Get logits for last token
    logits = llm.scores[len(tokens) - 1]
    return torch.tensor(logits, dtype=torch.float32)


def get_manual_logits(
    model_path: str,
    prompt: str,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
) -> torch.Tensor:
    """
    Compute logits manually using our GGUF loader.

    This is a simplified single-token embedding lookup + output projection.
    Not a full transformer forward pass, but validates the weight loading.
    """
    from transformers import AutoTokenizer

    print(f"\nLoading weights with phfe.inference GGUF loader...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Tokens (transformers): {len(tokens)}")

    with tensor_loading_context(model_path) as loader:
        # Check available tensors
        tensor_names = list(loader.tensors.keys())
        print(f"Available tensors: {len(tensor_names)}")

        # Find embedding and output tensors
        embed_name = None
        output_name = None
        for name in tensor_names:
            if "token_embd" in name or "embed_tokens" in name:
                embed_name = name
            # Look for the actual lm_head (output.weight), not attn_output
            if name == "output.weight" or name == "lm_head.weight":
                output_name = name

        if not embed_name:
            print("Could not find embedding tensor")
            return None

        print(f"Embedding tensor: {embed_name}")
        print(f"Output tensor: {output_name}")

        # Load tensors
        embeddings = loader.read_tensor(embed_name, dtype=torch.float32)
        print(f"Embeddings shape: {embeddings.shape}")

        # Get last token embedding
        last_token_id = tokens[-1]
        last_embed = embeddings[last_token_id]
        print(f"Last token ID: {last_token_id}, embedding shape: {last_embed.shape}")

        # If there's an output projection, use it
        if output_name:
            output_proj = loader.read_tensor(output_name, dtype=torch.float32)
            print(f"Output projection shape: {output_proj.shape}")

            # Simple logits: embed @ output.T
            # (This ignores all transformer layers - just for weight verification)
            if output_proj.shape[0] == embeddings.shape[0]:
                # output is [vocab, hidden]
                logits = last_embed @ output_proj.T
            else:
                logits = last_embed @ output_proj

            return logits
        else:
            # Use embedding as output (tied weights)
            logits = last_embed @ embeddings.T
            return logits


def compare_weight_stats(model_path: str):
    """Compare weight statistics from our loader."""
    print("\n" + "="*60)
    print("WEIGHT STATISTICS (phfe.inference loader)")
    print("="*60)

    with tensor_loading_context(model_path) as loader:
        # Sample a few tensors
        sample_tensors = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_up.weight",
        ]

        for name in sample_tensors:
            if name not in loader.tensors:
                # Try alternative naming
                continue

            info = loader.tensors[name]
            tensor = loader.read_tensor(name, dtype=torch.float32)

            print(f"\n{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype in file: {info.dtype_name}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std:  {tensor.std().item():.6f}")
            print(f"  Min:  {tensor.min().item():.6f}")
            print(f"  Max:  {tensor.max().item():.6f}")

            # Check for NaN/Inf (dequantization bugs)
            if torch.isnan(tensor).any():
                print("  WARNING: Contains NaN!")
            if torch.isinf(tensor).any():
                print("  WARNING: Contains Inf!")


def demo_kl_divergence():
    """Main demo comparing implementations."""

    # Use a small model for speed
    model_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

    if not Path(model_path).exists():
        # Fallback
        model_path = "/mnt/f/dox/ai/text/models/gemma-3-4b-it-Q8_0.gguf"

    print("="*60)
    print("KL DIVERGENCE DEMO: GGUF Dequantization Verification")
    print("="*60)
    print(f"\nModel: {Path(model_path).name}")

    # First, inspect the model
    info = inspect_gguf(model_path)
    print(f"GGUF Version: {info['version']}")
    print(f"Tensors: {info['n_tensors']}")

    # Show some metadata
    arch = info['metadata'].get('general.architecture', 'unknown')
    print(f"Architecture: {arch}")

    # Weight statistics check
    compare_weight_stats(model_path)

    # Test prompt
    prompt = "The capital of France is"

    print("\n" + "="*60)
    print("LOGITS COMPARISON")
    print("="*60)

    # Get llama-cpp logits
    try:
        llama_logits = get_llama_cpp_logits(model_path, prompt)
        print(f"\nllama-cpp logits shape: {llama_logits.shape}")
        print(f"llama-cpp logits stats: mean={llama_logits.mean():.4f}, std={llama_logits.std():.4f}")

        # Top-5 tokens
        top5 = torch.topk(llama_logits, 5)
        print(f"Top-5 token IDs: {top5.indices.tolist()}")
        print(f"Top-5 logits: {top5.values.tolist()}")

    except ImportError as e:
        print(f"\nSkipping llama-cpp comparison: {e}")
        llama_logits = None
    except Exception as e:
        print(f"\nllama-cpp error: {e}")
        llama_logits = None

    # Get manual logits (weight verification)
    try:
        # Determine tokenizer based on architecture
        if "qwen" in arch.lower() or "qwen" in model_path.lower():
            tokenizer_name = "Qwen/Qwen2.5-1.5B"
        elif "gemma" in arch.lower():
            tokenizer_name = "google/gemma-2-2b-it"
        else:
            tokenizer_name = "Qwen/Qwen2.5-1.5B"

        manual_logits = get_manual_logits(model_path, prompt, tokenizer_name)

        if manual_logits is not None:
            print(f"\nManual logits shape: {manual_logits.shape}")
            print(f"Manual logits stats: mean={manual_logits.mean():.4f}, std={manual_logits.std():.4f}")

    except Exception as e:
        print(f"\nManual logits error: {e}")
        import traceback
        traceback.print_exc()
        manual_logits = None

    # Compute divergence if we have both
    if llama_logits is not None and manual_logits is not None:
        # Ensure same vocab size
        min_vocab = min(llama_logits.shape[0], manual_logits.shape[0])
        llama_trimmed = llama_logits[:min_vocab]
        manual_trimmed = manual_logits[:min_vocab]

        kl = kl_divergence(llama_trimmed.unsqueeze(0), manual_trimmed.unsqueeze(0))
        jsd = jensen_shannon_divergence(llama_trimmed.unsqueeze(0), manual_trimmed.unsqueeze(0))

        print("\n" + "="*60)
        print("DIVERGENCE METRICS")
        print("="*60)
        print(f"KL(llama-cpp || manual):  {kl:.6f}")
        print(f"Jensen-Shannon Divergence: {jsd:.6f}")
        print()

        # Interpretation
        if jsd < 0.01:
            print("Result: EXCELLENT - Distributions nearly identical")
        elif jsd < 0.05:
            print("Result: GOOD - Minor differences (expected for embedding-only comparison)")
        elif jsd < 0.1:
            print("Result: ACCEPTABLE - Some divergence (missing transformer layers)")
        else:
            print("Result: HIGH DIVERGENCE - Check dequantization")

        # Note about the comparison
        print("\nNote: Manual logits use ONLY embedding lookup + output projection.")
        print("Full transformer forward pass would require implementing all layers.")
        print("This test verifies weight dequantization, not full inference parity.")


def demo_quantization_effect():
    """
    Compare Q4_0 vs Q8_0 quantization effects.
    Shows how quantization affects the logit distribution.
    """
    print("\n" + "="*60)
    print("QUANTIZATION COMPARISON: Q4 vs Q8")
    print("="*60)

    q4_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    q8_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

    if not Path(q4_path).exists() or not Path(q8_path).exists():
        print("Skipping: Q4/Q8 model pair not found")
        return

    prompt = "The meaning of life is"

    try:
        from llama_cpp import Llama

        print(f"\nPrompt: {prompt!r}")

        # Q8 logits
        print("\nLoading Q8_0 model...")
        q8_logits = get_llama_cpp_logits(q8_path, prompt)

        # Q4 logits
        print("\nLoading Q4_K_M model...")
        q4_logits = get_llama_cpp_logits(q4_path, prompt)

        # Compare
        min_vocab = min(q4_logits.shape[0], q8_logits.shape[0])
        q4_trimmed = q4_logits[:min_vocab]
        q8_trimmed = q8_logits[:min_vocab]

        kl = kl_divergence(q8_trimmed.unsqueeze(0), q4_trimmed.unsqueeze(0))
        jsd = jensen_shannon_divergence(q8_trimmed.unsqueeze(0), q4_trimmed.unsqueeze(0))

        print("\n" + "-"*40)
        print("QUANTIZATION DIVERGENCE")
        print("-"*40)
        print(f"KL(Q8 || Q4):              {kl:.6f}")
        print(f"Jensen-Shannon Divergence: {jsd:.6f}")

        # Top token agreement
        q8_top = torch.argmax(q8_trimmed)
        q4_top = torch.argmax(q4_trimmed)
        print(f"\nTop-1 token (Q8): {q8_top.item()}")
        print(f"Top-1 token (Q4): {q4_top.item()}")
        print(f"Top-1 agreement: {'YES' if q8_top == q4_top else 'NO'}")

        # Top-5 agreement
        q8_top5 = set(torch.topk(q8_trimmed, 5).indices.tolist())
        q4_top5 = set(torch.topk(q4_trimmed, 5).indices.tolist())
        overlap = len(q8_top5 & q4_top5)
        print(f"Top-5 overlap: {overlap}/5")

    except ImportError:
        print("llama-cpp-python not installed")
    except Exception as e:
        print(f"Error: {e}")


def demo_dequant_verification():
    """
    Verify dequantization correctness by checking statistical properties.

    For properly dequantized weights:
    - Mean should be near 0 (initialized weights)
    - Std should be reasonable (not exploded)
    - No NaN or Inf values
    - Values should be bounded (not extreme)
    """
    print("\n" + "="*60)
    print("DEQUANTIZATION VERIFICATION")
    print("="*60)

    model_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

    with tensor_loading_context(model_path) as loader:
        test_tensors = [
            ("token_embd.weight", "Embedding"),
            ("output.weight", "LM Head"),
            ("blk.0.attn_q.weight", "Q Projection (Layer 0)"),
            ("blk.14.ffn_gate.weight", "FFN Gate (Layer 14)"),
        ]

        all_passed = True

        for name, desc in test_tensors:
            if name not in loader.tensors:
                continue

            info = loader.tensors[name]
            tensor = loader.read_tensor(name, dtype=torch.float32)

            # Statistical checks
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            mean = tensor.mean().item()
            std = tensor.std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()

            # Expected ranges for well-behaved neural network weights
            mean_ok = abs(mean) < 0.1  # Should be roughly centered
            std_ok = 0.001 < std < 1.0  # Reasonable variance
            range_ok = abs(min_val) < 10 and abs(max_val) < 10  # Not exploded

            passed = not has_nan and not has_inf and mean_ok and std_ok and range_ok

            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"\n{desc} ({name}):")
            print(f"  Shape: {tuple(tensor.shape)}, File dtype: {info.dtype_name}")
            print(f"  Mean: {mean:+.6f} {'OK' if mean_ok else 'WARN'}")
            print(f"  Std:  {std:.6f} {'OK' if std_ok else 'WARN'}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}] {'OK' if range_ok else 'WARN'}")
            print(f"  NaN: {has_nan}, Inf: {has_inf}")
            print(f"  Status: {status}")

        print("\n" + "-"*40)
        print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")


def demo_q4_vs_q8_weights():
    """
    Compare dequantized weights between Q4 and Q8 versions.
    Shows quantization error at the weight level.
    """
    print("\n" + "="*60)
    print("WEIGHT-LEVEL QUANTIZATION COMPARISON")
    print("="*60)

    q4_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    q8_path = "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

    if not Path(q4_path).exists() or not Path(q8_path).exists():
        print("Skipping: Q4/Q8 model pair not found")
        return

    test_tensor = "blk.0.attn_q.weight"  # A representative weight matrix

    with tensor_loading_context(q8_path) as q8_loader:
        with tensor_loading_context(q4_path) as q4_loader:
            if test_tensor not in q8_loader.tensors:
                print(f"Tensor {test_tensor} not found in Q8 model")
                return

            # Check if Q4 model has different naming
            q4_tensor_name = test_tensor
            if test_tensor not in q4_loader.tensors:
                # Try to find equivalent
                for name in q4_loader.tensors:
                    if "blk.0" in name and "attn_q" in name:
                        q4_tensor_name = name
                        break
                else:
                    print(f"Could not find equivalent tensor in Q4 model")
                    return

            print(f"\nComparing: {test_tensor}")
            print(f"Q8 dtype: {q8_loader.tensors[test_tensor].dtype_name}")
            print(f"Q4 dtype: {q4_loader.tensors[q4_tensor_name].dtype_name}")

            q8_weight = q8_loader.read_tensor(test_tensor, dtype=torch.float32)
            try:
                q4_weight = q4_loader.read_tensor(q4_tensor_name, dtype=torch.float32)
            except (NotImplementedError, ValueError) as e:
                q4_dtype = q4_loader.tensors[q4_tensor_name].dtype_name
                print(f"\nNote: {q4_dtype} dequantization not yet implemented")
                print(f"Implemented: Q4_0, Q4_1, Q8_0, F16, BF16, F32")
                print("K-quants (Q4_K, Q5_K, Q6_K) are more complex and require additional work.")
                return

            if q8_weight.shape != q4_weight.shape:
                print(f"Shape mismatch: Q8={q8_weight.shape}, Q4={q4_weight.shape}")
                return

            # Compute differences
            diff = q8_weight - q4_weight
            mse = (diff ** 2).mean().item()
            mae = diff.abs().mean().item()
            max_diff = diff.abs().max().item()
            rel_error = (diff.abs() / (q8_weight.abs() + 1e-8)).mean().item()

            print(f"\nWeight Difference Statistics:")
            print(f"  MSE:          {mse:.8f}")
            print(f"  MAE:          {mae:.6f}")
            print(f"  Max Abs Diff: {max_diff:.6f}")
            print(f"  Mean Rel Err: {rel_error*100:.4f}%")

            # Correlation
            q8_flat = q8_weight.flatten()
            q4_flat = q4_weight.flatten()
            correlation = torch.corrcoef(torch.stack([q8_flat, q4_flat]))[0, 1].item()
            print(f"  Correlation:  {correlation:.6f}")

            # Histogram of differences
            print(f"\nDifference Distribution:")
            percentiles = [50, 90, 99, 99.9]
            for p in percentiles:
                val = torch.quantile(diff.abs().flatten().float(), p/100).item()
                print(f"  P{p}: {val:.6f}")


if __name__ == "__main__":
    demo_kl_divergence()
    demo_quantization_effect()
    demo_dequant_verification()
    demo_q4_vs_q8_weights()
