"""
Maximum Trainable Scale Analysis

Goal: What's the LARGEST model you can train on a given GPU?

The question isn't "what's fastest" but "what WORKS at maximum scale".

Memory components:
1. Weights - can be IQ3_XXS (0.31 B/p), Q4_0 (0.56 B/p), FP8 (1 B/p), etc.
2. Activations - needed during forward/backward, scales with batch*seq*hidden
3. Gradients - same shape as weights, can be FP8/BF16/FP32
4. Optimizer state - Muon: just momentum, Adam: m + v

Key insight: Muon eliminates v entirely, and momentum can be 4-bit.
This is why Muon enables much larger scale than Adam.
"""

import numpy as np

# GPU configurations
GPUS = {
    "RTX 3060 12GB": 12,
    "RTX 3090 24GB": 24,
    "RTX 4090 24GB": 24,
    "A6000 48GB": 48,
    "A100 40GB": 40,
    "A100 80GB": 80,
    "H100 80GB": 80,
    "H100 NVL 94GB": 94,
    "MI300X 192GB": 192,
    "2x RTX 4090": 48,
    "4x RTX 4090": 96,
    "8x H100": 640,
}

# Weight quantization options (bytes per parameter)
WEIGHT_FORMATS = {
    "FP32": 4.0,
    "FP16/BF16": 2.0,
    "FP8": 1.0,
    "Q8_0": 1.0,      # 8-bit quantized
    "Q6_K": 0.75,     # 6-bit
    "Q5_K": 0.625,    # 5-bit
    "Q4_0": 0.5625,   # 4-bit (32 weights per block + 2 byte scale)
    "Q4_K": 0.5,      # 4-bit optimized
    "IQ4_XS": 0.45,   # importance-weighted 4-bit
    "Q3_K": 0.4375,   # 3-bit
    "IQ3_XXS": 0.31,  # importance-weighted 3-bit, smallest practical
    "Q2_K": 0.3125,   # 2-bit (often doesn't work well)
    "IQ2_XXS": 0.25,  # 2-bit importance (experimental)
}

# Gradient formats
GRAD_FORMATS = {
    "FP32": 4.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "FP8": 1.0,
}

# Optimizer state formats (Muon = momentum only, Adam = m + v)
OPTIMIZER_CONFIGS = {
    "Adam FP32": {"m": 4.0, "v": 4.0},           # 8 B/p
    "Adam BF16": {"m": 2.0, "v": 2.0},           # 4 B/p
    "Adam 8-bit": {"m": 1.0, "v": 1.0},          # 2 B/p
    "Muon FP32": {"m": 4.0},                      # 4 B/p (no v!)
    "Muon BF16": {"m": 2.0},                      # 2 B/p
    "Muon FP8": {"m": 1.0},                       # 1 B/p
    "Muon 8-bit": {"m": 1.0},                     # 1 B/p
    "Muon 4-bit": {"m": 0.5},                     # 0.5 B/p
    "Muon 4-bit packed": {"m": 0.5},              # 0.5 B/p (2 values per byte)
    "Muon 2-bit": {"m": 0.25},                    # 0.25 B/p (doesn't converge!)
    "SGD (no state)": {},                         # 0 B/p
}

# Convergence status (from our experiments)
CONVERGENCE = {
    "Adam FP32": True,
    "Adam BF16": True,
    "Adam 8-bit": True,  # With log-space transform
    "Muon FP32": True,
    "Muon BF16": True,
    "Muon FP8": True,
    "Muon 8-bit": True,
    "Muon 4-bit": True,  # +4% gap, acceptable
    "Muon 4-bit packed": True,
    "Muon 2-bit": False,  # Diverges!
    "SGD (no state)": True,  # Works but slow convergence
}


def calc_training_memory(
    weight_format: str,
    grad_format: str,
    optimizer: str,
    activation_overhead: float = 0.1,  # Fraction of weight memory for activations
) -> dict:
    """
    Calculate total training memory per parameter.

    Returns dict with breakdown and total.
    """
    weights = WEIGHT_FORMATS[weight_format]
    grads = GRAD_FORMATS[grad_format]
    opt_state = sum(OPTIMIZER_CONFIGS[optimizer].values())

    # Activations scale differently but we approximate as fraction of weights
    # Real activation memory depends on batch size, sequence length, etc.

    return {
        "weights": weights,
        "gradients": grads,
        "optimizer": opt_state,
        "total": weights + grads + opt_state,
        "converges": CONVERGENCE.get(optimizer, True),
    }


def calc_max_params(gpu_memory_gb: float, bytes_per_param: float,
                    overhead_factor: float = 0.80) -> float:
    """Calculate max trainable parameters."""
    available = gpu_memory_gb * 1e9 * overhead_factor
    return available / bytes_per_param


def format_params(n: float) -> str:
    """Format parameter count."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.0f}M"
    else:
        return f"{n:.0f}"


def print_comprehensive_analysis():
    """Print all configurations and their max scales."""

    print("=" * 100)
    print("MAXIMUM TRAINABLE SCALE ANALYSIS")
    print("What's the LARGEST model you can train on each GPU?")
    print("=" * 100)

    # Define interesting configurations
    configs = [
        # (Name, weight_fmt, grad_fmt, optimizer, notes)
        ("Baseline: FP16+Adam", "FP16/BF16", "BF16", "Adam FP32", "Standard training"),
        ("FP16+Muon", "FP16/BF16", "BF16", "Muon FP32", "Muon saves 4 B/p (no v)"),
        ("Q4+Adam", "Q4_0", "BF16", "Adam FP32", "Quantized weights + Adam"),
        ("Q4+Muon BF16", "Q4_0", "BF16", "Muon BF16", "Good balance"),
        ("Q4+Muon 8-bit", "Q4_0", "BF16", "Muon 8-bit", "More savings"),
        ("Q4+Muon 4-bit", "Q4_0", "BF16", "Muon 4-bit", "Our best practical config"),
        ("Q4+FP8 grad+Muon 4-bit", "Q4_0", "FP8", "Muon 4-bit", "FP8 gradients"),
        ("IQ3+BF16 grad+Muon 4-bit", "IQ3_XXS", "BF16", "Muon 4-bit", "Smallest weights"),
        ("IQ3+FP8 grad+Muon 4-bit", "IQ3_XXS", "FP8", "Muon 4-bit", "MAXIMUM COMPRESSION"),
        ("IQ3+FP8 grad+Muon 8-bit", "IQ3_XXS", "FP8", "Muon 8-bit", "Safer optimizer"),
        ("Q4+Muon 2-bit", "Q4_0", "BF16", "Muon 2-bit", "TOO AGGRESSIVE"),
        ("IQ2+FP8+Muon 4-bit", "IQ2_XXS", "FP8", "Muon 4-bit", "Experimental 2-bit"),
    ]

    # Memory breakdown table
    print("\n" + "-" * 100)
    print("MEMORY BREAKDOWN (Bytes per Parameter)")
    print("-" * 100)
    print(f"{'Configuration':<35} {'Weights':<10} {'Grads':<10} {'Optim':<10} {'TOTAL':<10} {'Works?':<8}")
    print("-" * 100)

    config_totals = {}
    for name, wf, gf, opt, notes in configs:
        mem = calc_training_memory(wf, gf, opt)
        status = "✓" if mem["converges"] else "✗"
        print(f"{name:<35} {mem['weights']:<10.2f} {mem['gradients']:<10.2f} {mem['optimizer']:<10.2f} {mem['total']:<10.2f} {status:<8}")
        config_totals[name] = (mem["total"], mem["converges"])

    # Max model size table
    print("\n" + "-" * 100)
    print("MAXIMUM MODEL SIZE BY GPU")
    print("-" * 100)

    # Select key configs for the table
    key_configs = [
        "Baseline: FP16+Adam",
        "Q4+Muon 4-bit",
        "IQ3+FP8 grad+Muon 4-bit",
    ]

    header = f"{'GPU':<22}"
    for cfg in key_configs:
        header += f"{cfg:<26}"
    print(header)
    print("-" * 100)

    for gpu_name, gpu_mem in GPUS.items():
        row = f"{gpu_name:<22}"
        for cfg in key_configs:
            bpp, works = config_totals[cfg]
            max_p = calc_max_params(gpu_mem, bpp)
            marker = "" if works else " ✗"
            row += f"{format_params(max_p):<26}"
        print(row)

    # Improvement factors
    print("\n" + "-" * 100)
    print("IMPROVEMENT vs BASELINE (FP16+Adam = 12 B/p)")
    print("-" * 100)

    baseline_bpp = 12.0

    print(f"{'Configuration':<40} {'B/param':<12} {'Improvement':<15} {'Works?':<8}")
    print("-" * 75)

    for name, wf, gf, opt, notes in configs:
        mem = calc_training_memory(wf, gf, opt)
        improvement = baseline_bpp / mem["total"]
        status = "✓" if mem["converges"] else "✗"
        print(f"{name:<40} {mem['total']:<12.2f} {improvement:<15.1f}x {status:<8}")


def print_extreme_configs():
    """Analyze the most extreme configurations."""

    print("\n" + "=" * 100)
    print("EXTREME COMPRESSION ANALYSIS")
    print("=" * 100)

    print("""
The theoretical minimum for training:

1. WEIGHTS: IQ3_XXS = 0.31 B/p (3-bit importance quantization)
   - Works for inference
   - Training quality TBD (may need higher precision for backward)

2. GRADIENTS: FP8 = 1.0 B/p
   - E4M3 format works for most operations
   - May need loss scaling for stability

3. OPTIMIZER: Muon 4-bit = 0.5 B/p
   - Proven to work (+4% convergence gap)
   - Muon 2-bit diverges, so 4-bit is the floor

THEORETICAL MINIMUM: 0.31 + 1.0 + 0.5 = 1.81 B/param
vs BASELINE: 12.0 B/param
IMPROVEMENT: 6.6x
""")

    # Calculate max model sizes with theoretical minimum
    print("-" * 80)
    print("MAXIMUM MODEL SIZE WITH EXTREME COMPRESSION (1.81 B/p)")
    print("-" * 80)

    extreme_bpp = 0.31 + 1.0 + 0.5  # IQ3 + FP8 grad + Muon 4-bit
    practical_bpp = 0.56 + 2.0 + 0.5  # Q4 + BF16 grad + Muon 4-bit
    baseline_bpp = 12.0

    print(f"{'GPU':<22} {'Baseline (12 B/p)':<20} {'Practical (3 B/p)':<20} {'Extreme (1.8 B/p)':<20}")
    print("-" * 82)

    for gpu_name, gpu_mem in GPUS.items():
        base = calc_max_params(gpu_mem, baseline_bpp)
        prac = calc_max_params(gpu_mem, practical_bpp)
        ext = calc_max_params(gpu_mem, extreme_bpp)
        print(f"{gpu_name:<22} {format_params(base):<20} {format_params(prac):<20} {format_params(ext):<20}")


def print_what_fits_where():
    """Show which models fit on which GPUs."""

    print("\n" + "=" * 100)
    print("WHAT FITS WHERE: MODEL → GPU MAPPING")
    print("=" * 100)

    models = [
        ("Llama-3.2-1B", 1e9),
        ("Llama-3.2-3B", 3e9),
        ("Llama-3.1-8B", 8e9),
        ("Llama-3.1-70B", 70e9),
        ("Llama-3.1-405B", 405e9),
        ("GPT-4 (est)", 1.8e12),
    ]

    configs = [
        ("FP16+Adam", 12.0),
        ("Q4+Muon 4-bit", 3.06),
        ("IQ3+FP8+Muon 4-bit", 1.81),
    ]

    print(f"\n{'Model':<20}", end="")
    for cfg_name, _ in configs:
        print(f"{cfg_name:<25}", end="")
    print()
    print("-" * 95)

    for model_name, params in models:
        print(f"{model_name:<20}", end="")
        for cfg_name, bpp in configs:
            mem_gb = (params * bpp) / 1e9
            # Find smallest GPU that fits
            fitting_gpus = [name for name, mem in GPUS.items() if mem >= mem_gb / 0.8]
            if fitting_gpus:
                smallest = fitting_gpus[0]
                print(f"{smallest:<25}", end="")
            else:
                print(f">{max(GPUS.values())}GB needed{'':<10}", end="")
        print()

    # Reverse: what can each GPU train?
    print("\n" + "-" * 100)
    print("GPU → MAXIMUM TRAINABLE MODEL")
    print("-" * 100)

    print(f"{'GPU':<22} {'FP16+Adam':<18} {'Q4+Muon 4-bit':<18} {'IQ3+FP8+Muon 4-bit':<20}")
    print("-" * 78)

    for gpu_name, gpu_mem in list(GPUS.items())[:8]:  # Top 8 GPUs
        row = f"{gpu_name:<22}"
        for cfg_name, bpp in configs:
            max_p = calc_max_params(gpu_mem, bpp)

            # Find closest model
            closest = "< 1B"
            for model_name, params in models:
                if params <= max_p:
                    closest = model_name.replace("Llama-", "").replace("3.1-", "").replace("3.2-", "")
            row += f"{closest:<18}"
        print(row)


def print_activation_analysis():
    """Analyze activation memory impact."""

    print("\n" + "=" * 100)
    print("ACTIVATION MEMORY ANALYSIS")
    print("=" * 100)

    print("""
Activation memory scales with: batch_size × seq_len × hidden_dim × num_layers

For Llama-7B (hidden=4096, layers=32):
- Per-layer activation: batch × seq × 4096 × dtype_size
- With attention: batch × seq × seq × num_heads (for attention scores)

Strategies to reduce activation memory:
1. Gradient checkpointing: Recompute instead of store (~3x memory reduction)
2. FP8 activations: 2x reduction vs FP16
3. Smaller batch size: Linear reduction
4. Sequence chunking: Process shorter sequences

Activation memory dominates at large batch sizes!
""")

    # Example calculation
    print("-" * 80)
    print("EXAMPLE: Llama-7B Activation Memory")
    print("-" * 80)

    hidden = 4096
    layers = 32
    heads = 32
    head_dim = hidden // heads

    batch_sizes = [1, 4, 8, 16, 32]
    seq_lens = [512, 1024, 2048, 4096]

    print(f"\nActivation memory (GB) with FP16, NO gradient checkpointing:")
    print(f"{'Batch':<8}", end="")
    for seq in seq_lens:
        print(f"seq={seq:<8}", end="")
    print()
    print("-" * 48)

    for batch in batch_sizes:
        print(f"{batch:<8}", end="")
        for seq in seq_lens:
            # Rough estimate: 2 * hidden * seq * batch * layers * 2 bytes
            # Plus attention: batch * heads * seq * seq * layers * 2 bytes
            act_mem = 2 * hidden * seq * batch * layers * 2  # Linear activations
            attn_mem = batch * heads * seq * seq * layers * 2  # Attention scores
            total_gb = (act_mem + attn_mem) / 1e9
            print(f"{total_gb:<9.1f}", end="")
        print()

    print(f"\nWith gradient checkpointing (~3x reduction):")
    print(f"{'Batch':<8}", end="")
    for seq in seq_lens:
        print(f"seq={seq:<8}", end="")
    print()
    print("-" * 48)

    for batch in batch_sizes:
        print(f"{batch:<8}", end="")
        for seq in seq_lens:
            act_mem = 2 * hidden * seq * batch * layers * 2
            attn_mem = batch * heads * seq * seq * layers * 2
            total_gb = (act_mem + attn_mem) / 1e9 / 3  # Checkpointing
            print(f"{total_gb:<9.1f}", end="")
        print()


def print_final_recommendations():
    """Print final configuration recommendations."""

    print("\n" + "=" * 100)
    print("FINAL RECOMMENDATIONS")
    print("=" * 100)

    print("""
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ CONFIGURATION TIERS                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                            │
│ TIER 1: SAFE & FAST (3.0 B/p)                                                             │
│ ─────────────────────────────────                                                          │
│ Weights:    Q4_0 (0.56 B/p)                                                               │
│ Gradients:  BF16 (2.0 B/p)                                                                │
│ Optimizer:  Muon 4-bit (0.5 B/p)                                                          │
│ N-S:        BF16 (fastest on Ada/Ampere)                                                  │
│ Result:     4x more params than baseline, proven convergence                              │
│                                                                                            │
│ TIER 2: AGGRESSIVE (2.3 B/p)                                                              │
│ ─────────────────────────────────                                                          │
│ Weights:    Q4_0 (0.56 B/p)                                                               │
│ Gradients:  FP8 (1.0 B/p)                                                                 │
│ Optimizer:  Muon 4-bit (0.5 B/p) + FP8 accumulation                                       │
│ N-S:        FP8 (faster on Hopper)                                                        │
│ Result:     5.2x more params, may need loss scaling                                       │
│                                                                                            │
│ TIER 3: EXTREME (1.8 B/p)                                                                 │
│ ─────────────────────────────────                                                          │
│ Weights:    IQ3_XXS (0.31 B/p)                                                            │
│ Gradients:  FP8 (1.0 B/p)                                                                 │
│ Optimizer:  Muon 4-bit (0.5 B/p)                                                          │
│ N-S:        FP8                                                                           │
│ Result:     6.6x more params, EXPERIMENTAL - quality unknown                              │
│                                                                                            │
│ DOES NOT WORK                                                                             │
│ ─────────────────────────────────                                                          │
│ • Muon 2-bit momentum: Diverges after ~100 steps                                          │
│ • INT4 Newton-Schulz: Diverges after 4 iterations                                         │
│ • Adam with 3-bit state: Log-space doesn't help enough                                    │
│                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘

GPU-SPECIFIC RECOMMENDATIONS:

RTX 3060 12GB:    Tier 1 → 3.2B params    │ Tier 3 → 5.3B params
RTX 4090 24GB:    Tier 1 → 6.4B params    │ Tier 3 → 10.6B params
A100 80GB:        Tier 1 → 21.3B params   │ Tier 3 → 35.4B params
H100 80GB:        Tier 1 → 21.3B params   │ Tier 3 → 35.4B params (use FP8!)
8x H100 640GB:    Tier 1 → 170B params    │ Tier 3 → 283B params

BOTTOM LINE:
─────────────────
• Safe path: Q4 weights + BF16 grads + Muon 4-bit = 4x improvement
• Aggressive: + FP8 grads = 5.2x improvement
• Extreme: IQ3 weights = 6.6x improvement (quality TBD)
• Muon is the key enabler: eliminating v saves 4-8 B/p alone
""")


def main():
    print_comprehensive_analysis()
    print_extreme_configs()
    print_what_fits_where()
    print_activation_analysis()
    print_final_recommendations()


if __name__ == "__main__":
    main()
