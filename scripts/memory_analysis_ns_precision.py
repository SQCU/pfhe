"""
Memory Analysis: Newton-Schulz Precision vs Maximum Model Size

Calculate memory overhead and maximum trainable model size for each precision.

Components:
1. Weights (Q4_0 or higher precision)
2. Gradients (same precision as activations)
3. Optimizer state (Muon momentum - 4-bit or higher)
4. N-S intermediate buffers (A = X@X.T, temporary)

Key insight: N-S precision affects intermediate buffer size, not stored state.
"""

import matplotlib.pyplot as plt
import numpy as np

# GPU memory configurations (GB)
GPUS = {
    "RTX 3060 12GB": 12,
    "RTX 3090 24GB": 24,
    "RTX 4090 24GB": 24,
    "A100 40GB": 40,
    "A100 80GB": 80,
    "H100 80GB": 80,
    "H100 NVL 94GB": 94,
    "8x H100 (640GB)": 640,
}

# Bytes per parameter for different components
def calc_bytes_per_param(
    weight_bits: int = 4,      # Q4_0 = 4 bits
    grad_bits: int = 16,       # BF16 gradients
    momentum_bits: int = 4,    # 4-bit Muon momentum
    ns_precision: str = "bf16", # Newton-Schulz precision
):
    """Calculate total bytes per parameter for training."""

    # Weight storage
    weight_bytes = weight_bits / 8

    # Gradient storage (needed during backward)
    grad_bytes = grad_bits / 8

    # Optimizer state (Muon momentum only, no v!)
    momentum_bytes = momentum_bits / 8

    # N-S intermediate buffers are temporary, not per-param
    # But we need to account for peak memory during N-S

    return {
        "weights": weight_bytes,
        "gradients": grad_bytes,
        "momentum": momentum_bytes,
        "total_stored": weight_bytes + momentum_bytes,
        "total_training": weight_bytes + grad_bytes + momentum_bytes,
    }


def calc_ns_buffer_size(hidden_dim: int, ns_precision: str) -> float:
    """
    Calculate N-S intermediate buffer size in bytes.

    N-S needs: A = X @ X.T where X is (hidden_dim, hidden_dim)
    Plus A@A and B@X intermediates.

    Peak memory: ~3 * hidden_dim^2 * bytes_per_elem
    """
    bytes_per_elem = {
        "fp32": 4,
        "bf16": 2,
        "fp16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
    }

    bpe = bytes_per_elem.get(ns_precision, 2)

    # A = X@X.T (hidden x hidden)
    # A@A (hidden x hidden)
    # B@X (hidden x hidden)
    # Total peak: ~3 matrices
    return 3 * hidden_dim * hidden_dim * bpe


def calc_max_params(
    gpu_memory_gb: float,
    weight_bits: int = 4,
    grad_bits: int = 16,
    momentum_bits: int = 4,
    ns_precision: str = "bf16",
    hidden_dim: int = 4096,
    overhead_factor: float = 0.85,  # Reserve 15% for activations, etc.
) -> float:
    """Calculate maximum trainable parameters given GPU memory."""

    available_bytes = gpu_memory_gb * 1e9 * overhead_factor

    # Per-param memory
    mem = calc_bytes_per_param(weight_bits, grad_bits, momentum_bits, ns_precision)
    bytes_per_param = mem["total_training"]

    # N-S buffer (fixed cost based on hidden dim)
    ns_buffer = calc_ns_buffer_size(hidden_dim, ns_precision)

    # Max params = (available - ns_buffer) / bytes_per_param
    max_params = (available_bytes - ns_buffer) / bytes_per_param

    return max_params


def format_params(n: float) -> str:
    """Format parameter count nicely."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    else:
        return f"{n:.0f}"


def print_memory_table():
    """Print comprehensive memory analysis table."""

    print("=" * 90)
    print("MEMORY ANALYSIS: NEWTON-SCHULZ PRECISION VS MODEL SIZE")
    print("=" * 90)

    # Training configurations
    configs = [
        # (name, weight_bits, grad_bits, momentum_bits, ns_precision)
        ("FP16 + Adam (baseline)", 16, 16, 32, "fp32"),  # 2 + 2 + 8 = 12 B/p
        ("FP16 + Muon FP32", 16, 16, 32, "fp32"),        # 2 + 2 + 4 = 8 B/p
        ("BF16 + Muon BF16", 16, 16, 32, "bf16"),        # 2 + 2 + 4 = 8 B/p
        ("Q4 + Muon FP32 mom", 4, 16, 32, "fp32"),       # 0.5 + 2 + 4 = 6.5 B/p
        ("Q4 + Muon 8-bit mom", 4, 16, 8, "bf16"),       # 0.5 + 2 + 1 = 3.5 B/p
        ("Q4 + Muon 4-bit mom (BF16 N-S)", 4, 16, 4, "bf16"),  # 0.5 + 2 + 0.5 = 3 B/p
        ("Q4 + Muon 4-bit mom (FP8 N-S)", 4, 16, 4, "fp8"),    # 0.5 + 2 + 0.5 = 3 B/p
        ("Q4 + Muon 4-bit mom (INT8 N-S)", 4, 16, 4, "int8"),  # 0.5 + 2 + 0.5 = 3 B/p
    ]

    # Memory per parameter
    print("\n" + "-" * 90)
    print("BYTES PER PARAMETER")
    print("-" * 90)
    print(f"{'Configuration':<35} {'Weights':<10} {'Grads':<10} {'Momentum':<10} {'Total':<10}")
    print("-" * 90)

    for name, wb, gb, mb, ns in configs:
        mem = calc_bytes_per_param(wb, gb, mb, ns)
        # For Adam baseline, momentum is actually m+v = 8 bytes
        if "Adam" in name:
            total = mem["weights"] + mem["gradients"] + 8  # m + v
            print(f"{name:<35} {mem['weights']:<10.2f} {mem['gradients']:<10.2f} {'8.00':<10} {total:<10.2f}")
        else:
            total = mem["total_training"]
            print(f"{name:<35} {mem['weights']:<10.2f} {mem['gradients']:<10.2f} {mem['momentum']:<10.2f} {total:<10.2f}")

    # N-S buffer sizes
    print("\n" + "-" * 90)
    print("NEWTON-SCHULZ BUFFER SIZE (for hidden_dim=4096)")
    print("-" * 90)

    ns_precisions = ["fp32", "bf16", "fp16", "fp8", "int8", "int4"]
    print(f"{'N-S Precision':<15} {'Buffer Size':<15} {'vs FP32':<15}")
    print("-" * 45)

    fp32_size = calc_ns_buffer_size(4096, "fp32")
    for ns in ns_precisions:
        size = calc_ns_buffer_size(4096, ns)
        ratio = fp32_size / size
        print(f"{ns.upper():<15} {size/1e6:>10.1f} MB    {ratio:>5.1f}x smaller")

    # Max model sizes per GPU
    print("\n" + "-" * 90)
    print("MAXIMUM TRAINABLE MODEL SIZE BY GPU")
    print("-" * 90)

    # Simplified configs for the table
    simple_configs = [
        ("FP16+Adam", 16, 16, 64, "fp32"),      # Standard baseline (m+v = 8 bytes represented as 64 bits)
        ("Q4+Muon BF16", 4, 16, 4, "bf16"),     # Our best config
        ("Q4+Muon FP8", 4, 16, 4, "fp8"),       # With FP8 N-S
        ("Q4+Muon INT8", 4, 16, 4, "int8"),     # With INT8 N-S
    ]

    header = f"{'GPU':<20}"
    for name, *_ in simple_configs:
        header += f"{name:<15}"
    print(header)
    print("-" * 80)

    for gpu_name, gpu_mem in GPUS.items():
        row = f"{gpu_name:<20}"
        for name, wb, gb, mb, ns in simple_configs:
            # Adjust for Adam's actual memory (m + v = 8 bytes)
            if "Adam" in name:
                # FP16 weights (2) + FP16 grads (2) + FP32 m,v (8) = 12 B/p
                bytes_per_param = 12
                available = gpu_mem * 1e9 * 0.85
                ns_buf = calc_ns_buffer_size(4096, "fp32")
                max_p = (available - ns_buf) / bytes_per_param
            else:
                max_p = calc_max_params(gpu_mem, wb, gb, mb, ns, 4096)
            row += f"{format_params(max_p):<15}"
        print(row)

    # Memory efficiency comparison
    print("\n" + "-" * 90)
    print("MEMORY EFFICIENCY COMPARISON")
    print("-" * 90)

    baseline_bpp = 12  # FP16 + Adam

    configs_efficiency = [
        ("FP16 + Adam (baseline)", 12.0, "fp32", "1.0x"),
        ("FP16 + Muon FP32", 8.0, "fp32", "1.5x"),
        ("Q4 + Muon 8-bit", 3.5, "bf16", "3.4x"),
        ("Q4 + Muon 4-bit (BF16 N-S)", 3.0, "bf16", "4.0x"),
        ("Q4 + Muon 4-bit (FP8 N-S)", 3.0, "fp8", "4.0x"),
        ("Q4 + Muon 4-bit (INT8 N-S)", 3.0, "int8", "4.0x"),
    ]

    print(f"{'Configuration':<35} {'B/param':<12} {'N-S Prec':<12} {'Mem Savings':<12} {'N-S Speedup':<12}")
    print("-" * 83)

    ns_speedups = {"fp32": "1.0x", "bf16": "3.2x", "fp16": "3.4x", "fp8": "2.9x", "int8": "1.2x"}

    for name, bpp, ns, savings in configs_efficiency:
        speedup = ns_speedups.get(ns, "?")
        print(f"{name:<35} {bpp:<12.1f} {ns.upper():<12} {savings:<12} {speedup:<12}")


def create_visualization():
    """Create visualization of memory vs model size tradeoffs."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bytes per parameter breakdown
    ax1 = axes[0, 0]
    configs = [
        ("FP16+Adam", 2, 2, 8),       # weights, grads, optimizer
        ("FP16+Muon", 2, 2, 4),
        ("Q4+Muon 8b", 0.5, 2, 1),
        ("Q4+Muon 4b", 0.5, 2, 0.5),
    ]

    x = np.arange(len(configs))
    width = 0.25

    weights = [c[1] for c in configs]
    grads = [c[2] for c in configs]
    optim = [c[3] for c in configs]

    ax1.bar(x - width, weights, width, label='Weights', color='#2ecc71')
    ax1.bar(x, grads, width, label='Gradients', color='#3498db')
    ax1.bar(x + width, optim, width, label='Optimizer', color='#e74c3c')

    ax1.set_ylabel('Bytes per Parameter')
    ax1.set_title('Memory Breakdown by Component')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in configs])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Max model size by GPU
    ax2 = axes[0, 1]

    gpus_subset = ["RTX 4090 24GB", "A100 80GB", "H100 80GB", "8x H100 (640GB)"]
    gpu_mems = [GPUS[g] for g in gpus_subset]

    configs_plot = [
        ("FP16+Adam", 12),
        ("Q4+Muon BF16", 3),
        ("Q4+Muon FP8", 3),
    ]

    x = np.arange(len(gpus_subset))
    width = 0.25

    for i, (name, bpp) in enumerate(configs_plot):
        max_params = [(g * 1e9 * 0.85) / bpp / 1e9 for g in gpu_mems]
        ax2.bar(x + i*width - width/2, max_params, width, label=name)

    ax2.set_ylabel('Max Parameters (Billions)')
    ax2.set_title('Maximum Trainable Model Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gpus_subset, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: N-S buffer size by precision
    ax3 = axes[1, 0]

    ns_precs = ["FP32", "BF16", "FP16", "FP8", "INT8", "INT4"]
    buffer_sizes = [calc_ns_buffer_size(4096, p.lower()) / 1e6 for p in ns_precs]
    colors = ['#e74c3c', '#f39c12', '#f39c12', '#27ae60', '#27ae60', '#27ae60']

    bars = ax3.bar(ns_precs, buffer_sizes, color=colors)
    ax3.set_ylabel('Buffer Size (MB)')
    ax3.set_title('N-S Intermediate Buffer Size (hidden=4096)')
    ax3.grid(axis='y', alpha=0.3)

    # Add text annotations
    for bar, size in zip(bars, buffer_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{size:.0f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Tradeoff visualization
    ax4 = axes[1, 1]

    # Data: (name, bytes_per_param, ns_speedup, works)
    methods = [
        ("FP16+Adam", 12, 1.0, True),
        ("Q4+Muon FP32", 3, 1.0, True),
        ("Q4+Muon BF16", 3, 3.2, True),
        ("Q4+Muon FP8", 3, 2.9, True),
        ("Q4+Muon INT8", 3, 1.2, True),
        ("Q4+Muon INT4", 3, 1.0, False),
    ]

    for name, bpp, speedup, works in methods:
        marker = 'o' if works else 'x'
        color = '#27ae60' if works else '#e74c3c'
        size = 150 if works else 100
        ax4.scatter(bpp, speedup, s=size, marker=marker, c=color, label=name, alpha=0.8)
        ax4.annotate(name, (bpp, speedup), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    ax4.set_xlabel('Bytes per Parameter (lower = more params)')
    ax4.set_ylabel('N-S Speedup vs FP32 (higher = faster)')
    ax4.set_title('Memory vs Speed Tradeoff (✓ = works, ✗ = diverges)')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 14)
    ax4.set_ylim(0, 4)

    plt.tight_layout()
    plt.savefig('notes/ns_precision_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to notes/ns_precision_analysis.png")
    plt.close()


def print_practical_recommendations():
    """Print practical recommendations based on GPU."""

    print("\n" + "=" * 90)
    print("PRACTICAL RECOMMENDATIONS BY GPU")
    print("=" * 90)

    recommendations = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ GPU                │ Recommended Config           │ Max Model │ Notes              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ RTX 3060 12GB      │ Q4 + Muon 4-bit (BF16 N-S)  │ ~3.4B     │ BF16 fastest       │
│ RTX 4090 24GB      │ Q4 + Muon 4-bit (BF16 N-S)  │ ~6.8B     │ FP8 slower on Ada  │
│ A100 40GB          │ Q4 + Muon 4-bit (BF16 N-S)  │ ~11.3B    │ BF16 tensor cores  │
│ A100 80GB          │ Q4 + Muon 4-bit (BF16 N-S)  │ ~22.7B    │ Try FP8 if Hopper  │
│ H100 80GB          │ Q4 + Muon 4-bit (FP8 N-S)   │ ~22.7B    │ FP8 tensor cores!  │
│ 8x H100 640GB      │ Q4 + Muon 4-bit (FP8 N-S)   │ ~181B     │ Full FP8 pipeline  │
└─────────────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHTS:

1. MEMORY DOMINATES: N-S precision mainly affects speed, not stored memory
   - Stored: weights + gradients + momentum (fixed per config)
   - Temporary: N-S buffers (freed after each layer)

2. FP8 IS WORTH IT ON HOPPER:
   - Ada (RTX 40xx): FP8 slower due to torch._scaled_mm overhead
   - Hopper (H100): Native FP8 tensor cores, expect 1.5-2x N-S speedup

3. INT8 IS VIABLE BUT SLOW:
   - Same memory as FP8, but torch._int_mm not as optimized
   - Good fallback if FP8 unavailable

4. INT4 DOES NOT WORK:
   - N-S diverges after 4 iterations
   - Don't use for Newton-Schulz (unlike Muon momentum which works at 4-bit)

5. MEMORY BREAKDOWN (Q4 + Muon 4-bit):
   - Weights: 0.5 B/p (Q4_0)
   - Gradients: 2.0 B/p (BF16, needed for backward)
   - Momentum: 0.5 B/p (4-bit packed)
   - TOTAL: 3.0 B/p → 4x better than FP16+Adam (12 B/p)
"""
    print(recommendations)

    # Specific model examples
    print("\n" + "-" * 90)
    print("EXAMPLE: Training Llama-style Models")
    print("-" * 90)

    models = [
        ("Llama-7B", 7e9),
        ("Llama-13B", 13e9),
        ("Llama-70B", 70e9),
        ("Llama-405B", 405e9),
    ]

    configs = [
        ("FP16+Adam (12 B/p)", 12),
        ("Q4+Muon 4-bit (3 B/p)", 3),
    ]

    print(f"\n{'Model':<15}", end="")
    for cfg_name, _ in configs:
        print(f"{cfg_name:<25}", end="")
    print()
    print("-" * 65)

    for model_name, params in models:
        print(f"{model_name:<15}", end="")
        for cfg_name, bpp in configs:
            mem_gb = (params * bpp) / 1e9
            print(f"{mem_gb:>6.0f} GB{'':<17}", end="")
        print()

    print(f"\n{'Model':<15} {'Min GPUs (FP16+Adam)':<25} {'Min GPUs (Q4+Muon)':<25}")
    print("-" * 65)

    for model_name, params in models:
        fp16_mem = (params * 12) / 1e9
        q4_mem = (params * 3) / 1e9

        # Find minimum GPU config
        def min_gpus(mem_gb):
            if mem_gb <= 24: return "1x RTX 4090"
            elif mem_gb <= 80: return "1x H100"
            elif mem_gb <= 640: return f"{int(np.ceil(mem_gb/80))}x H100"
            else: return f"{int(np.ceil(mem_gb/80))}x H100"

        print(f"{model_name:<15} {min_gpus(fp16_mem):<25} {min_gpus(q4_mem):<25}")


def main():
    print_memory_table()
    create_visualization()
    print_practical_recommendations()


if __name__ == "__main__":
    main()
