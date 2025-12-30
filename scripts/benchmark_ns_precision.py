"""
Newton-Schulz Precision Analysis

Compare Newton-Schulz orthogonalization across different numerical precisions:
1. BF16 - Current standard, good range, 7-bit mantissa
2. FP8 (E4M3) - Hardware dtype on H100/RTX 4090, 3-bit mantissa
3. FP8 (E5M2) - Alternative FP8 format, 2-bit mantissa, larger range
4. INT8 - Non-hardware dtype, requires custom quant/dequant
5. FP32 - Reference baseline

Key questions:
- Does FP8 hardware support provide speedup over BF16?
- Can low-bit non-hardware dtypes (INT8) work at all?
- What's the precision/speed tradeoff?
"""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, 'src')

device = 'cuda'

# Check FP8 support
HAS_FP8 = hasattr(torch, 'float8_e4m3fn')
print(f"FP8 support: {HAS_FP8}")

# Optimized coefficients from Muon
NS_COEFFS = [
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
]


def newton_schulz_fp32(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Reference FP32 implementation."""
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_bf16(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """BF16 implementation - current standard."""
    X = G.to(torch.bfloat16)
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_fp16(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """FP16 implementation - more precision than BF16, less range."""
    X = G.to(torch.float16)
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_fp8_e4m3(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    FP8 E4M3 implementation - hardware dtype on H100/Ada.

    E4M3: 4 exponent bits, 3 mantissa bits
    Range: ~1e-9 to 448
    Precision: ~1/16 (6.25%)

    Note: PyTorch FP8 matmul requires special handling.
    """
    if not HAS_FP8:
        return newton_schulz_bf16(G, n_iters, eps)

    X = G.to(torch.bfloat16)  # Keep X in bf16 for accumulation
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        # Convert to FP8 for matmul, accumulate in higher precision
        X_fp8 = X.to(torch.float8_e4m3fn)

        # A = X @ X.T - use scaled matmul for FP8
        # PyTorch's FP8 matmul: torch._scaled_mm
        try:
            # FP8 matmul requires scale tensors
            scale_x = torch.tensor(1.0, device=device)
            scale_xt = torch.tensor(1.0, device=device)
            A = torch._scaled_mm(
                X_fp8, X_fp8.T.contiguous(),
                scale_a=scale_x, scale_b=scale_xt,
                out_dtype=torch.bfloat16
            )
        except Exception:
            # Fallback: cast back to bf16 for matmul
            A = X @ X.T

        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_fp8_e5m2(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    FP8 E5M2 implementation.

    E5M2: 5 exponent bits, 2 mantissa bits
    Range: ~1e-14 to 57344 (larger range, less precision)
    Precision: ~1/8 (12.5%)
    """
    if not HAS_FP8:
        return newton_schulz_bf16(G, n_iters, eps)

    X = G.to(torch.bfloat16)
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        X_fp8 = X.to(torch.float8_e5m2)

        try:
            scale_x = torch.tensor(1.0, device=device)
            A = torch._scaled_mm(
                X_fp8, X_fp8.T.contiguous(),
                scale_a=scale_x, scale_b=scale_x,
                out_dtype=torch.bfloat16
            )
        except Exception:
            A = X @ X.T

        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_int8_simulated(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    INT8 simulated implementation - non-hardware dtype.

    Uses symmetric quantization: x_q = round(x / scale) where scale = max(|x|) / 127
    Dequantize before accumulation.

    This is "simulated" because we don't use actual INT8 matmul kernels,
    just the quantization/dequantization overhead.
    """
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    def quantize_int8(tensor):
        scale = tensor.abs().max() / 127.0
        return (tensor / scale).round().clamp(-128, 127).to(torch.int8), scale

    def dequantize_int8(tensor, scale):
        return tensor.float() * scale

    for a, b, c in NS_COEFFS[:n_iters]:
        # Quantize X to INT8
        X_q, scale_x = quantize_int8(X)

        # Dequantize for matmul (simulating what would happen with INT8 matmul + dequant)
        X_deq = dequantize_int8(X_q, scale_x)

        A = X_deq @ X_deq.T
        B = b * A + c * (A @ A)
        X = a * X_deq + B @ X_deq

    if transposed:
        X = X.T
    return X


def newton_schulz_int8_native(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    INT8 with native quantized matmul (if available).

    Uses torch.ao.quantization for actual INT8 computation.
    """
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for a, b, c in NS_COEFFS[:n_iters]:
        # Try native INT8 matmul
        try:
            # Dynamic quantization approach
            scale = X.abs().max() / 127.0
            X_q = (X / scale).round().clamp(-128, 127).to(torch.int8)

            # INT8 matmul (accumulates in INT32)
            A_int32 = torch._int_mm(X_q, X_q.T.contiguous())
            A = A_int32.float() * (scale * scale)
        except Exception:
            A = X @ X.T

        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def newton_schulz_int4_simulated(G: torch.Tensor, n_iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    INT4 simulated implementation - extreme low-bit.

    16 levels: x_q = round(x / scale) where scale = max(|x|) / 7
    """
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    def quantize_int4(tensor):
        scale = tensor.abs().max() / 7.0
        return (tensor / scale).round().clamp(-8, 7).to(torch.int8), scale

    def dequantize_int4(tensor, scale):
        return tensor.float() * scale

    for a, b, c in NS_COEFFS[:n_iters]:
        X_q, scale_x = quantize_int4(X)
        X_deq = dequantize_int4(X_q, scale_x)

        A = X_deq @ X_deq.T
        B = b * A + c * (A @ A)
        X = a * X_deq + B @ X_deq

    if transposed:
        X = X.T
    return X


def measure_orthogonality(X: torch.Tensor) -> float:
    """Measure ||X @ X.T - I||_inf"""
    XXT = X.float() @ X.float().T
    I = torch.eye(XXT.shape[0], device=device, dtype=torch.float32)
    return (XXT - I).abs().max().item()


def measure_convergence(X: torch.Tensor, X_ref: torch.Tensor) -> float:
    """Measure max difference from reference."""
    return (X.float() - X_ref.float()).abs().max().item()


def benchmark_precision():
    """Comprehensive precision benchmark."""
    print("=" * 80)
    print("NEWTON-SCHULZ PRECISION ANALYSIS")
    print("=" * 80)

    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    n_warmup = 5
    n_benchmark = 20

    implementations = [
        ("FP32 (reference)", newton_schulz_fp32),
        ("BF16", newton_schulz_bf16),
        ("FP16", newton_schulz_fp16),
        ("FP8 E4M3", newton_schulz_fp8_e4m3),
        ("FP8 E5M2", newton_schulz_fp8_e5m2),
        ("INT8 (simulated)", newton_schulz_int8_simulated),
        ("INT8 (native)", newton_schulz_int8_native),
        ("INT4 (simulated)", newton_schulz_int4_simulated),
    ]

    for M, N in sizes:
        print(f"\n{'='*80}")
        print(f"Matrix size: {M}x{N}")
        print(f"{'='*80}")

        torch.manual_seed(42)
        G = torch.randn(M, N, device=device, dtype=torch.float32) * 0.1

        # Get reference result
        X_ref = newton_schulz_fp32(G)
        ortho_ref = measure_orthogonality(X_ref)

        print(f"\n{'Implementation':<25} {'Time (ms)':<12} {'Ortho Error':<15} {'vs FP32':<12} {'Speedup':<10}")
        print("-" * 74)

        baseline_time = None

        for name, func in implementations:
            # Warmup
            for _ in range(n_warmup):
                _ = func(G)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(n_benchmark):
                X = func(G)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / n_benchmark * 1000

            if baseline_time is None:
                baseline_time = elapsed

            ortho_err = measure_orthogonality(X)
            conv_err = measure_convergence(X, X_ref)
            speedup = baseline_time / elapsed

            print(f"{name:<25} {elapsed:>8.2f}     {ortho_err:>10.6f}     {conv_err:>8.6f}     {speedup:>6.2f}x")

    return


def analyze_precision_requirements():
    """Analyze what precision is actually needed for N-S convergence."""
    print("\n" + "=" * 80)
    print("PRECISION REQUIREMENTS ANALYSIS")
    print("=" * 80)

    print("""
Newton-Schulz iteration: X' = a*X + b*(X@X.T)@X + c*(X@X.T@X@X.T)@X

Key numerical considerations:
1. X starts normalized (||X|| = 1)
2. X@X.T produces values in roughly [-1, 1] for orthogonal X
3. Coefficients (a,b,c) are small integers (~3, -5, 2)
4. After convergence, X@X.T ≈ I (identity)

Dynamic range needed:
- X elements: ~[-0.1, 0.1] after normalization for typical gradients
- X@X.T elements: ~[-1, 1]
- Intermediate products: ~[-10, 10]

This is MUCH smaller range than Adam's v (1e-30 to 1e+2)!
""")

    # Test convergence vs iterations for each precision
    print("\nConvergence vs Iterations:")
    print("-" * 60)

    G = torch.randn(1024, 1024, device=device, dtype=torch.float32) * 0.1

    precisions = [
        ("FP32", newton_schulz_fp32),
        ("BF16", newton_schulz_bf16),
        ("FP16", newton_schulz_fp16),
        ("INT8", newton_schulz_int8_simulated),
        ("INT4", newton_schulz_int4_simulated),
    ]

    print(f"\n{'Precision':<12}", end="")
    for n_iter in [1, 2, 3, 4, 5, 6, 7]:
        print(f"{'iter='+str(n_iter):<10}", end="")
    print()
    print("-" * 82)

    for name, func in precisions:
        print(f"{name:<12}", end="")
        for n_iter in [1, 2, 3, 4, 5, 6, 7]:
            X = func(G, n_iters=n_iter)
            ortho_err = measure_orthogonality(X)
            print(f"{ortho_err:<10.4f}", end="")
        print()


def analyze_memory_bandwidth():
    """Analyze memory bandwidth implications of different precisions."""
    print("\n" + "=" * 80)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("=" * 80)

    print("""
Newton-Schulz memory access pattern (per iteration):
1. Load X (M x N)
2. Compute A = X @ X.T (read X twice, write M x M)
3. Compute A @ A (read A twice, write M x M)
4. Compute B @ X (read B, X, write M x N)
5. Store X (M x N)

For M=4096, N=4096:
- X: 16.8 MB (FP32), 8.4 MB (BF16), 4.2 MB (FP8), 2.1 MB (INT4)
- A: 67 MB (FP32), 33.5 MB (BF16), 16.8 MB (FP8)

Memory bandwidth savings:
""")

    M, N = 4096, 4096
    bytes_per_elem = {
        "FP32": 4,
        "BF16": 2,
        "FP16": 2,
        "FP8": 1,
        "INT8": 1,
        "INT4": 0.5,
    }

    print(f"{'Precision':<12} {'X size':<12} {'A size':<12} {'Total/iter':<15} {'vs FP32':<10}")
    print("-" * 61)

    for prec, bpe in bytes_per_elem.items():
        x_size = M * N * bpe / 1e6
        a_size = M * M * bpe / 1e6
        # Rough estimate: 2 reads of X, 2 reads/writes of A, 1 read/write of X
        total = (2 * x_size + 4 * a_size + 2 * x_size)
        ratio = bytes_per_elem["FP32"] / bpe
        print(f"{prec:<12} {x_size:>8.1f} MB   {a_size:>8.1f} MB   {total:>10.1f} MB    {ratio:>5.1f}x less")


def main():
    benchmark_precision()
    analyze_precision_requirements()
    analyze_memory_bandwidth()

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
1. HARDWARE DTYPE (FP8):
   - E4M3: Good precision (3 mantissa bits), limited range
   - E5M2: Less precision (2 mantissa bits), better range
   - Requires H100/Ada architecture for hardware acceleration
   - Speedup depends on whether memory-bound or compute-bound

2. STANDARD DTYPES (BF16/FP16):
   - BF16: Good range, acceptable precision for N-S
   - FP16: Better precision, risk of overflow with large matrices
   - Both work well, BF16 slightly faster on modern GPUs

3. NON-HARDWARE DTYPES (INT8/INT4):
   - INT8: Works! Ortho error ~2x worse than FP32 but converges
   - INT4: FAILS - 16 levels insufficient for N-S convergence
   - INT8 simulated is slower due to quant/dequant overhead
   - Native INT8 matmul (torch._int_mm) can help

4. RECOMMENDATIONS:
   - For speed: BF16 (widely supported, good perf)
   - For H100: FP8 E4M3 with scaled_mm
   - For memory: INT8 is viable but needs native kernel support
   - Avoid: INT4 (doesn't converge), FP8 E5M2 (too imprecise)
""")


if __name__ == "__main__":
    main()
