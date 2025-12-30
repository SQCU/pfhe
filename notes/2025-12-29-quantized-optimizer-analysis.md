# Quantized Optimizer State Analysis

## Summary

Quantizing optimizer state (m, v) alongside weights enables 4x memory reduction for training, but requires careful handling of v's dynamic range.

## Memory Comparison (per parameter)

| Configuration | Bytes/Param | Notes |
|---------------|-------------|-------|
| FP16 + FP32 Adam | 10.0 | 2B weight + 8B state |
| Q4_0 + FP32 Adam | 8.6 | Worse! State dominates |
| Q4_0 + 8-bit Codebook | 2.6 | 3.9x savings |
| Q4_0 + IQ3 Codebook | ~1.3 | 7.7x savings (theoretical) |

## Key Insights

### 1. Log-Space Transform is Essential

v (second moment) spans 70+ orders of magnitude (1e-30 to 1e+2). Direct quantization fails.

```
v range:     1e-30 to 1e+2  (impossible to quantize)
log(v) range: -69 to 4.6    (easily quantizable)
```

### 2. Density-Aware Codebook Adds ~5pp

Optimizer state distributions are non-uniform (~40% entropy):
- m: Gaussian around 0
- log(v): Clustered around [-28, -22]

Allocating more codebook entries to dense regions improves precision:
- Uniform codebook: +3.7% gap to FP32
- Density-aware: -1.7% gap (beats FP32!)

### 3. Stochastic Rounding Acts as Regularization

Probabilistic rounding reduces bias accumulation and provides implicit regularization, explaining why quantized optimizer can beat FP32 baseline.

## Throughput Analysis

| Implementation | Speed vs FP16+Adam |
|----------------|-------------------|
| Python (current) | 0.14x (7x slower) |
| Optimized Python | 0.2x (5x slower) |
| Fused CUDA kernel | ~0.7x (estimated) |

### Bottleneck Breakdown (4M params)

- EMA update: 1.5ms
- Log transform: 1.1ms
- Codebook lookup: 1.1ms
- Stochastic quantize: 1.8ms
- Total overhead: ~8ms vs ~1ms for fused FP32 Adam

## Scaling Examples

| Model | FP16+Adam | Q4+Codebook | Fits on... |
|-------|-----------|-------------|------------|
| 1B | 10 GB | 2.6 GB | RTX 3060 12GB |
| 7B | 70 GB | 17.9 GB | RTX 4090 24GB |
| 70B | 700 GB | 179.4 GB | 8x H100 |

## Implementation Files

- `src/phfe/inference/gguf_vtensor/log_quantized_adam.py` - Log-space optimizer
- `src/phfe/inference/gguf_vtensor/iq3_opt_codebook.py` - Custom codebooks
- `src/phfe/inference/gguf_vtensor/quantized_adam.py` - Q8_0 optimizer state
- `scripts/test_codebook_convergence.py` - Convergence benchmarks

## To Make Practical

1. Fused CUDA kernel for quantize/dequantize + Adam update
2. Pack 3-bit indices for IQ3-level compression
3. Async quantization (overlap with forward pass)

## Fused Triton Kernel Results

Implemented fused Triton kernels for codebook Adam:

| Implementation | Time (4M params) | vs PyTorch Adam |
|----------------|------------------|-----------------|
| PyTorch Adam   | 0.048 ms         | 1.0x            |
| Python codebook| ~1.0 ms          | ~20x slower     |
| Triton codebook| 0.69 ms          | 14x slower      |
| Triton linear  | 0.61 ms          | 12.6x slower    |

### Why Still Slower?

PyTorch's fused Adam is extremely optimized:
- Single kernel launch for all operations
- Optimized memory access patterns
- No log/exp operations needed

Our kernel has overhead from:
- Log-space transform (exp for dequant, log for requant)
- Multiple memory passes (load indices, load codebook, store indices)

### Real-World Impact

During training, optimizer step is typically <5% of iteration time:
- Forward pass: ~40%
- Backward pass: ~50%
- Optimizer step: ~5%
- Data loading: ~5%

So 12x slower optimizer → ~1.6x slower total training (not 12x).

For memory-bound scenarios (large batch, large model), the compute
overhead may be hidden by memory bandwidth anyway.

### When to Use

Use quantized optimizer when:
1. GPU memory is the bottleneck (can't fit model otherwise)
2. Training time is acceptable (fine-tuning, not pretraining)
3. Need to train larger models on consumer hardware

Don't use when:
1. Have sufficient VRAM for FP32 state
2. Training throughput is critical
3. Pretraining at scale

## IQ3 vs 8-bit Comparison

Tested three quantized optimizer variants with fused Triton kernels:

| Variant | State Size | Throughput | Convergence |
|---------|------------|------------|-------------|
| PyTorch AdamW | 8 B/param | 1.0x | baseline |
| IQ3 no-logspace | 0.75 B/param | 8.4x slower | +191% gap (DIVERGES) |
| IQ3 logspace | 0.75 B/param | 13.0x slower | +190% gap (DIVERGES) |
| AdamW 8-bit log | 2 B/param | 12.6x slower | +10.7% gap (WORKS) |

### Key Finding: 3 bits is not enough

Both IQ3 variants diverge regardless of log-space transform. 8 levels simply
cannot represent the required precision for optimizer state EMAs.

Log-space doesn't help IQ3 because:
- Without log: v range 1e-12 to 1e-2 → 8 levels can't cover it
- With log: log(v) range -28 to -5 → 8 levels = ~3 units/level, still too coarse

### Throughput Analysis

IQ3 no-logspace is faster (8.4x) than IQ3 logspace (13.0x) because:
- No exp/log operations needed
- But it diverges anyway, so speed doesn't matter

8-bit logspace is the only practical option:
- 12.6x slower than PyTorch (optimizer step only)
- 4x memory savings
- +10.7% convergence gap (acceptable)

## Conclusion

**The practical quantized optimizer is 8-bit log-space AdamW:**
- 4x memory reduction (8 → 2 bytes/param for state)
- 12.6x slower optimizer step (but optimizer is ~5% of training time)
- ~1.6x slower total training throughput
- +10.7% convergence gap (acceptable for fine-tuning)

**IQ3 (3-bit) does not work** - insufficient precision causes divergence regardless
of log-space transform. The 10.7x memory savings is not achievable with current
approach.

**Future work**: Error feedback or higher-order corrections might enable lower-bit
quantization, but 8-bit is the practical minimum for now.

## Muon: Newton-Schulz Alternative

Muon uses Newton-Schulz orthogonalization instead of Adam's moment estimates,
completely sidestepping the dynamic range problem.

### Benchmark Results

| Optimizer | State Memory | Convergence | Speed |
|-----------|--------------|-------------|-------|
| Adam FP32 | 8 B/param | baseline | 1.0x |
| Adam 8-bit log | 2 B/param | +10.7% gap | 0.08x |
| Muon + momentum | 4 B/param | **-6.7% gap** | 1.1x |
| Muon zero-state | 0 B/param | +1.6% gap | 0.7x |

### Key Insight

Muon doesn't need second moment (v) at all! Newton-Schulz orthogonalization:
```
X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k.T @ X_k
```

This means:
- No v with its 70+ orders of magnitude dynamic range
- No log-space transform needed
- Optional momentum (just m, not both m and v)

### Memory Comparison (with Q4 weights)

| Config | Total B/param | 7B Model |
|--------|---------------|----------|
| Adam FP32 state | 10.56 | 73.9 GB |
| Adam 8-bit state | 4.56 | 31.9 GB |
| Muon + momentum | 6.56 | 45.9 GB |
| **Muon zero-state** | **2.56** | **17.9 GB** |

### Recommendation

**For minimum memory**: Muon zero-state (2.56 B/param, 4.1x savings)
- Slightly worse convergence (+1.6%)
- Slower due to Newton-Schulz iterations
- But fits much larger models

**For best convergence**: Muon with momentum (6.56 B/param)
- Actually beats Adam (-6.7% better!)
- Faster than Adam on this benchmark
- Half the state memory of Adam

**For proven reliability**: Adam 8-bit log-space (4.56 B/param)
- Well-understood convergence properties
- 12.6x slower optimizer step
- But optimizer is small fraction of training time

## Low-Bit Muon Momentum

Since Muon only needs m (not v), and m has narrow dynamic range, we can
quantize it more aggressively than Adam's state.

### Benchmark Results

| Variant | State Size | Convergence | Speed |
|---------|------------|-------------|-------|
| FP32 momentum | 4.0 B/p | baseline | 1.0x |
| 8-bit momentum | 1.0 B/p | +0.0% | 0.79x |
| **4-bit momentum** | **0.5 B/p** | **+4.0%** | 0.78x |
| 3-bit momentum | 0.38 B/p | +76.0% (too aggressive) | 0.74x |
| 2-bit momentum | 0.25 B/p | NaN (explodes) | - |

### Key Finding: 4-bit Works!

Unlike Adam where 3-bit fails catastrophically, Muon's momentum can be
quantized to 4 bits with only +4% convergence gap. This is because:

1. **m has narrow range**: typically [-0.01, 0.01], symmetric around 0
2. **No accumulation issues**: momentum is a simple EMA, not squared values
3. **16 levels sufficient**: covers the distribution well

### Final Memory Budget (with 4-bit Muon momentum)

| Component | Bytes/param |
|-----------|-------------|
| Q4 weights | 0.56 |
| Gradients (FP16) | 2.0 |
| 4-bit momentum | 0.5 |
| **TOTAL** | **3.06** |

vs FP16 + Adam: 12 B/param → **3.9x compression!**

### Scaling

| Model | FP16 + Adam | Q4 + 4-bit Muon | Savings |
|-------|-------------|-----------------|---------|
| 7B | 84 GB | 21.4 GB | 3.9x |
| 70B | 840 GB | 214 GB | 3.9x |

This enables training 70B on 8x H100 (80GB each) that would otherwise need 32x.

## Fused Triton Kernel for 4-bit Muon

Implemented fused Triton kernels for 4-bit Muon optimizer step:

### Performance Results (4M params)

| Implementation | Time (ms) | vs FP32 Python | Memory |
|----------------|-----------|----------------|--------|
| Python FP32 momentum | 0.140 | 1.0x | 4.0 B/param |
| **Fused 4-bit (deterministic)** | **0.023** | **6.1x faster** | 1.0 B/param |
| Fused 4-bit (stochastic) | 0.096 | 1.5x faster | 1.0 B/param |
| Fused 4-bit (packed) | 0.065 | 2.2x faster | 0.5 B/param |
| Newton-Schulz (5 iters) | 2.512 | - | overhead |

### Convergence Results

| Variant | Final Loss | Gap vs FP32 |
|---------|------------|-------------|
| FP32 Python | 0.1357 | baseline |
| Fused 4-bit (deterministic) | 0.1383 | **+1.9%** |
| Fused 4-bit (stochastic) | 0.1445 | +6.5% |

### Key Findings

1. **Newton-Schulz dominates**: 96% of total step time is N-S orthogonalization
2. **Deterministic > Stochastic**: For 4-bit, deterministic rounding works better
3. **6x faster momentum step**: Fused kernel eliminates Python overhead
4. **Packed storage works**: 0.5 B/param with minimal slowdown

### Architecture

The fused kernel combines:
1. Dequantize 4-bit momentum (linear scaling)
2. Momentum EMA update: `m_new = β*m + ortho_grad`
3. Requantize to 4-bit (with optional stochastic rounding)
4. Compute update: `update = ortho_grad + β*m_new`
5. Apply to weights

Newton-Schulz stays in PyTorch because matrix ops (`X @ X.T @ X`) don't
fuse well into element-wise Triton kernels.

### Implementation

```python
from phfe.inference.gguf_vtensor.fused_muon_4bit import FusedMuon4Bit

optimizer = FusedMuon4Bit(
    model,
    lr=0.02,
    momentum=0.95,
    n_iters=5,
    stochastic=False,  # Deterministic is faster and better
    packed=False,      # True for 0.5 B/param (experimental)
)
```

### Total Training Memory Budget

With fused 4-bit Muon:

| Component | Bytes/param |
|-----------|-------------|
| Q4 weights | 0.56 |
| Gradients (FP16) | 2.0 |
| 4-bit momentum (packed) | 0.5 |
| **TOTAL** | **3.06** |

vs FP16 + Adam: 12 B/param → **3.9x compression**

## Triton Newton-Schulz Optimization

Newton-Schulz dominates Muon step time (~96%), so optimizing it matters.
Implemented Triton kernels for the matmul-heavy N-S iteration.

### N-S Only Benchmark

| Size | PyTorch (ms) | Triton (ms) | Speedup |
|------|--------------|-------------|---------|
| 256x256 | 0.52 | 0.98 | 0.53x (slower) |
| 1024x1024 | 0.49 | 0.99 | 0.50x (slower) |
| 2048x2048 | 2.03 | 1.94 | 1.04x |
| **4096x4096** | 18.34 | 13.69 | **1.34x** |

### Full Muon Step Benchmark (N-S + 4-bit momentum)

| Layer Size | PyTorch | Triton | Hybrid | Speedup |
|------------|---------|--------|--------|---------|
| 256x256 | 0.61 ms | 1.09 ms | 0.62 ms | 1.0x |
| 1024x1024 | 0.61 ms | 1.06 ms | 0.57 ms | 1.1x |
| **4096x4096** | 18.24 ms | 14.35 ms | **13.85 ms** | **1.32x** |
| **4096x11008** | 38.14 ms | 26.35 ms | **26.43 ms** | **1.45x** |

### Key Findings

1. **Triton wins for LLM-sized layers**: 1.3-1.5x faster for >=2048
2. **PyTorch wins for small layers**: cuBLAS overhead lower than Triton launch
3. **Hybrid is optimal**: Automatically selects best implementation

### Implementation

```python
from phfe.inference.gguf_vtensor.newton_schulz_triton import newton_schulz_hybrid

# Automatic selection based on matrix size
ortho_grad = newton_schulz_hybrid(grad_2d, n_iters=5, size_threshold=2048)
```

### Real-World Impact

For 7B model training:
- Without Triton N-S: ~38ms per layer
- With Triton N-S: ~26ms per layer
- **~1.45x faster optimizer step overall**

Combined with 4-bit momentum (8x memory savings), this makes Muon practical
for consumer GPU training of large models
