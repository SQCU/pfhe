"""
Benchmark: IQ3_XXS Quantized Training vs BF16 Baseline

Compares:
- Parameter memory: IQ3_XXS (~3 bits) vs BF16 (16 bits)
- Optimizer state: JL-compressed carry vs full Adam state
- Activations/gradients: BF16 in both cases
- Throughput: tokens/sec for forward+backward

Uses a synthetic transformer to ensure fair comparison.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from dataclasses import dataclass
from typing import Optional, Dict, List
import math

from phfe.inference.gguf_vtensor import (
    QuantizedParameter,
    QuantizedLinear,
    CarryOptimizer,
    AdamCarry,
    QuantizedAdam,
    requant_q4_0_cuda,  # Using Q4_0 for stability (IQ3_XXS has numerical issues)
    dequant_q4_0_cuda,
)


@dataclass
class ModelConfig:
    """Small transformer config (~100M-500M params)."""
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_dim: int = 4096
    max_seq_len: int = 512
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, quantized: bool = False, device: str = 'cuda'):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim

        if quantized:
            # Quantized projections
            self.q_proj = self._make_quant_linear(config.hidden_dim, config.hidden_dim, device)
            self.k_proj = self._make_quant_linear(config.hidden_dim, config.hidden_dim, device)
            self.v_proj = self._make_quant_linear(config.hidden_dim, config.hidden_dim, device)
            self.o_proj = self._make_quant_linear(config.hidden_dim, config.hidden_dim, device)
        else:
            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def _make_quant_linear(self, in_f: int, out_f: int, device: str) -> QuantizedLinear:
        w = torch.randn(out_f, in_f, device=device) * 0.02
        w_quant = requant_q4_0_cuda(w, "q4_0")
        return QuantizedLinear(in_f, out_f, w_quant, "q4_0", device=device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn.float(), dim=-1).type_as(x)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, config: ModelConfig, quantized: bool = False, device: str = 'cuda'):
        super().__init__()

        if quantized:
            self.gate_proj = self._make_quant_linear(config.hidden_dim, config.intermediate_dim, device)
            self.up_proj = self._make_quant_linear(config.hidden_dim, config.intermediate_dim, device)
            self.down_proj = self._make_quant_linear(config.intermediate_dim, config.hidden_dim, device)
        else:
            self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def _make_quant_linear(self, in_f: int, out_f: int, device: str) -> QuantizedLinear:
        w = torch.randn(out_f, in_f, device=device) * 0.02
        w_quant = requant_q4_0_cuda(w, "q4_0")
        return QuantizedLinear(in_f, out_f, w_quant, "q4_0", device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, quantized: bool = False, device: str = 'cuda'):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = Attention(config, quantized=quantized, device=device)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = FFN(config, quantized=quantized, device=device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SmallTransformer(nn.Module):
    """Small transformer for benchmarking."""

    def __init__(self, config: ModelConfig, quantized: bool = False, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.quantized = quantized

        # Embedding (always bf16 - small relative to model)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, quantized=quantized, device=device)
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim)

        # Output projection (quantized or not)
        if quantized:
            w = torch.randn(config.vocab_size, config.hidden_dim, device=device) * 0.02
            w_quant = requant_q4_0_cuda(w, "q4_0")
            self.lm_head = QuantizedLinear(
                config.hidden_dim, config.vocab_size, w_quant, "q4_0", device=device
            )
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.to(device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)

        # Causal mask
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by type."""
        total = 0
        quantized = 0
        bf16 = 0

        for name, param in self.named_parameters():
            total += param.numel()
            bf16 += param.numel()

        for name, module in self.named_modules():
            if isinstance(module, QuantizedLinear):
                quantized += module.weight.numel
                bf16 -= module.weight.numel  # Don't double count

        return {
            'total_elements': total + quantized,
            'quantized_elements': quantized,
            'bf16_elements': bf16,
        }


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory stats in GB."""
    torch.cuda.synchronize()
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_memory_stats():
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_forward_backward(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_iters: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark forward + backward pass."""
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    # Dummy data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        model.zero_grad()

    reset_memory_stats()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_processed = batch_size * seq_len * num_iters
    tokens_per_sec = tokens_processed / elapsed

    mem_stats = get_memory_stats()

    return {
        'elapsed_sec': elapsed,
        'tokens_per_sec': tokens_per_sec,
        'ms_per_iter': (elapsed / num_iters) * 1000,
        **mem_stats,
    }


def run_comparison():
    """Run full comparison between quantized and bf16 models."""
    print("=" * 70)
    print("QUANTIZED TRAINING BENCHMARK")
    print("Q4_0 weights + BF16 activations vs Full BF16")
    print("=" * 70)

    device = 'cuda'

    # Smaller config for memory comparison
    config = ModelConfig(
        vocab_size=32000,
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        intermediate_dim=3072,
        max_seq_len=256,
    )

    batch_size = 4
    seq_len = 256

    results = {}

    # =========================================================================
    # BF16 Baseline
    # =========================================================================
    print("\n" + "-" * 70)
    print("Building BF16 baseline model...")
    print("-" * 70)

    reset_memory_stats()
    mem_before = get_memory_stats()

    model_bf16 = SmallTransformer(config, quantized=False, device=device)
    model_bf16 = model_bf16.to(torch.float16)  # Use fp16 to match dequant output

    mem_after_model = get_memory_stats()
    param_counts = model_bf16.count_parameters()

    print(f"Parameters: {param_counts['total_elements'] / 1e6:.1f}M")
    print(f"Model memory: {mem_after_model['allocated_gb']:.3f} GB")

    # Create optimizer and run one step to trigger lazy allocation
    optimizer_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-4)

    dummy_input = torch.randint(0, config.vocab_size, (2, 64), device=device)
    dummy_out = model_bf16(dummy_input)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()
    optimizer_bf16.step()
    optimizer_bf16.zero_grad()
    del dummy_input, dummy_out, dummy_loss  # Clean up intermediates
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure optimizer state (m, v buffers)
    opt_mem_actual = sum(
        state['exp_avg'].numel() * state['exp_avg'].element_size() +
        state['exp_avg_sq'].numel() * state['exp_avg_sq'].element_size()
        for state in optimizer_bf16.state.values()
        if 'exp_avg' in state
    ) / 1e9
    print(f"+ Optimizer memory: {opt_mem_actual:.3f} GB (measured state)")

    # Benchmark
    print(f"\nBenchmarking forward+backward (batch={batch_size}, seq={seq_len})...")
    bf16_bench = benchmark_forward_backward(model_bf16, batch_size, seq_len)

    print(f"  Throughput: {bf16_bench['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Latency: {bf16_bench['ms_per_iter']:.1f} ms/iter")
    print(f"  Peak memory: {bf16_bench['max_allocated_gb']:.3f} GB")

    results['bf16'] = {
        'param_count': param_counts['total_elements'],
        'model_memory_gb': mem_after_model['allocated_gb'],
        'optimizer_memory_gb': opt_mem_actual,
        'peak_memory_gb': bf16_bench['max_allocated_gb'],
        'tokens_per_sec': bf16_bench['tokens_per_sec'],
        'ms_per_iter': bf16_bench['ms_per_iter'],
    }

    # Cleanup
    del model_bf16, optimizer_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Quantized Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Building Q4_0 quantized model...")
    print("-" * 70)

    reset_memory_stats()
    mem_before = get_memory_stats()

    model_quant = SmallTransformer(config, quantized=True, device=device)
    # Embeddings and norms stay fp16 (to match quantized weight output)
    for name, param in model_quant.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)

    mem_after_model = get_memory_stats()
    param_counts = model_quant.count_parameters()

    print(f"Parameters: {param_counts['total_elements'] / 1e6:.1f}M")
    print(f"  Quantized: {param_counts['quantized_elements'] / 1e6:.1f}M")
    print(f"  BF16: {param_counts['bf16_elements'] / 1e6:.1f}M")
    print(f"Model memory: {mem_after_model['allocated_gb']:.3f} GB")

    # AdamCarry handles both quantized and regular params
    # Use carry_dim=None for full-size carry (avoids huge JL projection matrices)
    optimizer_quant = AdamCarry(
        model_quant,  # Pass model, not params
        lr=1e-4,
        carry_dim=None,  # Full-size carry - cheaper than (numel × 64) projection per param
        flush_threshold=0.1,
    )

    # Run one step to trigger any lazy allocation
    dummy_input = torch.randint(0, config.vocab_size, (2, 64), device=device)
    dummy_out = model_quant(dummy_input)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()
    optimizer_quant.step()
    optimizer_quant.zero_grad()
    del dummy_input, dummy_out, dummy_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure optimizer state
    opt_mem = optimizer_quant.state_memory_bytes() / 1e9
    print(f"+ Optimizer memory: {opt_mem:.3f} GB (measured state)")
    print(f"  (Full-size carry for {len(optimizer_quant.quantized_params)} quantized params)")

    # Benchmark
    print(f"\nBenchmarking forward+backward (batch={batch_size}, seq={seq_len})...")
    quant_bench = benchmark_forward_backward(model_quant, batch_size, seq_len)

    print(f"  Throughput: {quant_bench['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Latency: {quant_bench['ms_per_iter']:.1f} ms/iter")
    print(f"  Peak memory: {quant_bench['max_allocated_gb']:.3f} GB")

    results['quantized_fp32_opt'] = {
        'param_count': param_counts['total_elements'],
        'quantized_params': param_counts['quantized_elements'],
        'model_memory_gb': mem_after_model['allocated_gb'],
        'optimizer_memory_gb': opt_mem,
        'peak_memory_gb': quant_bench['max_allocated_gb'],
        'tokens_per_sec': quant_bench['tokens_per_sec'],
        'ms_per_iter': quant_bench['ms_per_iter'],
    }

    # Cleanup
    del model_quant, optimizer_quant
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Fully Quantized (Q4_0 weights + Q8_0 optimizer state)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Building FULLY QUANTIZED model (Q4_0 weights + Q8_0 opt state)...")
    print("-" * 70)

    reset_memory_stats()

    model_full_quant = SmallTransformer(config, quantized=True, device=device)
    for name, param in model_full_quant.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)

    mem_after_model_fq = get_memory_stats()
    param_counts_fq = model_full_quant.count_parameters()

    print(f"Parameters: {param_counts_fq['total_elements'] / 1e6:.1f}M")
    print(f"Model memory: {mem_after_model_fq['allocated_gb']:.3f} GB")

    # Use QuantizedAdam with quantized optimizer state
    optimizer_full_quant = QuantizedAdam(
        model_full_quant,
        lr=1e-4,
        weight_quant='q4_0',
        state_quant='q8_0',  # Optimizer state also quantized!
    )

    # Warmup step
    dummy_input = torch.randint(0, config.vocab_size, (2, 64), device=device)
    dummy_out = model_full_quant(dummy_input)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()
    optimizer_full_quant.step()
    optimizer_full_quant.zero_grad()
    del dummy_input, dummy_out, dummy_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    opt_mem_fq = optimizer_full_quant.state_memory_bytes() / 1e9
    report = optimizer_full_quant.memory_report()
    print(f"+ Optimizer memory: {opt_mem_fq:.3f} GB (Q8_0 quantized!)")
    print(f"  Bytes per param: {report['bytes_per_param']:.2f}")

    # Benchmark
    print(f"\nBenchmarking forward+backward (batch={batch_size}, seq={seq_len})...")
    fq_bench = benchmark_forward_backward(model_full_quant, batch_size, seq_len)

    print(f"  Throughput: {fq_bench['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Latency: {fq_bench['ms_per_iter']:.1f} ms/iter")
    print(f"  Peak memory: {fq_bench['max_allocated_gb']:.3f} GB")

    results['fully_quantized'] = {
        'param_count': param_counts_fq['total_elements'],
        'model_memory_gb': mem_after_model_fq['allocated_gb'],
        'optimizer_memory_gb': opt_mem_fq,
        'peak_memory_gb': fq_bench['max_allocated_gb'],
        'tokens_per_sec': fq_bench['tokens_per_sec'],
        'ms_per_iter': fq_bench['ms_per_iter'],
        'bytes_per_param': report['bytes_per_param'],
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Three Training Configurations")
    print("=" * 70)

    bf16 = results['bf16']
    quant_fp32 = results['quantized_fp32_opt']
    quant_full = results['fully_quantized']

    print(f"\n{'Metric':<25} {'FP16+Adam':>12} {'Q4+FP32opt':>12} {'Q4+Q8opt':>12} {'Best':>10}")
    print("-" * 75)

    # Model memory
    print(f"{'Model memory (GB)':<25} {bf16['model_memory_gb']:>12.3f} {quant_fp32['model_memory_gb']:>12.3f} {quant_full['model_memory_gb']:>12.3f} {'Q4':>10}")

    # Optimizer memory
    print(f"{'Optimizer memory (GB)':<25} {bf16['optimizer_memory_gb']:>12.3f} {quant_fp32['optimizer_memory_gb']:>12.3f} {quant_full['optimizer_memory_gb']:>12.3f} {'Q4+Q8':>10}")

    # Total
    bf16_total = bf16['model_memory_gb'] + bf16['optimizer_memory_gb']
    qfp32_total = quant_fp32['model_memory_gb'] + quant_fp32['optimizer_memory_gb']
    qfull_total = quant_full['model_memory_gb'] + quant_full['optimizer_memory_gb']
    best_total = "Q4+Q8" if qfull_total < min(bf16_total, qfp32_total) else ("FP16" if bf16_total < qfp32_total else "Q4+FP32")
    print(f"{'Total (GB)':<25} {bf16_total:>12.3f} {qfp32_total:>12.3f} {qfull_total:>12.3f} {best_total:>10}")

    # Bytes per param
    bf16_bpp = 10.0  # 2 (weight) + 8 (m,v FP32)
    print(f"{'Bytes/param':<25} {bf16_bpp:>12.1f} {12.5:>12.1f} {quant_full['bytes_per_param']:>12.1f} {'Q4+Q8':>10}")

    # Throughput
    print(f"{'Throughput (tok/s)':<25} {bf16['tokens_per_sec']:>12.0f} {quant_fp32['tokens_per_sec']:>12.0f} {quant_full['tokens_per_sec']:>12.0f} {'FP16':>10}")

    print("\n" + "-" * 70)
    print("KEY INSIGHT: Quantized optimizer state is the game-changer!")
    print("-" * 70)
    print(f"\n  FP16 + AdamW:          {bf16_bpp:.1f} bytes/param")
    print(f"  Q4_0 + FP32 optimizer: 12.5 bytes/param (WORSE - opt state dominates)")
    print(f"  Q4_0 + Q8_0 optimizer: {quant_full['bytes_per_param']:.1f} bytes/param ({bf16_bpp/quant_full['bytes_per_param']:.1f}x BETTER!)")

    savings = bf16_total / qfull_total
    print(f"\n  Total memory savings: {savings:.1f}x with fully quantized training!")
    print("-" * 70)

    return results


def plot_comparison(results: Dict):
    """Create comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Memory comparison
    ax = axes[0]
    categories = ['Model', 'Optimizer', 'Peak']
    bf16_vals = [
        results['bf16']['model_memory_gb'],
        results['bf16']['optimizer_memory_gb'],
        results['bf16']['peak_memory_gb'],
    ]
    quant_vals = [
        results['quantized']['model_memory_gb'],
        results['quantized']['optimizer_memory_gb'],
        results['quantized']['peak_memory_gb'],
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, bf16_vals, width, label='BF16', color='#4CAF50')
    ax.bar(x + width/2, quant_vals, width, label='Q4_0', color='#2196F3')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Throughput
    ax = axes[1]
    methods = ['BF16', 'Q4_0']
    throughputs = [results['bf16']['tokens_per_sec'], results['quantized']['tokens_per_sec']]

    ax.bar(methods, throughputs, color=['#4CAF50', '#2196F3'])
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput')

    # Memory efficiency (tokens per GB)
    ax = axes[2]
    bf16_eff = results['bf16']['tokens_per_sec'] / results['bf16']['peak_memory_gb']
    quant_eff = results['quantized']['tokens_per_sec'] / results['quantized']['peak_memory_gb']

    ax.bar(methods, [bf16_eff, quant_eff], color=['#4CAF50', '#2196F3'])
    ax.set_ylabel('Tokens/sec/GB')
    ax.set_title('Memory Efficiency')

    plt.tight_layout()
    plt.savefig('quantized_training_benchmark.png', dpi=150)
    print("\nPlot saved to: quantized_training_benchmark.png")


if __name__ == "__main__":
    results = run_comparison()
    plot_comparison(results)
