"""
Analyze optimizer state distributions to design optimal codebooks.

Key question: Does log-space + custom codebook compound benefits,
or is log-space alone sufficient (making distribution ~uniform)?
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
import math

from phfe.inference.gguf_vtensor import QuantizedLinear, requant_q4_0_cuda


def analyze_distributions():
    """Train a model and analyze m, v distributions."""

    device = 'cuda'
    torch.manual_seed(42)

    # Create a realistic small model
    class SmallMLP(nn.Module):
        def __init__(self, dim=512, layers=4):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(layers):
                w = torch.randn(dim, dim, device=device) * 0.02
                w_q = requant_q4_0_cuda(w, 'q4_0')
                self.layers.append(QuantizedLinear(dim, dim, w_q, 'q4_0', device=device))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
            return x

    model = SmallMLP()

    # Standard Adam (FP32) to collect true distributions
    # We'll manually track m and v
    params = []
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'grad_holder'):
            params.append(module.weight)

    # Initialize m, v in FP32
    m_states = [torch.zeros(p.numel, device=device) for p in params]
    v_states = [torch.zeros(p.numel, device=device) for p in params]

    beta1, beta2 = 0.9, 0.999

    # Training loop to accumulate realistic m, v values
    print("Training to accumulate optimizer state distributions...")
    x = torch.randn(32, 512, device=device, dtype=torch.float16)
    y = torch.randn(32, 512, device=device, dtype=torch.float16)

    for step in range(100):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()

        # Update m, v manually
        for i, p in enumerate(params):
            if p.grad is not None:
                grad = p.grad.float().view(-1)
                m_states[i] = beta1 * m_states[i] + (1 - beta1) * grad
                v_states[i] = beta2 * v_states[i] + (1 - beta2) * grad.square()
                p.zero_grad()

        if step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    # Concatenate all m, v values
    all_m = torch.cat(m_states).cpu().numpy()
    all_v = torch.cat(v_states).cpu().numpy()
    all_log_v = np.log(np.clip(all_v, 1e-38, None))

    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Analyze m distribution
    print("\n--- First Moment (m) ---")
    print(f"  Range: [{all_m.min():.2e}, {all_m.max():.2e}]")
    print(f"  Mean:  {all_m.mean():.2e}")
    print(f"  Std:   {all_m.std():.2e}")
    print(f"  Shape: {'Gaussian-like' if abs(all_m.mean()) < all_m.std() else 'Skewed'}")

    # Histogram of m
    hist_m, bins_m = np.histogram(all_m, bins=20)
    print(f"  Distribution (20 bins):")
    max_count = max(hist_m)
    for i, (count, left, right) in enumerate(zip(hist_m, bins_m[:-1], bins_m[1:])):
        bar = '#' * int(40 * count / max_count)
        print(f"    [{left:+.2e}, {right:+.2e}): {bar}")

    # Analyze v distribution (raw)
    print("\n--- Second Moment (v) - Raw ---")
    print(f"  Range: [{all_v.min():.2e}, {all_v.max():.2e}]")
    print(f"  Dynamic range: {all_v.max() / max(all_v.min(), 1e-38):.2e}")
    print(f"  Mean:  {all_v.mean():.2e}")
    print(f"  Std:   {all_v.std():.2e}")

    # Analyze log(v) distribution
    print("\n--- Second Moment - log(v) ---")
    print(f"  Range: [{all_log_v.min():.1f}, {all_log_v.max():.1f}]")
    print(f"  Span:  {all_log_v.max() - all_log_v.min():.1f}")
    print(f"  Mean:  {all_log_v.mean():.1f}")
    print(f"  Std:   {all_log_v.std():.1f}")

    # Histogram of log(v)
    hist_logv, bins_logv = np.histogram(all_log_v, bins=20)
    print(f"  Distribution (20 bins):")
    max_count = max(hist_logv)
    for i, (count, left, right) in enumerate(zip(hist_logv, bins_logv[:-1], bins_logv[1:])):
        bar = '#' * int(40 * count / max_count)
        print(f"    [{left:+6.1f}, {right:+6.1f}): {bar}")

    # Uniformity analysis
    print("\n" + "=" * 70)
    print("UNIFORMITY ANALYSIS (for codebook design)")
    print("=" * 70)

    # Compute entropy as a measure of uniformity
    def compute_entropy(hist):
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    max_entropy = np.log2(20)  # Maximum for 20 bins

    entropy_m = compute_entropy(hist_m)
    entropy_logv = compute_entropy(hist_logv)

    print(f"\nEntropy (higher = more uniform, max = {max_entropy:.2f} for 20 bins):")
    print(f"  m:      {entropy_m:.2f} ({entropy_m/max_entropy*100:.0f}% of max)")
    print(f"  log(v): {entropy_logv:.2f} ({entropy_logv/max_entropy*100:.0f}% of max)")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if entropy_logv / max_entropy > 0.8:
        print("""
  log(v) is already quite uniform ({:.0f}% entropy).

  A custom codebook would provide MARGINAL gains:
  - Uniform quantization is near-optimal for uniform distributions
  - Custom codebook might squeeze out 5-10% precision improvement
  - Main benefit: could use 3 bits instead of 4 for same precision

  RECOMMENDATION: Log-space + Q4_0 is probably sufficient.
  Custom codebook worth it only if you need <1.5 bytes/param.
""".format(entropy_logv/max_entropy*100))
    else:
        print("""
  log(v) is non-uniform ({:.0f}% entropy).

  A custom codebook WOULD help significantly:
  - Can allocate more codewords to high-density regions
  - Potential 20-40% precision improvement

  RECOMMENDATION: Design iq3_xxs_opt codebook.
""".format(entropy_logv/max_entropy*100))

    # Suggest codebook design
    print("\n--- Suggested Codebook Design for iq3_xxs_opt ---")

    # For m: Find optimal quantization points
    percentiles_m = np.percentile(all_m, np.linspace(0, 100, 65))  # 64 codewords + endpoints
    print(f"\nm codeword suggestions (64 entries, percentile-based):")
    print(f"  Min: {percentiles_m[0]:.2e}, Max: {percentiles_m[-1]:.2e}")
    print(f"  Median entries span: [{percentiles_m[28]:.2e}, {percentiles_m[36]:.2e}]")

    # For log(v): Find optimal quantization points
    percentiles_logv = np.percentile(all_log_v, np.linspace(0, 100, 65))
    print(f"\nlog(v) codeword suggestions (64 entries, percentile-based):")
    print(f"  Min: {percentiles_logv[0]:.1f}, Max: {percentiles_logv[-1]:.1f}")
    print(f"  Median entries span: [{percentiles_logv[28]:.1f}, {percentiles_logv[36]:.1f}]")

    return all_m, all_v, all_log_v


if __name__ == "__main__":
    analyze_distributions()
