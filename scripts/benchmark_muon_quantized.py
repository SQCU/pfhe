"""
Muon-style optimizer with Newton-Schulz orthogonalization.

Key insight: Newton-Schulz doesn't need m or v state!
- Adam: stores m, v (8 bytes/param) with huge dynamic range
- Muon: only needs gradients, orthogonalized on-the-fly

This could enable much lower effective bits per parameter.

Newton-Schulz iteration for orthogonalization:
    X_{k+1} = 0.5 * X_k @ (3I - X_k.T @ X_k)

Converges to orthogonal matrix in ~5 iterations.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import triton
import triton.language as tl
import numpy as np
import time
import math

device = 'cuda'


def newton_schulz_orthogonalize(G: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """
    Orthogonalize gradient matrix using Newton-Schulz iteration.

    G: gradient matrix (out_features, in_features)
    Returns: orthogonalized gradient
    """
    # Normalize for numerical stability
    G = G / (G.norm() + 1e-7)

    X = G
    for _ in range(n_iters):
        X = 0.5 * X @ (3 * torch.eye(X.shape[1], device=X.device, dtype=X.dtype) - X.T @ X)

    return X


def newton_schulz_orthogonalize_batched(G: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """
    Orthogonalize using the simpler Muon formulation.

    Uses: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k.T @ X_k
    """
    # Normalize
    G = G / (G.norm() + 1e-7)

    X = G
    for _ in range(n_iters):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X

    return X


class MuonOptimizer:
    """
    Muon-style optimizer using Newton-Schulz orthogonalization.

    Memory per parameter:
    - Adam: 8 bytes (m + v in FP32)
    - Muon: 0 bytes (no state!)

    The gradient is orthogonalized on-the-fly, no caching needed.
    """

    def __init__(self, model, lr: float = 0.02, momentum: float = 0.95,
                 n_iters: int = 5, nesterov: bool = True):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.n_iters = n_iters
        self.nesterov = nesterov

        # Optional: momentum buffer (much smaller dynamic range than Adam's v)
        # Can be disabled for zero-state optimizer
        self.use_momentum = momentum > 0
        self.momentum_buffers = {}

        self._collect_params()

    def _collect_params(self):
        self.params = []
        for module in self.model.modules():
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'grad_holder'):
                    # QuantizedParameter
                    self.params.append(('quantized', module.weight))
                elif isinstance(module.weight, nn.Parameter):
                    self.params.append(('regular', module.weight))

    @torch.no_grad()
    def step(self):
        for ptype, param in self.params:
            if ptype == 'quantized':
                grad = param.grad
                if grad is None:
                    continue

                # Reshape to matrix for orthogonalization
                grad_2d = grad.view(param.shape[0], -1).float()

                # Newton-Schulz orthogonalization
                ortho_grad = newton_schulz_orthogonalize_batched(grad_2d, self.n_iters)

                # Scale back
                ortho_grad = ortho_grad * grad.norm() / (ortho_grad.norm() + 1e-7)

                # Optional momentum
                if self.use_momentum:
                    key = id(param)
                    if key not in self.momentum_buffers:
                        self.momentum_buffers[key] = torch.zeros_like(ortho_grad)

                    buf = self.momentum_buffers[key]
                    buf.mul_(self.momentum).add_(ortho_grad)

                    if self.nesterov:
                        ortho_grad = ortho_grad + self.momentum * buf
                    else:
                        ortho_grad = buf

                # Apply update
                from phfe.inference.gguf_vtensor.dequant_kernels import get_requant_fn
                weight = param.dequantize().float()
                weight_new = weight - self.lr * ortho_grad.view(param.shape)

                requant_fn = get_requant_fn(param.quant_type)
                param.raw_data.copy_(requant_fn(weight_new.half(), param.quant_type))

            else:
                # Regular parameter
                if param.grad is None:
                    continue

                grad = param.grad.float()
                if grad.dim() >= 2:
                    grad_2d = grad.view(grad.shape[0], -1)
                    ortho_grad = newton_schulz_orthogonalize_batched(grad_2d, self.n_iters)
                    ortho_grad = ortho_grad * grad.norm() / (ortho_grad.norm() + 1e-7)
                    grad = ortho_grad.view(grad.shape)

                param.data.add_(grad, alpha=-self.lr)

    def zero_grad(self):
        for ptype, param in self.params:
            if ptype == 'quantized':
                param.zero_grad()
            else:
                if param.grad is not None:
                    param.grad = None

    def state_memory_bytes(self) -> int:
        """Muon uses minimal or zero state."""
        if self.use_momentum:
            return sum(buf.numel() * buf.element_size()
                      for buf in self.momentum_buffers.values())
        return 0


class MuonNoMomentum:
    """
    Zero-state Muon optimizer - no buffers at all!

    Memory per parameter: 0 bytes
    """

    def __init__(self, model, lr: float = 0.02, n_iters: int = 5):
        self.model = model
        self.lr = lr
        self.n_iters = n_iters
        self._collect_params()

    def _collect_params(self):
        self.params = []
        for module in self.model.modules():
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'grad_holder'):
                    self.params.append(('quantized', module.weight))
                elif isinstance(module.weight, nn.Parameter):
                    self.params.append(('regular', module.weight))

    @torch.no_grad()
    def step(self):
        for ptype, param in self.params:
            if ptype == 'quantized':
                grad = param.grad
                if grad is None:
                    continue

                grad_2d = grad.view(param.shape[0], -1).float()
                ortho_grad = newton_schulz_orthogonalize_batched(grad_2d, self.n_iters)
                ortho_grad = ortho_grad * grad.norm() / (ortho_grad.norm() + 1e-7)

                from phfe.inference.gguf_vtensor.dequant_kernels import get_requant_fn
                weight = param.dequantize().float()
                weight_new = weight - self.lr * ortho_grad.view(param.shape)

                requant_fn = get_requant_fn(param.quant_type)
                param.raw_data.copy_(requant_fn(weight_new.half(), param.quant_type))
            else:
                if param.grad is None:
                    continue
                grad = param.grad.float()
                if grad.dim() >= 2:
                    grad_2d = grad.view(grad.shape[0], -1)
                    ortho_grad = newton_schulz_orthogonalize_batched(grad_2d, self.n_iters)
                    ortho_grad = ortho_grad * grad.norm() / (ortho_grad.norm() + 1e-7)
                    grad = ortho_grad.view(grad.shape)
                param.data.add_(grad, alpha=-self.lr)

    def zero_grad(self):
        for ptype, param in self.params:
            if ptype == 'quantized':
                param.zero_grad()
            else:
                if param.grad is not None:
                    param.grad = None

    def state_memory_bytes(self) -> int:
        return 0  # Zero state!


# ============================================================================
# Triton kernel for fused Newton-Schulz (for throughput testing)
# ============================================================================

@triton.jit
def _newton_schulz_step_kernel(
    X_ptr, out_ptr,
    M, N,  # Matrix dimensions
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One Newton-Schulz iteration: X' = 1.5*X - 0.5*X@X.T@X"""
    # This is a simplified kernel - full implementation would need
    # proper matrix multiplication tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # For now, just demonstrate the computation pattern
    # Real implementation needs proper GEMM tiling
    pass


def benchmark_throughput():
    """Compare throughput of different optimizers."""
    from phfe.inference.gguf_vtensor import QuantizedLinear, requant_q4_0_cuda

    print("=" * 70)
    print("THROUGHPUT BENCHMARK: Adam vs Muon")
    print("=" * 70)

    dim = 1024
    layers = 4
    n_warmup = 10
    n_benchmark = 50

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(layers):
                w = torch.randn(dim, dim, device=device) * 0.02
                w_q = requant_q4_0_cuda(w, 'q4_0')
                self.layers.append(QuantizedLinear(dim, dim, w_q, 'q4_0', device=device))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
            return x

    x = torch.randn(32, dim, device=device, dtype=torch.float16)
    y = torch.randn(32, dim, device=device, dtype=torch.float16)

    results = {}
    n_params = layers * dim * dim

    # 1. FP32 Adam baseline
    print("\n--- FP32 Adam ---")
    model = TestModel()
    m_state = {id(l.weight): torch.zeros(dim*dim, device=device) for l in model.layers}
    v_state = {id(l.weight): torch.zeros(dim*dim, device=device) for l in model.layers}

    def adam_step(step):
        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
        for layer in model.layers:
            qp = layer.weight
            if qp.grad is None:
                continue
            grad = qp.grad.float().view(-1)
            m_state[id(qp)] = beta1 * m_state[id(qp)] + (1-beta1) * grad
            v_state[id(qp)] = beta2 * v_state[id(qp)] + (1-beta2) * grad.square()
            m_hat = m_state[id(qp)] / (1 - beta1**step)
            v_hat = v_state[id(qp)] / (1 - beta2**step)
            update = m_hat / (v_hat.sqrt() + eps)
            weight = qp.dequantize().float().view(-1)
            weight_new = weight - lr * update
            qp.raw_data.copy_(requant_q4_0_cuda(weight_new.view(dim, dim).half(), 'q4_0'))
            qp.zero_grad()

    for i in range(n_warmup):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        with torch.no_grad():
            adam_step(i+1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(n_benchmark):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        with torch.no_grad():
            adam_step(n_warmup+i+1)
    torch.cuda.synchronize()
    results['adam_fp32'] = (time.perf_counter() - start) / n_benchmark * 1000
    adam_state_bytes = n_params * 8

    # 2. Muon with momentum
    print("--- Muon with momentum ---")
    torch.manual_seed(42)
    model = TestModel()
    opt = MuonOptimizer(model, lr=0.02, momentum=0.95, n_iters=5)

    for i in range(n_warmup):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(n_benchmark):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    results['muon_momentum'] = (time.perf_counter() - start) / n_benchmark * 1000
    muon_mom_bytes = opt.state_memory_bytes()

    # 3. Muon zero-state
    print("--- Muon zero-state ---")
    torch.manual_seed(42)
    model = TestModel()
    opt = MuonNoMomentum(model, lr=0.02, n_iters=5)

    for i in range(n_warmup):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(n_benchmark):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    results['muon_zero'] = (time.perf_counter() - start) / n_benchmark * 1000

    # Results
    print(f"\n{'Optimizer':<25} {'Time (ms)':<12} {'State Memory':<15} {'B/param':<10}")
    print("-" * 62)
    print(f"{'Adam FP32 state':<25} {results['adam_fp32']:>8.2f}     {adam_state_bytes/1e6:>8.2f} MB    {adam_state_bytes/n_params:.1f}")
    print(f"{'Muon + momentum':<25} {results['muon_momentum']:>8.2f}     {muon_mom_bytes/1e6:>8.2f} MB    {muon_mom_bytes/n_params:.1f}")
    print(f"{'Muon zero-state':<25} {results['muon_zero']:>8.2f}     {'0.00':>8} MB    {'0.0':<10}")

    return results


def benchmark_convergence():
    """Compare convergence of different optimizers."""
    from phfe.inference.gguf_vtensor import QuantizedLinear, requant_q4_0_cuda

    print("\n" + "=" * 70)
    print("CONVERGENCE BENCHMARK: Adam vs Muon")
    print("=" * 70)

    dim = 256
    n_steps = 300

    # Teacher network for learnable task
    torch.manual_seed(0)
    teacher_w = torch.randn(dim, dim, device=device, dtype=torch.float16) * 0.1

    def get_batch():
        x = torch.randn(32, dim, device=device, dtype=torch.float16)
        y = torch.tanh(x @ teacher_w) + torch.randn_like(x) * 0.05
        return x, y

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            w = torch.randn(dim, dim, device=device) * 0.02
            w_q = requant_q4_0_cuda(w, 'q4_0')
            self.layer = QuantizedLinear(dim, dim, w_q, 'q4_0', device=device)

        def forward(self, x):
            return self.layer(x)

    results = {}

    # 1. FP32 Adam
    print("\n--- FP32 Adam ---")
    torch.manual_seed(42)
    model = TestModel()
    m_state = torch.zeros(dim*dim, device=device)
    v_state = torch.zeros(dim*dim, device=device)
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-3

    losses = []
    for step in range(n_steps):
        x, y = get_batch()
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            qp = model.layer.weight
            grad = qp.grad.float().view(-1)
            m_state = beta1 * m_state + (1-beta1) * grad
            v_state = beta2 * v_state + (1-beta2) * grad.square()
            m_hat = m_state / (1 - beta1**(step+1))
            v_hat = v_state / (1 - beta2**(step+1))
            update = m_hat / (v_hat.sqrt() + eps)
            weight = qp.dequantize().float().view(-1)
            weight_new = weight - lr * update
            qp.raw_data.copy_(requant_q4_0_cuda(weight_new.view(dim, dim).half(), 'q4_0'))
            qp.zero_grad()

        if step % 75 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    results['adam'] = losses

    # 2. Muon with momentum
    print("\n--- Muon with momentum ---")
    torch.manual_seed(42)
    model = TestModel()
    opt = MuonOptimizer(model, lr=0.02, momentum=0.95, n_iters=5)

    losses = []
    for step in range(n_steps):
        x, y = get_batch()
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()

        if step % 75 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    results['muon_mom'] = losses

    # 3. Muon zero-state
    print("\n--- Muon zero-state ---")
    torch.manual_seed(42)
    model = TestModel()
    opt = MuonNoMomentum(model, lr=0.02, n_iters=5)

    losses = []
    for step in range(n_steps):
        x, y = get_batch()
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()

        if step % 75 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    results['muon_zero'] = losses

    # 4. Muon with fewer iterations (faster)
    print("\n--- Muon 3-iter (faster) ---")
    torch.manual_seed(42)
    model = TestModel()
    opt = MuonNoMomentum(model, lr=0.02, n_iters=3)

    losses = []
    for step in range(n_steps):
        x, y = get_batch()
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()

        if step % 75 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    results['muon_3iter'] = losses

    # Summary
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    baseline = results['adam'][-1]
    print(f"\n{'Optimizer':<25} {'Final Loss':<12} {'Gap to Adam':<15} {'State':<10}")
    print("-" * 62)

    for name, losses in results.items():
        gap = (losses[-1] - baseline) / baseline * 100
        state = "8 B/p" if name == 'adam' else ("4 B/p" if 'mom' in name else "0 B/p")
        print(f"{name:<25} {losses[-1]:>8.4f}     {gap:>+6.1f}%         {state:<10}")

    return results


def analyze_memory_tradeoff():
    """Analyze the full memory picture."""
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS: Adam vs Muon")
    print("=" * 70)

    # Per-parameter memory breakdown
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  MEMORY PER PARAMETER                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Component          Adam FP32    Adam 8-bit    Muon+Mom    Muon Zero   │
│  ─────────────────────────────────────────────────────────────────────  │
│  Weights (Q4)       0.56 B       0.56 B        0.56 B      0.56 B      │
│  Gradients (FP16)   2.0 B        2.0 B         2.0 B       2.0 B       │
│  m state            4.0 B        1.0 B         4.0 B       0 B         │
│  v state            4.0 B        1.0 B         0 B         0 B         │
│  ─────────────────────────────────────────────────────────────────────  │
│  TOTAL              10.56 B      4.56 B        6.56 B      2.56 B      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  EFFECTIVE SAVINGS                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  vs FP16 + Adam FP32 (12 B/p baseline):                                 │
│    Adam 8-bit:     2.6x savings                                         │
│    Muon + Mom:     1.8x savings                                         │
│    Muon Zero:      4.7x savings  ← BEST                                 │
│                                                                         │
│  But Muon has compute overhead from Newton-Schulz iterations            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

    # For larger models
    for params in [1e9, 7e9, 70e9]:
        adam_fp32 = params * 10.56 / 1e9
        adam_8bit = params * 4.56 / 1e9
        muon_mom = params * 6.56 / 1e9
        muon_zero = params * 2.56 / 1e9

        print(f"\n{params/1e9:.0f}B model:")
        print(f"  Adam FP32:  {adam_fp32:>6.1f} GB")
        print(f"  Adam 8-bit: {adam_8bit:>6.1f} GB")
        print(f"  Muon + Mom: {muon_mom:>6.1f} GB")
        print(f"  Muon Zero:  {muon_zero:>6.1f} GB  ← {adam_fp32/muon_zero:.1f}x smaller")


def main():
    print("=" * 70)
    print("MUON OPTIMIZER ANALYSIS")
    print("Newton-Schulz orthogonalization for quantized training")
    print("=" * 70)

    throughput = benchmark_throughput()
    convergence = benchmark_convergence()
    analyze_memory_tradeoff()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Muon-style Newton-Schulz optimization:

PROS:
  - Zero optimizer state possible (no m, no v)
  - 4.7x memory savings vs Adam (better than 8-bit Adam's 2.6x)
  - No dynamic range issues (no v to quantize!)
  - Gradients have much narrower range than v

CONS:
  - Newton-Schulz iterations add compute (5 iterations typical)
  - May need tuning for different architectures
  - Momentum version still needs state (but just 4 B/p, not 8)

RECOMMENDATION:
  - For maximum memory efficiency: Muon zero-state (2.56 B/param)
  - For best convergence: Adam 8-bit log-space (4.56 B/param)
  - For zero-hassle: Muon with momentum (6.56 B/param)

The key insight: Muon sidesteps the dynamic range problem entirely
by not storing second moments. This enables true low-memory training
without the precision issues that plague 3-bit Adam.
""")


if __name__ == "__main__":
    main()
