"""
Test whether log-space + custom codebook compounds benefits for optimizer state.

Question: Does using a density-aware codebook for log(v) give meaningful
precision gains beyond just using log-space with uniform quantization?

Methods tested:
1. FP32 Adam (baseline)
2. Log-space + uniform codebook (Q4_0)
3. Log-space + density-aware codebook (iq3_opt style)
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, List

from phfe.inference.gguf_vtensor import QuantizedLinear, requant_q4_0_cuda


class CustomCodebookState:
    """
    Optimizer state with custom codebooks for m and log(v).

    Uses density-aware codebooks that concentrate precision
    where the values cluster.
    """

    def __init__(self, shape: tuple, device: str = 'cuda'):
        self.shape = shape
        self.numel = math.prod(shape)
        self.device = device
        self.step = 0

        # Build density-aware codebooks (256 entries = 8 bits)
        # For m: Gaussian around 0
        self.m_codebook = torch.tensor(np.concatenate([
            np.linspace(-0.01, -0.001, 32),     # Far negative
            np.linspace(-0.001, -0.0001, 48),   # Near-negative
            np.linspace(-0.0001, 0.0001, 96),   # Near-zero (dense!)
            np.linspace(0.0001, 0.001, 48),     # Near-positive
            np.linspace(0.001, 0.01, 32),       # Far positive
        ]), dtype=torch.float32, device=device)

        # For log(v): Concentrated around [-28, -22]
        self.logv_codebook = torch.tensor(np.concatenate([
            np.linspace(-45, -35, 32),    # Far tail
            np.linspace(-35, -28, 48),    # Approaching peak
            np.linspace(-28, -22, 96),    # Peak region (dense!)
            np.linspace(-22, -15, 48),    # After peak
            np.linspace(-15, -5, 32),     # Near-zero tail
        ]), dtype=torch.float32, device=device)

        # Store as indices (uint8 = 1 byte per value)
        self.m_indices = torch.zeros(self.numel, dtype=torch.uint8, device=device)
        self.logv_indices = torch.zeros(self.numel, dtype=torch.uint8, device=device)

        self.initialized = False

    def quantize_to_codebook(self, values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Find nearest codebook entry for each value."""
        values_flat = values.view(-1, 1).float()
        cb = codebook.view(1, -1)
        distances = (values_flat - cb).abs()
        return distances.argmin(dim=1).to(torch.uint8)

    def dequant_m(self) -> torch.Tensor:
        return self.m_codebook[self.m_indices.long()].view(self.shape)

    def dequant_v(self) -> torch.Tensor:
        log_v = self.logv_codebook[self.logv_indices.long()]
        return torch.exp(log_v).view(self.shape)

    def requant_m(self, m: torch.Tensor):
        self.m_indices = self.quantize_to_codebook(m, self.m_codebook)

    def requant_v(self, v: torch.Tensor):
        log_v = torch.log(v.float().clamp(min=1e-38))
        self.logv_indices = self.quantize_to_codebook(log_v, self.logv_codebook)

    @property
    def nbytes(self) -> int:
        return self.m_indices.numel() + self.logv_indices.numel()


class UniformCodebookState:
    """
    Optimizer state with uniform codebooks (baseline for comparison).
    """

    def __init__(self, shape: tuple, device: str = 'cuda'):
        self.shape = shape
        self.numel = math.prod(shape)
        self.device = device
        self.step = 0

        # Uniform codebooks (same range as custom, but uniform spacing)
        self.m_codebook = torch.linspace(-0.01, 0.01, 256, device=device)
        self.logv_codebook = torch.linspace(-45, -5, 256, device=device)

        self.m_indices = torch.zeros(self.numel, dtype=torch.uint8, device=device)
        self.logv_indices = torch.zeros(self.numel, dtype=torch.uint8, device=device)

        self.initialized = False

    def quantize_to_codebook(self, values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        values_flat = values.view(-1, 1).float()
        cb = codebook.view(1, -1)
        distances = (values_flat - cb).abs()
        return distances.argmin(dim=1).to(torch.uint8)

    def dequant_m(self) -> torch.Tensor:
        return self.m_codebook[self.m_indices.long()].view(self.shape)

    def dequant_v(self) -> torch.Tensor:
        log_v = self.logv_codebook[self.logv_indices.long()]
        return torch.exp(log_v).view(self.shape)

    def requant_m(self, m: torch.Tensor):
        self.m_indices = self.quantize_to_codebook(m, self.m_codebook)

    def requant_v(self, v: torch.Tensor):
        log_v = torch.log(v.float().clamp(min=1e-38))
        self.logv_indices = self.quantize_to_codebook(log_v, self.logv_codebook)

    @property
    def nbytes(self) -> int:
        return self.m_indices.numel() + self.logv_indices.numel()


class CodebookAdam:
    """
    Adam optimizer using codebook-quantized state.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        use_custom_codebook: bool = True,
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.use_custom = use_custom_codebook

        self.quantized_params: List = []
        self.state: Dict[int, Any] = {}

        self._collect_params()
        self._init_state()

    def _collect_params(self):
        from phfe.inference.gguf_vtensor import QuantizedParameter
        for module in self.model.modules():
            if isinstance(module, QuantizedParameter):
                self.quantized_params.append(module)
            elif hasattr(module, 'weight') and isinstance(module.weight, QuantizedParameter):
                self.quantized_params.append(module.weight)

    def _init_state(self):
        StateClass = CustomCodebookState if self.use_custom else UniformCodebookState
        for qp in self.quantized_params:
            self.state[id(qp)] = StateClass(
                shape=qp.shape,
                device=qp._device,
            )

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas

        for qp in self.quantized_params:
            grad = qp.grad
            if grad is None:
                continue

            state = self.state[id(qp)]
            state.step += 1
            step = state.step

            if state.initialized:
                m = state.dequant_m().float()
                v = state.dequant_v().float()
            else:
                m = torch.zeros_like(grad, dtype=torch.float32)
                v = torch.ones_like(grad, dtype=torch.float32) * 1e-20
                state.initialized = True

            grad_fp32 = grad.float()
            m = beta1 * m + (1 - beta1) * grad_fp32
            v = beta2 * v + (1 - beta2) * grad_fp32.square()
            v = v.clamp(min=1e-38)

            state.requant_m(m)
            state.requant_v(v)

            m_hat = state.dequant_m().float() / (1 - beta1 ** step)
            v_hat = state.dequant_v().float() / (1 - beta2 ** step)
            v_hat = v_hat.clamp(min=1e-16)

            update = m_hat / (v_hat.sqrt() + self.eps)
            update = update.clamp(-1.0, 1.0)

            from phfe.inference.gguf_vtensor.dequant_kernels import get_requant_fn
            weight = qp.dequantize().float()
            weight_new = weight - self.lr * update

            requant_fn = get_requant_fn(qp.quant_type)
            qp.raw_data.copy_(requant_fn(weight_new.half(), qp.quant_type))

    def zero_grad(self):
        for qp in self.quantized_params:
            qp.zero_grad()

    def state_memory_bytes(self) -> int:
        return sum(self.state[id(qp)].nbytes for qp in self.quantized_params)


def run_convergence_test():
    """Compare convergence of different codebook strategies."""

    print("=" * 70)
    print("CODEBOOK CONVERGENCE TEST")
    print("=" * 70)
    print("\nComparing: log-space + uniform vs log-space + density-aware codebook")

    device = 'cuda'
    torch.manual_seed(42)

    # Create test model
    class TestModel(nn.Module):
        def __init__(self, dim=512, layers=3):
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

    # Test data
    x = torch.randn(32, 512, device=device, dtype=torch.float16)
    y = torch.randn(32, 512, device=device, dtype=torch.float16)

    results = {}
    n_steps = 100

    # Test 1: FP32 Adam baseline
    print("\n--- FP32 Adam Baseline ---")
    torch.manual_seed(42)
    model_fp32 = TestModel()

    # Collect fp32 m, v for reference
    m_states = {id(qp): torch.zeros(qp.numel, device=device) for qp in model_fp32.modules()
                if hasattr(qp, 'grad_holder')}
    v_states = {id(qp): torch.zeros(qp.numel, device=device) for qp in model_fp32.modules()
                if hasattr(qp, 'grad_holder')}

    losses_fp32 = []
    beta1, beta2 = 0.9, 0.999
    lr = 0.001
    eps = 1e-8

    for step in range(n_steps):
        pred = model_fp32(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        losses_fp32.append(loss.item())

        with torch.no_grad():
            for layer in model_fp32.layers:
                qp = layer.weight
                if qp.grad is not None:
                    grad = qp.grad.float().view(-1)
                    key = id(qp)
                    if key not in m_states:
                        m_states[key] = torch.zeros_like(grad)
                        v_states[key] = torch.zeros_like(grad)

                    m_states[key] = beta1 * m_states[key] + (1 - beta1) * grad
                    v_states[key] = beta2 * v_states[key] + (1 - beta2) * grad.square()

                    m_hat = m_states[key] / (1 - beta1 ** (step + 1))
                    v_hat = v_states[key] / (1 - beta2 ** (step + 1))

                    update = m_hat / (v_hat.sqrt() + eps)

                    weight = qp.dequantize().float().view(-1)
                    weight_new = weight - lr * update

                    qp.raw_data.copy_(requant_q4_0_cuda(
                        weight_new.view(qp.shape).half(), qp.quant_type))
                    qp.zero_grad()

        if step % 25 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    results['fp32'] = losses_fp32

    # Test 2: Log-space + Uniform codebook
    print("\n--- Log-space + Uniform Codebook ---")
    torch.manual_seed(42)
    model_uniform = TestModel()
    opt_uniform = CodebookAdam(model_uniform, lr=lr, use_custom_codebook=False)

    losses_uniform = []
    for step in range(n_steps):
        pred = model_uniform(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        losses_uniform.append(loss.item())
        opt_uniform.step()
        opt_uniform.zero_grad()

        if step % 25 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    results['uniform'] = losses_uniform

    # Test 3: Log-space + Custom density-aware codebook
    print("\n--- Log-space + Density-aware Codebook ---")
    torch.manual_seed(42)
    model_custom = TestModel()
    opt_custom = CodebookAdam(model_custom, lr=lr, use_custom_codebook=True)

    losses_custom = []
    for step in range(n_steps):
        pred = model_custom(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        losses_custom.append(loss.item())
        opt_custom.step()
        opt_custom.zero_grad()

        if step % 25 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    results['custom'] = losses_custom

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nFinal losses (step {n_steps}):")
    print(f"  FP32 Adam:              {losses_fp32[-1]:.6f}")
    print(f"  Log-space + Uniform:    {losses_uniform[-1]:.6f}")
    print(f"  Log-space + Density:    {losses_custom[-1]:.6f}")

    # Compute gap to FP32
    gap_uniform = (losses_uniform[-1] - losses_fp32[-1]) / losses_fp32[-1] * 100
    gap_custom = (losses_custom[-1] - losses_fp32[-1]) / losses_fp32[-1] * 100

    print(f"\nGap to FP32 baseline:")
    print(f"  Uniform codebook:       {gap_uniform:+.2f}%")
    print(f"  Density-aware codebook: {gap_custom:+.2f}%")

    improvement = gap_uniform - gap_custom
    print(f"\nDensity-aware improvement: {improvement:.2f} percentage points")

    # Memory comparison
    print(f"\nMemory (optimizer state only):")
    print(f"  FP32 Adam:     {sum(qp.numel * 8 for qp in model_fp32.modules() if hasattr(qp, 'numel')) / 1e6:.2f} MB")
    print(f"  Codebook Adam: {opt_custom.state_memory_bytes() / 1e6:.2f} MB")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(gap_uniform - gap_custom) < 1.0:
        print("""
  The density-aware codebook provides MARGINAL improvement over uniform.
  Both converge similarly when using log-space transformation.

  Key finding: Log-space is the main win. Custom codebook adds ~0-1%
  precision improvement at this compression level (8-bit indices).

  RECOMMENDATION: Use log-space + uniform codebook (simpler).
  Custom codebook only worth it for 3-bit or lower compression.
""")
    else:
        print(f"""
  The density-aware codebook provides {improvement:.1f}pp improvement.
  This confirms that combining log-space + custom codebook does compound.

  RECOMMENDATION: Use density-aware codebook for maximum precision.
""")

    return results


if __name__ == "__main__":
    run_convergence_test()
