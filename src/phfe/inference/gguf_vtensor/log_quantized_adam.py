"""
Log-Space Quantized Adam Optimizer

The key insight: v (second moment) spans many orders of magnitude,
breaking direct quantization. Solution: store log(v) instead.

Dynamic range analysis:
  - Gradients: ~1e-6 to 1e-1 (varies by layer)
  - v = EMA(grad²): ~1e-12 to 1e-2
  - log(v): ~-28 to -5 (much more uniform!)

By storing log(v), we can use low-bit quantization (even IQ3_XXS)
without losing the ability to represent tiny v values.

Memory comparison (per parameter):
  FP16 + AdamW:     2B + 8B = 10 bytes/param
  Q4 + Q8 opt:      0.5B + 2B = 2.5 bytes/param (our previous approach)
  Q4 + IQ3 log opt: 0.5B + 0.8B = 1.3 bytes/param (this approach!)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import math

from .vtensor import QuantizedParameter
from .dequant_kernels import (
    dequant_q4_0_cuda, requant_q4_0_cuda,
    dequant_iq3_xxs_cuda, requant_iq3_xxs_cuda,
)


class LogQuantizedState:
    """
    Optimizer state with log-space quantization for v.

    m is stored directly (signed, moderate dynamic range).
    v is stored as log(v) (always positive, huge dynamic range → compressed).
    """

    def __init__(
        self,
        shape: tuple,
        device: str = 'cuda',
        m_quant_type: str = 'q4_0',   # For first moment (signed)
        v_quant_type: str = 'q4_0',   # For log(second moment)
    ):
        self.shape = shape
        self.numel = math.prod(shape)
        self.device = device
        self.m_quant_type = m_quant_type
        self.v_quant_type = v_quant_type
        self.step = 0

        # Log-space parameters for v
        # v typically ranges from 1e-30 to 1e+2
        # log(v) ranges from -69 to 4.6
        # We'll scale to fit in quantization range
        self.log_v_min = -40.0  # log(1e-17) - reasonable floor
        self.log_v_max = 10.0   # log(22000) - reasonable ceiling
        self.log_v_scale = self.log_v_max - self.log_v_min

        # Calculate storage sizes
        if m_quant_type == 'q4_0':
            m_bytes_per_block = 18
            block_size = 32
        elif m_quant_type == 'iq3_xxs':
            m_bytes_per_block = 68  # Approximate for 256-element superblock
            block_size = 256
        else:
            raise ValueError(f"Unsupported m_quant_type: {m_quant_type}")

        n_blocks_m = (self.numel + block_size - 1) // block_size

        if v_quant_type == 'q4_0':
            v_bytes_per_block = 18
            v_block_size = 32
        elif v_quant_type == 'iq3_xxs':
            v_bytes_per_block = 68
            v_block_size = 256
        else:
            raise ValueError(f"Unsupported v_quant_type: {v_quant_type}")

        n_blocks_v = (self.numel + v_block_size - 1) // v_block_size

        # Allocate quantized storage
        self.m_quant = torch.zeros(n_blocks_m * m_bytes_per_block,
                                    dtype=torch.uint8, device=device)
        self.log_v_quant = torch.zeros(n_blocks_v * v_bytes_per_block,
                                        dtype=torch.uint8, device=device)

        self.initialized = False
        self._m_block_size = block_size
        self._v_block_size = v_block_size

    def _get_dequant_fn(self, quant_type):
        if quant_type == 'q4_0':
            return dequant_q4_0_cuda
        elif quant_type == 'iq3_xxs':
            return dequant_iq3_xxs_cuda
        raise ValueError(f"Unknown quant type: {quant_type}")

    def _get_requant_fn(self, quant_type):
        if quant_type == 'q4_0':
            return requant_q4_0_cuda
        elif quant_type == 'iq3_xxs':
            return requant_iq3_xxs_cuda
        raise ValueError(f"Unknown quant type: {quant_type}")

    def dequant_m(self) -> torch.Tensor:
        """Dequantize first moment."""
        dequant_fn = self._get_dequant_fn(self.m_quant_type)
        return dequant_fn(self.m_quant, self.numel, self.device).view(self.shape)

    def dequant_v(self) -> torch.Tensor:
        """Dequantize second moment (stored in log space)."""
        dequant_fn = self._get_dequant_fn(self.v_quant_type)

        # Dequantize log(v)
        log_v_normalized = dequant_fn(self.log_v_quant, self.numel, self.device)

        # Denormalize: [0, 1] → [log_v_min, log_v_max]
        log_v = log_v_normalized.float() * self.log_v_scale + self.log_v_min

        # Exponentiate to get v
        v = torch.exp(log_v).view(self.shape)

        return v

    def requant_m(self, m: torch.Tensor):
        """Requantize first moment."""
        requant_fn = self._get_requant_fn(self.m_quant_type)
        self.m_quant = requant_fn(m.view(-1), self.m_quant_type)

    def requant_v(self, v: torch.Tensor):
        """Requantize second moment in log space."""
        requant_fn = self._get_requant_fn(self.v_quant_type)

        # Convert to log space
        v_clamped = v.float().clamp(min=1e-38)  # Avoid log(0)
        log_v = torch.log(v_clamped)

        # Normalize to [0, 1] range for quantization
        log_v_normalized = (log_v - self.log_v_min) / self.log_v_scale
        log_v_normalized = log_v_normalized.clamp(0, 1)  # Ensure in range

        # Quantize
        self.log_v_quant = requant_fn(log_v_normalized.view(-1).half(), self.v_quant_type)

    @property
    def nbytes(self) -> int:
        return self.m_quant.numel() + self.log_v_quant.numel()


class LogQuantizedAdam:
    """
    Adam with log-space quantization for optimizer state.

    Achieves ~1.3 bytes/param by:
    - Weights: Q4_0 (0.5 bytes/param)
    - m: Q4_0 (0.5 bytes/param)
    - v: Q4_0 in log space (0.5 bytes/param)

    The log-space trick handles v's huge dynamic range:
    - v = EMA(grad²) ranges from 1e-30 to 1e+2
    - log(v) ranges from -69 to 4.6 (easily quantizable!)

    Usage:
        optimizer = LogQuantizedAdam(model, lr=1e-4)

        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_quant: str = 'q4_0',
        state_quant: str = 'q4_0',  # Both m and log(v) use this
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_quant = weight_quant
        self.state_quant = state_quant

        self.quantized_params: List[QuantizedParameter] = []
        self.regular_params: List[nn.Parameter] = []
        self.state: Dict[int, Any] = {}

        self._collect_params()
        self._init_state()

    def _collect_params(self):
        """Collect quantized and regular parameters."""
        for module in self.model.modules():
            if isinstance(module, QuantizedParameter):
                self.quantized_params.append(module)
            elif hasattr(module, 'weight') and isinstance(module.weight, QuantizedParameter):
                self.quantized_params.append(module.weight)

        for param in self.model.parameters():
            if param.requires_grad:
                self.regular_params.append(param)

    def _init_state(self):
        """Initialize optimizer state."""
        for qp in self.quantized_params:
            self.state[id(qp)] = LogQuantizedState(
                shape=qp.shape,
                device=qp._device,
                m_quant_type=self.state_quant,
                v_quant_type=self.state_quant,
            )

        for p in self.regular_params:
            self.state[id(p)] = {
                'step': 0,
                'm': torch.zeros_like(p),
                'v': torch.zeros_like(p),
            }

    @torch.no_grad()
    def step(self):
        """Perform optimization step."""
        beta1, beta2 = self.betas

        for qp in self.quantized_params:
            self._step_quantized(qp, beta1, beta2)

        for p in self.regular_params:
            if p.grad is not None:
                self._step_regular(p, beta1, beta2)

    def _step_quantized(self, qp: QuantizedParameter, beta1: float, beta2: float):
        """Update quantized parameter with log-quantized optimizer state."""
        grad = qp.grad
        if grad is None:
            return

        state = self.state[id(qp)]
        state.step += 1
        step = state.step

        # Dequantize m, v
        if state.initialized:
            m = state.dequant_m().float()
            v = state.dequant_v().float()
        else:
            m = torch.zeros_like(grad, dtype=torch.float32)
            v = torch.ones_like(grad, dtype=torch.float32) * 1e-8  # Small positive init
            state.initialized = True

        # Update m, v
        grad_fp32 = grad.float()
        m = beta1 * m + (1 - beta1) * grad_fp32
        v = beta2 * v + (1 - beta2) * grad_fp32.square()

        # Clamp v to valid range for log
        v = v.clamp(min=1e-38, max=1e10)

        # Requantize
        state.requant_m(m)
        state.requant_v(v)

        # Bias correction (use fresh dequant for better precision)
        m_hat = state.dequant_m().float() / (1 - beta1 ** step)
        v_hat = state.dequant_v().float() / (1 - beta2 ** step)

        # Compute update
        v_hat = v_hat.clamp(min=1e-16)  # Safety
        update = m_hat / (v_hat.sqrt() + self.eps)
        update = update.clamp(-1.0, 1.0)  # Prevent explosion

        # Apply to weight
        from .dequant_kernels import get_requant_fn
        weight = qp.dequantize().float()
        weight_new = weight - self.lr * update

        requant_fn = get_requant_fn(qp.quant_type)
        qp.raw_data.copy_(requant_fn(weight_new.half(), qp.quant_type))

    def _step_regular(self, p: nn.Parameter, beta1: float, beta2: float):
        """Standard Adam for regular parameters."""
        state = self.state[id(p)]
        state['step'] += 1
        step = state['step']

        m, v = state['m'], state['v']
        grad = p.grad

        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        p.data.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients."""
        for qp in self.quantized_params:
            qp.zero_grad()

        for p in self.regular_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_memory_bytes(self) -> int:
        """Calculate total optimizer state memory."""
        total = 0

        for qp in self.quantized_params:
            state = self.state[id(qp)]
            total += state.nbytes

        for p in self.regular_params:
            state = self.state[id(p)]
            total += state['m'].numel() * state['m'].element_size()
            total += state['v'].numel() * state['v'].element_size()

        return total

    def memory_report(self) -> Dict[str, float]:
        """Get detailed memory breakdown."""
        quant_weight_bytes = sum(qp.nbytes_quantized for qp in self.quantized_params)
        quant_state_bytes = sum(self.state[id(qp)].nbytes for qp in self.quantized_params)

        regular_weight_bytes = sum(p.numel() * p.element_size() for p in self.regular_params)
        regular_state_bytes = sum(
            self.state[id(p)]['m'].numel() * 4 * 2
            for p in self.regular_params
        )

        total_params = sum(qp.numel for qp in self.quantized_params) + \
                       sum(p.numel() for p in self.regular_params)

        return {
            'quantized_weights_gb': quant_weight_bytes / 1e9,
            'quantized_state_gb': quant_state_bytes / 1e9,
            'regular_weights_gb': regular_weight_bytes / 1e9,
            'regular_state_gb': regular_state_bytes / 1e9,
            'total_gb': (quant_weight_bytes + quant_state_bytes +
                        regular_weight_bytes + regular_state_bytes) / 1e9,
            'bytes_per_param': (quant_weight_bytes + quant_state_bytes +
                               regular_weight_bytes + regular_state_bytes) / total_params,
        }


def test_log_quantized_adam():
    """Test the log-quantized Adam optimizer."""
    from .vtensor import QuantizedLinear

    print("=" * 60)
    print("TEST: Log-Quantized Adam Optimizer")
    print("=" * 60)

    device = 'cuda'
    torch.manual_seed(42)

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self, dim=256):
            super().__init__()

            w1 = torch.randn(dim, dim, device=device) * 0.02
            w2 = torch.randn(dim, dim, device=device) * 0.02

            w1_q = requant_q4_0_cuda(w1, 'q4_0')
            w2_q = requant_q4_0_cuda(w2, 'q4_0')

            self.fc1 = QuantizedLinear(dim, dim, w1_q, 'q4_0', device=device)
            self.fc2 = QuantizedLinear(dim, dim, w2_q, 'q4_0', device=device)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleModel()

    # Create optimizer
    optimizer = LogQuantizedAdam(
        model,
        lr=0.01,
        state_quant='q4_0',  # Both m and log(v) in Q4_0
    )

    # Memory report
    report = optimizer.memory_report()
    print(f"\nMemory breakdown:")
    print(f"  Quantized weights: {report['quantized_weights_gb']*1000:.2f} MB")
    print(f"  Quantized state:   {report['quantized_state_gb']*1000:.2f} MB")
    print(f"  Total:             {report['total_gb']*1000:.2f} MB")
    print(f"  Bytes per param:   {report['bytes_per_param']:.2f}")

    # Compare
    fp16_bpp = 10.0
    savings = fp16_bpp / report['bytes_per_param']
    print(f"  vs FP16+AdamW:     {savings:.1f}x savings")

    # Training test
    print(f"\nTraining test...")
    x = torch.randn(8, 256, device=device, dtype=torch.float16)
    y = torch.randn(8, 256, device=device, dtype=torch.float16)

    losses = []
    for step in range(20):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    if losses[-1] < losses[0]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\nLoss improved by {improvement:.1f}%")
        print("Log-Quantized Adam: ✓")
    else:
        print("\nWARNING: Loss did not improve")

    return optimizer


if __name__ == "__main__":
    test_log_quantized_adam()
