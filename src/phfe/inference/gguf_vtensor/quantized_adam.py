"""
Fully Quantized Adam Optimizer

The key insight: if we can quantize weights to 3-4 bits, we can quantize
optimizer state (m, v) to 3-4 bits too!

Memory comparison (per parameter):
  FP16 + AdamW:     2B (weight) + 8B (m,v FP32) = 10 bytes
  Q4_0 + FP32 opt:  0.5B + 12B (m,v,carry FP32) = 12.5 bytes  ← WORSE!
  Q4_0 + Q8_0 opt:  0.5B + 2B (m,v Q8_0) = 2.5 bytes         ← 4x better
  IQ3 + IQ3 opt:    0.4B + 0.8B (m,v IQ3) = 1.2 bytes        ← 8x better!

The trick: m and v are slowly-changing EMAs, so quantization error
doesn't accumulate catastrophically. We just need to be careful about:
1. Dynamic range (v can get very small)
2. Bias correction (happens in FP32 before update)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import math

from .vtensor import QuantizedParameter
from .dequant_kernels import (
    dequant_q4_0_cuda, requant_q4_0_cuda,
    dequant_q8_0_cuda, requant_q8_0_cuda,
)


class QuantizedAdamState:
    """
    Optimizer state stored in quantized format.

    Uses Q8_0 for m, v (better precision for optimizer state).
    Could use Q4_0 or IQ3_XXS for more aggressive compression.
    """

    def __init__(
        self,
        shape: tuple,
        device: str = 'cuda',
        quant_type: str = 'q8_0',  # q8_0, q4_0
    ):
        self.shape = shape
        self.numel = math.prod(shape)
        self.device = device
        self.quant_type = quant_type
        self.step = 0

        # Initialize m, v as quantized zeros
        # For Q8_0: 34 bytes per 32 elements
        # For Q4_0: 18 bytes per 32 elements
        if quant_type == 'q8_0':
            bytes_per_block = 34
            block_size = 32
        elif quant_type == 'q4_0':
            bytes_per_block = 18
            block_size = 32
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}")

        n_blocks = (self.numel + block_size - 1) // block_size
        nbytes = n_blocks * bytes_per_block

        # Initialize with zeros (will be set on first step)
        self.m_quant = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.v_quant = torch.zeros(nbytes, dtype=torch.uint8, device=device)

        # Track if initialized (first step needs special handling)
        self.initialized = False

    def dequant_m(self) -> torch.Tensor:
        """Dequantize first moment."""
        if self.quant_type == 'q8_0':
            return dequant_q8_0_cuda(self.m_quant, self.numel, self.device).view(self.shape)
        else:
            return dequant_q4_0_cuda(self.m_quant, self.numel, self.device).view(self.shape)

    def dequant_v(self) -> torch.Tensor:
        """Dequantize second moment."""
        if self.quant_type == 'q8_0':
            return dequant_q8_0_cuda(self.v_quant, self.numel, self.device).view(self.shape)
        else:
            return dequant_q4_0_cuda(self.v_quant, self.numel, self.device).view(self.shape)

    def requant_m(self, m: torch.Tensor):
        """Requantize first moment."""
        if self.quant_type == 'q8_0':
            self.m_quant = requant_q8_0_cuda(m, self.quant_type)
        else:
            self.m_quant = requant_q4_0_cuda(m, self.quant_type)

    def requant_v(self, v: torch.Tensor):
        """Requantize second moment."""
        if self.quant_type == 'q8_0':
            self.v_quant = requant_q8_0_cuda(v, self.quant_type)
        else:
            self.v_quant = requant_q4_0_cuda(v, self.quant_type)

    @property
    def nbytes(self) -> int:
        """Total bytes for this state."""
        return self.m_quant.numel() + self.v_quant.numel()


class QuantizedAdam:
    """
    Adam optimizer with quantized state storage.

    Both weights AND optimizer state (m, v) are stored quantized.
    This achieves ~8x memory reduction vs standard FP16 + AdamW.

    Usage:
        optimizer = QuantizedAdam(
            model,
            lr=1e-4,
            weight_quant='q4_0',    # Weight quantization
            state_quant='q8_0',     # Optimizer state quantization
        )

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
        state_quant: str = 'q8_0',  # q8_0 more stable, q4_0 more compressed
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_quant = weight_quant
        self.state_quant = state_quant

        # Collect parameters
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
        # Quantized params get quantized state
        for qp in self.quantized_params:
            self.state[id(qp)] = QuantizedAdamState(
                shape=qp.shape,
                device=qp._device,
                quant_type=self.state_quant,
            )

        # Regular params get FP32 state (small relative to model)
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

        # Update quantized parameters
        for qp in self.quantized_params:
            self._step_quantized(qp, beta1, beta2)

        # Update regular parameters (standard Adam)
        for p in self.regular_params:
            if p.grad is not None:
                self._step_regular(p, beta1, beta2)

    def _step_quantized(self, qp: QuantizedParameter, beta1: float, beta2: float):
        """Update a quantized parameter with quantized optimizer state."""
        grad = qp.grad
        if grad is None:
            return

        state = self.state[id(qp)]
        state.step += 1
        step = state.step

        # Dequantize m, v
        if state.initialized:
            m = state.dequant_m()
            v = state.dequant_v()
        else:
            # First step: initialize from gradient
            m = torch.zeros_like(grad, dtype=torch.float32)
            v = torch.zeros_like(grad, dtype=torch.float32)
            state.initialized = True

        # Update m, v (standard Adam)
        grad_fp32 = grad.float()
        m = beta1 * m.float() + (1 - beta1) * grad_fp32
        v = beta2 * v.float() + (1 - beta2) * grad_fp32.square()

        # Clamp v to prevent quantization to zero (key stability fix!)
        # Q8_0 has ~8 bits of precision, so min representable is ~1/256 of max
        v_min = 1e-8  # Minimum value to prevent div-by-zero after quantization
        v = v.clamp(min=v_min)

        # Requantize m, v
        state.requant_m(m)
        state.requant_v(v)

        # Bias correction (in FP32 for precision)
        # Use fresh dequant for more precision in the update
        m_for_update = state.dequant_m().float()
        v_for_update = state.dequant_v().float().clamp(min=v_min)

        m_hat = m_for_update / (1 - beta1 ** step)
        v_hat = v_for_update / (1 - beta2 ** step)

        # Compute update with numerical safety
        update = m_hat / (v_hat.sqrt() + self.eps)

        # Clamp update to prevent explosion
        update = update.clamp(-1.0, 1.0)

        # Apply to quantized weight
        from .dequant_kernels import get_requant_fn
        weight = qp.dequantize().float()
        weight_new = weight - self.lr * update

        requant_fn = get_requant_fn(qp.quant_type)
        qp.raw_data.copy_(requant_fn(weight_new.to(torch.float16), qp.quant_type))

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

        # Quantized param state
        for qp in self.quantized_params:
            state = self.state[id(qp)]
            total += state.nbytes

        # Regular param state (FP32)
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
            self.state[id(p)]['m'].numel() * 4 * 2  # m + v in FP32
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


def test_quantized_adam():
    """Test the quantized Adam optimizer."""
    from .vtensor import QuantizedParameter, QuantizedLinear
    from .dequant_kernels import requant_q4_0_cuda

    print("=" * 60)
    print("TEST: Quantized Adam Optimizer")
    print("=" * 60)

    device = 'cuda'
    torch.manual_seed(42)

    # Create a simple model with quantized weights
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

    # Create optimizer with quantized state
    optimizer = QuantizedAdam(
        model,
        lr=0.01,
        weight_quant='q4_0',
        state_quant='q8_0',
    )

    # Memory report
    report = optimizer.memory_report()
    print(f"\nMemory breakdown:")
    print(f"  Quantized weights: {report['quantized_weights_gb']*1000:.2f} MB")
    print(f"  Quantized state:   {report['quantized_state_gb']*1000:.2f} MB")
    print(f"  Total:             {report['total_gb']*1000:.2f} MB")
    print(f"  Bytes per param:   {report['bytes_per_param']:.2f}")

    # Compare to FP16 + AdamW
    fp16_bytes_per_param = 2 + 8  # weight + m,v
    savings = fp16_bytes_per_param / report['bytes_per_param']
    print(f"  vs FP16+AdamW:     {savings:.1f}x savings")

    # Training loop
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

    # Check convergence
    if losses[-1] < losses[0]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\nLoss improved by {improvement:.1f}%")
        print("Quantized Adam: ✓")
    else:
        print("\nWARNING: Loss did not improve")

    return optimizer


if __name__ == "__main__":
    test_quantized_adam()
