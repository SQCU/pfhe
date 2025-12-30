"""
Carry Optimizer for Quantized Training

Standard quantization floors destroy small gradient updates:
    weight_q4 = quantize(0.500)
    grad = 0.001
    weight_new = quantize(0.500 - 0.001) = quantize(0.499) = 0.500  # Lost!

Carry buffers accumulate these micro-updates until they're significant
enough to survive requantization.

JL Compression:
    By Johnson-Lindenstrauss lemma, we can project n-dimensional vectors
    to O(log(n)/ε²) dimensions while preserving pairwise distances within
    (1±ε) factor. This lets us store carry buffers at ~1000 floats instead
    of millions, with controllable approximation error.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Iterator, Union
import math

from .vtensor import QuantizedParameter


class CarryOptimizer:
    """
    Optimizer for quantized parameters with carry buffer accumulation.

    Does NOT inherit from torch.optim.Optimizer because PyTorch's Optimizer
    requires Tensor params, but we handle QuantizedParameter modules.

    For each QuantizedParameter:
    1. Accumulate gradients in a (possibly JL-compressed) carry buffer
    2. When carry magnitude exceeds threshold, flush to quantized weights
    3. Keep residual that didn't survive requantization

    This allows training to make progress even when individual updates
    are smaller than the quantization floor.

    Usage:
        optimizer = CarryOptimizer(
            model,  # Pass the model, not model.parameters()
            lr=1e-4,
            carry_dim=64,      # JL projection dimension (None = full size)
            flush_threshold=0.1,  # Flush when ||carry|| > threshold
        )

        for batch in dataloader:
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        carry_dim: Optional[int] = 64,
        flush_threshold: float = 0.1,
    ):
        self.model = model
        self.lr = lr
        self.carry_dim = carry_dim
        self.flush_threshold = flush_threshold

        # Collect quantized and regular parameters
        self.quantized_params: List[QuantizedParameter] = []
        self.regular_params: List[nn.Parameter] = []
        self.state: Dict[Any, Dict] = {}

        self._collect_params()
        self._init_carry_buffers()

    def _collect_params(self):
        """Collect quantized and regular parameters from model."""
        for module in self.model.modules():
            if isinstance(module, QuantizedParameter):
                self.quantized_params.append(module)
            elif hasattr(module, 'weight') and isinstance(module.weight, QuantizedParameter):
                self.quantized_params.append(module.weight)

        # Regular parameters (excluding those in QuantizedParameter modules)
        seen_qparams = set(id(qp) for qp in self.quantized_params)
        for name, param in self.model.named_parameters():
            # Skip if this is inside a QuantizedParameter
            if param.requires_grad:
                self.regular_params.append(param)

    def _init_carry_buffers(self):
        """Initialize carry buffers for quantized parameters."""
        for qp in self.quantized_params:
            state = {}
            numel = qp.numel
            device = qp.raw_data.device

            if self.carry_dim is not None:
                state['proj_matrix'] = self._create_jl_matrix(numel, self.carry_dim, device)
                state['carry'] = torch.zeros(self.carry_dim, device=device)
                state['carry_dim'] = self.carry_dim
            else:
                state['carry'] = torch.zeros(numel, device=device)
                state['carry_dim'] = None

            self.state[id(qp)] = state

    def _create_jl_matrix(self, n: int, k: int, device) -> torch.Tensor:
        """Create Johnson-Lindenstrauss random projection matrix."""
        return torch.randn(n, k, device=device) / math.sqrt(k)

    @torch.no_grad()
    def step(self):
        """Perform optimization step."""
        # Update quantized parameters
        for qp in self.quantized_params:
            self._step_quantized(qp)

        # Update regular parameters with simple SGD
        for p in self.regular_params:
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-self.lr)

    def _step_quantized(self, param: QuantizedParameter) -> None:
        """Update a quantized parameter with carry accumulation."""
        grad = param.grad
        if grad is None:
            return

        state = self.state[id(param)]
        carry = state['carry']
        carry_dim = state['carry_dim']

        grad_flat = grad.float().view(-1)

        if carry_dim is not None:
            proj = state['proj_matrix']
            grad_compressed = torch.mv(proj.T, grad_flat * self.lr)
            carry.add_(grad_compressed)

            carry_norm = torch.norm(carry)
            if carry_norm > self.flush_threshold:
                carry_full = torch.mv(proj, carry)
                self._apply_update(param, carry_full)
                carry.zero_()
        else:
            carry.add_(grad_flat * self.lr)

            carry_norm = torch.norm(carry)
            if carry_norm > self.flush_threshold:
                self._apply_update(param, carry)
                carry.zero_()

    def _apply_update(self, param: QuantizedParameter, update: torch.Tensor) -> None:
        """Apply accumulated update to quantized weights."""
        from .dequant_kernels import get_requant_fn

        weight = param.dequantize()
        weight_new = weight - update.view(param.shape)

        requant_fn = get_requant_fn(param.quant_type)
        param.raw_data.copy_(requant_fn(weight_new, param.quant_type))

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

    def get_carry_stats(self) -> Dict[str, Any]:
        """Get statistics about carry buffer usage."""
        stats = {
            'quantized_params': len(self.quantized_params),
            'regular_params': len(self.regular_params),
            'carry_norms': [],
            'compression_ratios': [],
        }

        for qp in self.quantized_params:
            state = self.state[id(qp)]
            stats['carry_norms'].append(torch.norm(state['carry']).item())
            stats['compression_ratios'].append(qp.compression_ratio)

        return stats

    def state_memory_bytes(self) -> int:
        """Calculate optimizer state memory usage."""
        total = 0
        for qp in self.quantized_params:
            state = self.state[id(qp)]
            total += state['carry'].numel() * state['carry'].element_size()
            if 'proj_matrix' in state:
                total += state['proj_matrix'].numel() * state['proj_matrix'].element_size()
        return total


class AdamCarry:
    """
    Adam optimizer variant for quantized parameters with carry buffers.

    Maintains momentum and variance in the compressed carry space.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        carry_dim: Optional[int] = 64,
        flush_threshold: float = 0.1,
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.carry_dim = carry_dim
        self.flush_threshold = flush_threshold

        self.quantized_params: List[QuantizedParameter] = []
        self.regular_params: List[nn.Parameter] = []
        self.state: Dict[Any, Dict] = {}

        self._collect_params()
        self._init_state()

    def _collect_params(self):
        """Collect quantized and regular parameters from model."""
        for module in self.model.modules():
            if isinstance(module, QuantizedParameter):
                self.quantized_params.append(module)
            elif hasattr(module, 'weight') and isinstance(module.weight, QuantizedParameter):
                self.quantized_params.append(module.weight)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.regular_params.append(param)

    def _init_state(self):
        """Initialize optimizer state."""
        # Quantized params
        for qp in self.quantized_params:
            state = {'step': 0}
            numel = qp.numel
            device = qp.raw_data.device

            if self.carry_dim is not None:
                state['proj'] = torch.randn(numel, self.carry_dim, device=device) / math.sqrt(self.carry_dim)
                state['m'] = torch.zeros(self.carry_dim, device=device)
                state['v'] = torch.zeros(self.carry_dim, device=device)
                state['carry'] = torch.zeros(self.carry_dim, device=device)
            else:
                state['m'] = torch.zeros(numel, device=device)
                state['v'] = torch.zeros(numel, device=device)
                state['carry'] = torch.zeros(numel, device=device)

            self.state[id(qp)] = state

        # Regular params
        for p in self.regular_params:
            self.state[id(p)] = {
                'step': 0,
                'm': torch.zeros_like(p),
                'v': torch.zeros_like(p),
            }

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas

        for qp in self.quantized_params:
            self._step_quantized_adam(qp, beta1, beta2)

        for p in self.regular_params:
            if p.grad is not None:
                self._step_standard_adam(p, beta1, beta2)

    def _step_quantized_adam(self, p: QuantizedParameter, beta1: float, beta2: float):
        grad = p.grad
        if grad is None:
            return

        state = self.state[id(p)]
        state['step'] += 1

        m, v = state['m'], state['v']
        carry = state['carry']

        grad_flat = grad.float().view(-1)

        if 'proj' in state:
            proj = state['proj']
            grad_c = torch.mv(proj.T, grad_flat)

            m.mul_(beta1).add_(grad_c, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad_c, grad_c, value=1 - beta2)

            step = state['step']
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            update_c = m_hat / (v_hat.sqrt() + self.eps)
            carry.add_(update_c, alpha=self.lr)

            if torch.norm(carry) > self.flush_threshold:
                carry_full = torch.mv(proj, carry)
                self._apply_update(p, carry_full)
                carry.zero_()
        else:
            m.mul_(beta1).add_(grad_flat, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad_flat, grad_flat, value=1 - beta2)

            step = state['step']
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            update = m_hat / (v_hat.sqrt() + self.eps)
            carry.add_(update, alpha=self.lr)

            if torch.norm(carry) > self.flush_threshold:
                self._apply_update(p, carry)
                carry.zero_()

    def _step_standard_adam(self, p: nn.Parameter, beta1: float, beta2: float):
        state = self.state[id(p)]
        state['step'] += 1

        m, v = state['m'], state['v']
        grad = p.grad

        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = state['step']
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        p.data.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def _apply_update(self, p: QuantizedParameter, update: torch.Tensor):
        from .dequant_kernels import get_requant_fn

        weight = p.dequantize()
        weight_new = weight - update.view(p.shape)

        requant_fn = get_requant_fn(p.quant_type)
        p.raw_data.copy_(requant_fn(weight_new, p.quant_type))

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
        """Calculate optimizer state memory usage."""
        total = 0
        for qp in self.quantized_params:
            state = self.state[id(qp)]
            for key in ['m', 'v', 'carry']:
                if key in state:
                    total += state[key].numel() * state[key].element_size()
            if 'proj' in state:
                total += state['proj'].numel() * state['proj'].element_size()

        for p in self.regular_params:
            state = self.state[id(p)]
            total += state['m'].numel() * state['m'].element_size()
            total += state['v'].numel() * state['v'].element_size()

        return total
