"""
GGUF Virtual Tensors with Autograd Support

Quantized weights that:
1. Store in compressed format (saving VRAM)
2. Dequantize on-the-fly in forward pass
3. Pass gradients via Straight-Through Estimator (STE)
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass

from .dequant_kernels import QUANT_REGISTRY, get_dequant_fn


class DequantSTE(Function):
    """
    Straight-Through Estimator for quantized weights.
    
    Forward: dequantize quantized bytes → fp16/bf16 tensor
    Backward: pass gradient through unchanged (as if no quantization)
    
    This is the standard approach for training quantized networks.
    The gradient is an unbiased estimator of the true gradient direction,
    even though magnitudes may be off due to quantization effects.
    """
    
    @staticmethod
    def forward(
        ctx,
        raw_data: torch.Tensor,
        shape: Tuple[int, ...],
        quant_type: str,
        numel: int,
    ) -> torch.Tensor:
        """
        Dequantize on forward pass.
        
        Args:
            raw_data: Quantized bytes on GPU (uint8 tensor)
            shape: Logical tensor shape
            quant_type: Quantization type string ("q4_0", "iq3_xxs", etc)
            numel: Number of elements
            
        Returns:
            Dequantized tensor in fp16
        """
        # Save for backward (we need shape info)
        ctx.shape = shape
        ctx.quant_type = quant_type
        
        # Get the appropriate dequant function
        dequant_fn = get_dequant_fn(quant_type)
        
        # Dequantize
        result = dequant_fn(raw_data, numel, str(raw_data.device))
        return result.view(shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[None, None, None, None]:
        """
        STE backward: pass gradient through unchanged.
        
        The gradient w.r.t. the quantized bytes is conceptually the same
        as the gradient w.r.t. the dequantized values. We return None for
        raw_data because we don't actually backprop into bytes - instead,
        the QuantizedParameter captures grad_output and handles the update.
        """
        # We don't return gradient for raw_data, shape, quant_type, numel
        # The gradient is captured by the QuantizedParameter wrapper
        return None, None, None, None


class DequantSTEWithGradCapture(Function):
    """
    STE variant that captures gradients for external handling.
    
    Used when we want to accumulate gradients in a carry buffer
    rather than immediately applying them to quantized weights.
    """
    
    @staticmethod
    def forward(
        ctx,
        raw_data: torch.Tensor,
        shape: Tuple[int, ...],
        quant_type: str,
        numel: int,
        grad_holder: Any,  # Mutable object to store gradient
    ) -> torch.Tensor:
        ctx.grad_holder = grad_holder
        ctx.shape = shape
        
        dequant_fn = get_dequant_fn(quant_type)
        result = dequant_fn(raw_data, numel, str(raw_data.device))
        return result.view(shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[None, None, None, None, None]:
        # Capture gradient in holder for external processing
        ctx.grad_holder.grad = grad_output.clone()
        return None, None, None, None, None


@dataclass
class GradHolder:
    """Simple mutable container for captured gradients."""
    grad: Optional[torch.Tensor] = None


class QuantizedParameter(nn.Module):
    """
    A quantized weight that behaves like a trainable parameter.
    
    Key properties:
    - Stores weight in quantized format (uint8 bytes)
    - Dequantizes on-the-fly when accessed
    - Gradients flow through via STE
    - Optional carry buffer for sub-quantization-floor updates
    
    Usage:
        qparam = QuantizedParameter(raw_bytes, shape, "q4_0")
        
        # In forward pass:
        weight = qparam()  # Returns dequantized tensor
        output = F.linear(x, weight)
        
        # After backward:
        qparam.apply_gradient(lr=1e-4)  # Updates quantized weight
    """
    
    def __init__(
        self,
        raw_data: torch.Tensor,
        shape: Tuple[int, ...],
        quant_type: str,
        device: str = "cuda",
        enable_carry: bool = False,
        carry_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.shape = shape
        self.quant_type = quant_type
        self._device = device
        self.enable_carry = enable_carry
        
        # Calculate numel
        self.numel = 1
        for d in shape:
            self.numel *= d
        
        # Store quantized data as buffer (not parameter - saves memory tracking)
        self.register_buffer("raw_data", raw_data.to(device))
        
        # Gradient holder for capturing STE gradients
        self.grad_holder = GradHolder()
        
        # Carry buffer for accumulating sub-quantization updates
        if enable_carry:
            # JL-compressed carry or full-size
            if carry_dim is not None:
                # Random projection matrix (fixed, not learned)
                # JL says we need O(log(n)/eps^2) dims to preserve distances
                self.register_buffer(
                    "proj_matrix",
                    torch.randn(self.numel, carry_dim, device=device) / (carry_dim ** 0.5)
                )
                self.register_buffer(
                    "carry",
                    torch.zeros(carry_dim, device=device)
                )
                self.carry_dim = carry_dim
            else:
                self.register_buffer(
                    "carry",
                    torch.zeros(self.numel, device=device)
                )
                self.carry_dim = None
        else:
            self.carry = None
            self.carry_dim = None
        
        # Cache dequantized weight for repeated access within same forward
        self._cached_dequant: Optional[torch.Tensor] = None
        self._cache_valid = False
    
    def forward(self) -> torch.Tensor:
        """
        Dequantize and return weight tensor.
        
        Uses STE for gradient computation.
        """
        return DequantSTEWithGradCapture.apply(
            self.raw_data,
            self.shape,
            self.quant_type,
            self.numel,
            self.grad_holder,
        )
    
    def __call__(self) -> torch.Tensor:
        return self.forward()
    
    @property
    def grad(self) -> Optional[torch.Tensor]:
        """Access captured gradient."""
        return self.grad_holder.grad
    
    def zero_grad(self):
        """Clear captured gradient."""
        self.grad_holder.grad = None
    
    def dequantize(self) -> torch.Tensor:
        """
        Full dequantization without autograd.
        
        Use this for inspection/debugging, not training.
        """
        with torch.no_grad():
            dequant_fn = get_dequant_fn(self.quant_type)
            return dequant_fn(self.raw_data, self.numel, self._device).view(self.shape)
    
    def apply_gradient(
        self,
        lr: float,
        requant_fn: Optional[Callable] = None,
    ) -> None:
        """
        Apply captured gradient to quantized weight.
        
        Two strategies:
        1. Direct requantization (default): dequant → update → requant
        2. With carry: accumulate in carry buffer, flush when significant
        
        Args:
            lr: Learning rate
            requant_fn: Optional custom requantization function
        """
        if self.grad_holder.grad is None:
            return
        
        grad = self.grad_holder.grad
        
        if self.enable_carry and self.carry is not None:
            self._apply_with_carry(grad, lr)
        else:
            self._apply_direct(grad, lr, requant_fn)
        
        # Clear gradient after applying
        self.grad_holder.grad = None
    
    def _apply_direct(
        self,
        grad: torch.Tensor,
        lr: float,
        requant_fn: Optional[Callable],
    ) -> None:
        """Direct update: dequant → update → requant."""
        # Dequantize current weight
        weight = self.dequantize()
        
        # Apply gradient
        weight = weight - lr * grad
        
        # Requantize
        if requant_fn is None:
            requant_fn = get_requant_fn(self.quant_type)
        
        self.raw_data.copy_(requant_fn(weight, self.quant_type))
    
    def _apply_with_carry(self, grad: torch.Tensor, lr: float) -> None:
        """
        Update with carry buffer for sub-quantization accumulation.
        
        The carry buffer stores accumulated updates that are too small
        to survive requantization. When carry grows large enough, we
        flush it to the actual weights.
        """
        update = lr * grad.view(-1)
        
        if self.carry_dim is not None:
            # JL-compressed: project update to low-dim space
            compressed_update = torch.mv(self.proj_matrix.T, update)
            self.carry.add_(compressed_update)
            
            # Check if carry is significant enough to flush
            carry_norm = torch.norm(self.carry)
            if carry_norm > 0.1:  # Threshold for flushing
                # Reconstruct full-dim update
                full_carry = torch.mv(self.proj_matrix, self.carry)
                
                # Apply to weight
                weight = self.dequantize()
                weight = weight - full_carry.view(self.shape)
                
                # Requantize
                requant_fn = get_requant_fn(self.quant_type)
                self.raw_data.copy_(requant_fn(weight, self.quant_type))
                
                # Reset carry
                self.carry.zero_()
        else:
            # Full-size carry
            self.carry.add_(update)
            
            # Periodic flush
            carry_norm = torch.norm(self.carry)
            if carry_norm > 0.1:
                weight = self.dequantize()
                weight = weight - self.carry.view(self.shape)
                
                requant_fn = get_requant_fn(self.quant_type)
                self.raw_data.copy_(requant_fn(weight, self.quant_type))
                
                self.carry.zero_()
    
    @property
    def nbytes_quantized(self) -> int:
        """Size of quantized storage."""
        return self.raw_data.numel()
    
    @property  
    def nbytes_fp16(self) -> int:
        """Size if stored as fp16."""
        return self.numel * 2
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs fp16."""
        return self.nbytes_fp16 / self.nbytes_quantized


class QuantizedLinear(nn.Module):
    """
    Linear layer with quantized weight.
    
    Drop-in replacement for nn.Linear with ~4x memory savings.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        raw_weight: torch.Tensor,
        quant_type: str,
        bias: Optional[torch.Tensor] = None,
        device: str = "cuda",
        enable_carry: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = QuantizedParameter(
            raw_weight,
            shape=(out_features, in_features),
            quant_type=quant_type,
            device=device,
            enable_carry=enable_carry,
        )
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight()
        return nn.functional.linear(x.to(weight.dtype), weight, self.bias)
    
    def apply_gradient(self, lr: float) -> None:
        """Apply captured gradients."""
        self.weight.apply_gradient(lr)


def get_requant_fn(quant_type: str) -> Callable:
    """
    Get requantization function for a quant type.
    
    Requantization: fp16/fp32 → quantized bytes
    """
    from .dequant_kernels import REQUANT_REGISTRY
    
    if quant_type not in REQUANT_REGISTRY:
        raise NotImplementedError(
            f"Requantization for {quant_type} not implemented. "
            f"Ask Claude to add it - see SKILL.md for template."
        )
    
    return REQUANT_REGISTRY[quant_type]
