"""
GGUF VTensor: Training Through Quantized Weights

This package enables training LLMs through quantized GGUF weights using:
1. Virtual tensors that store weights in quantized format
2. On-the-fly dequantization during forward pass
3. Straight-Through Estimator (STE) for backward pass
4. Optional carry buffers for sub-quantization-floor updates

Key insight: There's no technical barrier to training through GGUF formats.
The barrier is purely social/epistemic - people think GGUF is "inference only"
because that's how llama.cpp uses it.

Usage:
    from gguf_vtensor import QuantizedParameter, QuantizedLinear, CarryOptimizer
    
    # Load quantized weight
    qparam = QuantizedParameter(raw_bytes, shape, "q4_0")
    
    # Use in forward pass (dequantizes automatically)
    weight = qparam()
    output = F.linear(x, weight)
    
    # Backward pass works via STE
    loss.backward()
    
    # Update with carry buffer accumulation
    optimizer.step()
"""

from .vtensor import (
    QuantizedParameter,
    QuantizedLinear,
    DequantSTE,
    DequantSTEWithGradCapture,
    GradHolder,
)

from .dequant_kernels import (
    # Functions
    get_dequant_fn,
    get_requant_fn,
    dequant_q4_0_cuda,
    dequant_q8_0_cuda,
    dequant_iq3_xxs_cuda,
    requant_q4_0_cuda,
    requant_q8_0_cuda,
    requant_iq3_xxs_cuda,
    # Registries
    QUANT_REGISTRY,
    REQUANT_REGISTRY,
    GGML_TYPE_TO_STR,
    # Constants
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_IQ3_XXS,
)

from .carry_optimizer import (
    CarryOptimizer,
    AdamCarry,
)

__all__ = [
    # Core vtensor
    "QuantizedParameter",
    "QuantizedLinear",
    "DequantSTE",
    "DequantSTEWithGradCapture",
    "GradHolder",
    # Dequant functions
    "get_dequant_fn",
    "get_requant_fn",
    "dequant_q4_0_cuda",
    "dequant_q8_0_cuda", 
    "dequant_iq3_xxs_cuda",
    "requant_q4_0_cuda",
    "requant_q8_0_cuda",
    "requant_iq3_xxs_cuda",
    # Registries
    "QUANT_REGISTRY",
    "REQUANT_REGISTRY",
    "GGML_TYPE_TO_STR",
    # Optimizers
    "CarryOptimizer",
    "AdamCarry",
    # Constants
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_IQ3_XXS",
]

__version__ = "0.1.0"
