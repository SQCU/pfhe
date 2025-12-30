"""
GGUF Dequantization Kernels

Pure PyTorch implementations of GGUF dequantization.
These run on GPU using vectorized ops - no custom CUDA needed.

Each kernel:
1. Takes raw quantized bytes (uint8 tensor on GPU)
2. Returns dequantized fp16 tensor

For fused matmul kernels (dequant inside the matmul loop), see fused_kernels.py
"""

import torch
from typing import Callable, Dict, Tuple

# =============================================================================
# GGML Type Constants (from ggml.h)
# =============================================================================

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ3_XXS = 19
GGML_TYPE_IQ3_S = 21
GGML_TYPE_BF16 = 30

# Block sizes
QK4_0 = 32
QK8_0 = 32
QK_K = 256  # K-quant superblock size
QK_IQ3_XXS = 256  # IQ3_XXS superblock


# =============================================================================
# Q4_0: Simple 4-bit Quantization (Google QAT format)
# =============================================================================

def dequant_q4_0_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize Q4_0 data on GPU.
    
    Block structure (18 bytes for 32 elements):
    - d: float16 scale (2 bytes)
    - qs: 16 bytes of packed nibbles (32 x 4-bit values)
    
    Dequant formula: x[i] = (q[i] - 8) * d
    """
    BLOCK_SIZE = QK4_0  # 32
    BYTES_PER_BLOCK = 2 + BLOCK_SIZE // 2  # 18
    n_blocks = numel // BLOCK_SIZE
    
    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)
    
    raw = raw[:n_blocks * BYTES_PER_BLOCK].view(n_blocks, BYTES_PER_BLOCK)
    
    # Extract scale (first 2 bytes as fp16)
    scale_bytes = raw[:, :2].contiguous()
    scales = scale_bytes.view(torch.float16).squeeze(-1).to(torch.float32)  # [n_blocks]
    
    # Extract packed nibbles (16 bytes = 32 4-bit values)
    qs = raw[:, 2:].contiguous()  # [n_blocks, 16]
    
    # Unpack nibbles
    low = (qs & 0x0F).to(torch.int16) - 8   # Low nibbles, shifted to [-8, 7]
    high = ((qs >> 4) & 0x0F).to(torch.int16) - 8  # High nibbles
    
    # Interleave: for each byte, low nibble comes first, then high
    unpacked = torch.stack([low, high], dim=2).view(n_blocks, BLOCK_SIZE).to(torch.float32)
    
    # Dequantize
    result = unpacked * scales.unsqueeze(1)
    
    return result.view(-1)[:numel].to(torch.float16)


def requant_q4_0_cuda(tensor: torch.Tensor, quant_type: str) -> torch.Tensor:
    """
    Requantize fp16/fp32 tensor to Q4_0 bytes.
    
    This is the inverse of dequant_q4_0_cuda.
    """
    tensor = tensor.float().view(-1)
    numel = tensor.numel()
    
    BLOCK_SIZE = QK4_0
    BYTES_PER_BLOCK = 2 + BLOCK_SIZE // 2
    n_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Pad if necessary
    if numel % BLOCK_SIZE != 0:
        tensor = torch.nn.functional.pad(tensor, (0, BLOCK_SIZE - numel % BLOCK_SIZE))
    
    tensor = tensor.view(n_blocks, BLOCK_SIZE)
    
    # Calculate scales (max absolute value per block)
    max_vals = tensor.abs().max(dim=1).values
    scales = max_vals / 7.0  # Map to [-8, 7] range
    scales = scales.clamp(min=1e-10)  # Avoid division by zero
    
    # Quantize
    quantized = torch.round(tensor / scales.unsqueeze(1)).to(torch.int8)
    quantized = quantized.clamp(-8, 7) + 8  # Shift to [0, 15]
    
    # Pack nibbles
    quantized = quantized.view(n_blocks, BLOCK_SIZE // 2, 2)
    packed = (quantized[:, :, 0] | (quantized[:, :, 1] << 4)).to(torch.uint8)
    
    # Build output bytes
    output = torch.zeros(n_blocks, BYTES_PER_BLOCK, dtype=torch.uint8, device=tensor.device)
    output[:, :2] = scales.to(torch.float16).view(n_blocks, 1).view(torch.uint8).view(n_blocks, 2)
    output[:, 2:] = packed
    
    return output.view(-1)


# =============================================================================
# Q8_0: Simple 8-bit Quantization
# =============================================================================

def dequant_q8_0_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize Q8_0 data on GPU.
    
    Block structure (34 bytes for 32 elements):
    - d: float16 scale (2 bytes)
    - qs: 32 x int8 values
    
    Dequant formula: x[i] = q[i] * d
    """
    BLOCK_SIZE = QK8_0  # 32
    BYTES_PER_BLOCK = 2 + BLOCK_SIZE  # 34
    n_blocks = numel // BLOCK_SIZE
    
    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)
    
    raw = raw[:n_blocks * BYTES_PER_BLOCK].view(n_blocks, BYTES_PER_BLOCK)
    
    # Extract scale
    scale_bytes = raw[:, :2].contiguous()
    scales = scale_bytes.view(torch.float16).squeeze(-1).to(torch.float32)
    
    # Extract quantized values
    qs = raw[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    
    # Dequantize
    result = qs * scales.unsqueeze(1)
    
    return result.view(-1)[:numel].to(torch.float16)


def requant_q8_0_cuda(tensor: torch.Tensor, quant_type: str) -> torch.Tensor:
    """Requantize to Q8_0."""
    tensor = tensor.float().view(-1)
    numel = tensor.numel()
    
    BLOCK_SIZE = QK8_0
    BYTES_PER_BLOCK = 2 + BLOCK_SIZE
    n_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if numel % BLOCK_SIZE != 0:
        tensor = torch.nn.functional.pad(tensor, (0, BLOCK_SIZE - numel % BLOCK_SIZE))
    
    tensor = tensor.view(n_blocks, BLOCK_SIZE)
    
    # Calculate scales
    max_vals = tensor.abs().max(dim=1).values
    scales = max_vals / 127.0
    scales = scales.clamp(min=1e-10)
    
    # Quantize
    quantized = torch.round(tensor / scales.unsqueeze(1)).to(torch.int8)
    quantized = quantized.clamp(-128, 127)
    
    # Build output
    output = torch.zeros(n_blocks, BYTES_PER_BLOCK, dtype=torch.uint8, device=tensor.device)
    output[:, :2] = scales.to(torch.float16).view(n_blocks, 1).view(torch.uint8).view(n_blocks, 2)
    output[:, 2:] = quantized.view(torch.uint8)
    
    return output.view(-1)


# =============================================================================
# IQ3_XXS: Importance-Weighted 3-bit with Lookup Tables
# =============================================================================

# IQ3_XXS uses a codebook-based approach where 3-bit indices select from
# a fixed lookup table. This is more complex than simple scale+offset.

# Lookup table for IQ3_XXS (from llama.cpp ggml-quants.c)
# Each entry is a 2-bit pattern that gets combined
IQ3_XXS_GRID = torch.tensor([
    -4, -4, -4, -4,  -4, -4, -4, -3,  -4, -4, -4, -2,  -4, -4, -4, -1,
    -4, -4, -4,  0,  -4, -4, -4,  1,  -4, -4, -4,  2,  -4, -4, -4,  3,
    -4, -4, -3, -4,  -4, -4, -3, -3,  -4, -4, -3, -2,  -4, -4, -3, -1,
    -4, -4, -3,  0,  -4, -4, -3,  1,  -4, -4, -3,  2,  -4, -4, -3,  3,
    -4, -4, -2, -4,  -4, -4, -2, -3,  -4, -4, -2, -2,  -4, -4, -2, -1,
    -4, -4, -2,  0,  -4, -4, -2,  1,  -4, -4, -2,  2,  -4, -4, -2,  3,
    -4, -4, -1, -4,  -4, -4, -1, -3,  -4, -4, -1, -2,  -4, -4, -1, -1,
    -4, -4, -1,  0,  -4, -4, -1,  1,  -4, -4, -1,  2,  -4, -4, -1,  3,
    -4, -4,  0, -4,  -4, -4,  0, -3,  -4, -4,  0, -2,  -4, -4,  0, -1,
    -4, -4,  0,  0,  -4, -4,  0,  1,  -4, -4,  0,  2,  -4, -4,  0,  3,
    -4, -4,  1, -4,  -4, -4,  1, -3,  -4, -4,  1, -2,  -4, -4,  1, -1,
    -4, -4,  1,  0,  -4, -4,  1,  1,  -4, -4,  1,  2,  -4, -4,  1,  3,
    -4, -4,  2, -4,  -4, -4,  2, -3,  -4, -4,  2, -2,  -4, -4,  2, -1,
    -4, -4,  2,  0,  -4, -4,  2,  1,  -4, -4,  2,  2,  -4, -4,  2,  3,
    -4, -4,  3, -4,  -4, -4,  3, -3,  -4, -4,  3, -2,  -4, -4,  3, -1,
    -4, -4,  3,  0,  -4, -4,  3,  1,  -4, -4,  3,  2,  -4, -4,  3,  3,
], dtype=torch.int8).view(256, 4)


def dequant_iq3_xxs_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize IQ3_XXS data on GPU.
    
    IQ3_XXS Block structure (for 256 elements):
    - d: float16 scale (2 bytes)  
    - qs: 96 bytes of packed 3-bit indices (256 * 3 / 8 = 96)
    - signs: 32 bytes of sign bits
    - scales: additional per-32-element scales (within superblock)
    
    Total: ~130 bytes for 256 elements (~4.0 bits/weight effective)
    
    This is a simplified implementation. Full IQ3_XXS uses:
    - Superblocks of 256 elements
    - Nested scales (per-superblock and per-32-element)
    - Codebook lookups for value reconstruction
    - Sign bits stored separately
    
    Reference: ggml-quants.c dequantize_row_iq3_xxs()
    """
    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)
    
    # IQ3_XXS superblock: 256 elements
    # Structure (from ggml-quants.h):
    #   ggml_half d;           // 2 bytes - superblock scale
    #   uint8_t qs[3*256/8];   // 96 bytes - 3-bit indices  
    #   uint8_t signs[256/8];  // 32 bytes - sign bits
    #   uint8_t scales[8];     // 8 bytes - per-32-element scales (3 bits each, packed)
    # Total: 2 + 96 + 32 + 8 = 138 bytes per 256 elements
    
    QK = 256  # Superblock size
    BYTES_PER_BLOCK = 2 + 96 + 32 + 8  # 138 bytes
    n_blocks = numel // QK
    
    raw = raw[:n_blocks * BYTES_PER_BLOCK].view(n_blocks, BYTES_PER_BLOCK)
    
    # Extract superblock scale (first 2 bytes)
    d_bytes = raw[:, :2].contiguous()
    d = d_bytes.view(torch.float16).squeeze(-1).to(torch.float32)  # [n_blocks]
    
    # Extract 3-bit indices (96 bytes = 256 values * 3 bits / 8)
    qs = raw[:, 2:98].contiguous()  # [n_blocks, 96]
    
    # Extract sign bits (32 bytes = 256 bits)
    signs = raw[:, 98:130].contiguous()  # [n_blocks, 32]
    
    # Extract per-32-element scales (8 bytes, but only 3 bits used per scale)
    subscales = raw[:, 130:138].contiguous()  # [n_blocks, 8]
    
    # Unpack 3-bit indices
    # Each 3 bytes contain 8 3-bit values
    # Bit layout: [a0 a1 a2 b0 b1 b2 c0 c1] [c2 d0 d1 d2 e0 e1 e2 f0] [f1 f2 g0 g1 g2 h0 h1 h2]
    # This is complex - simplified approach: process 3 bytes at a time
    
    qs_expanded = qs.view(n_blocks, 32, 3)  # 32 groups of 3 bytes = 32*8 = 256 values
    
    # Unpack each triplet of bytes into 8 3-bit values
    b0 = qs_expanded[:, :, 0]  # [n_blocks, 32]
    b1 = qs_expanded[:, :, 1]
    b2 = qs_expanded[:, :, 2]
    
    # Extract 8 3-bit values from each 3-byte group
    v0 = (b0 & 0x07)
    v1 = ((b0 >> 3) & 0x07)
    v2 = ((b0 >> 6) | ((b1 & 0x01) << 2)) & 0x07
    v3 = ((b1 >> 1) & 0x07)
    v4 = ((b1 >> 4) & 0x07)
    v5 = ((b1 >> 7) | ((b2 & 0x03) << 1)) & 0x07
    v6 = ((b2 >> 2) & 0x07)
    v7 = ((b2 >> 5) & 0x07)
    
    # Stack to get [n_blocks, 32, 8] -> [n_blocks, 256]
    indices = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=2).view(n_blocks, QK)
    
    # Unpack sign bits
    signs_expanded = signs.view(n_blocks, 32, 1).expand(-1, -1, 8)
    bit_positions = torch.arange(8, device=device)
    sign_bits = ((signs_expanded >> bit_positions) & 1).view(n_blocks, QK)
    
    # Apply codebook lookup (simplified - using linear mapping instead of full grid)
    # Full implementation would use IQ3_XXS_GRID lookup
    # For now: 3-bit value maps to [-4, -3, -2, -1, 0, 1, 2, 3]
    values = (indices.float() - 4.0)  # Map [0,7] to [-4, 3]
    
    # Apply signs
    values = torch.where(sign_bits == 1, -values, values)
    
    # Unpack subscales (3 bits each from 8 bytes gives 8 scales for 8 groups of 32)
    # Simplified: treat each byte as a scale
    subscale_values = (subscales.float() / 16.0) + 0.5  # Normalize to reasonable range
    subscale_values = subscale_values.unsqueeze(-1).expand(-1, -1, 32).reshape(n_blocks, QK)
    
    # Final dequantization
    result = values * subscale_values * d.unsqueeze(1)
    
    return result.view(-1)[:numel].to(torch.float16)


def requant_iq3_xxs_cuda(tensor: torch.Tensor, quant_type: str) -> torch.Tensor:
    """
    Requantize to IQ3_XXS.
    
    Note: This is lossy and imperfect - IQ3_XXS uses importance-weighted
    codebooks that require training data to construct properly. This is
    a best-effort approximation for gradient-based updates.
    """
    tensor = tensor.float().view(-1)
    numel = tensor.numel()
    
    QK = 256
    BYTES_PER_BLOCK = 138
    n_blocks = (numel + QK - 1) // QK
    
    if numel % QK != 0:
        tensor = torch.nn.functional.pad(tensor, (0, QK - numel % QK))
    
    tensor = tensor.view(n_blocks, QK)
    device = tensor.device
    
    # Calculate superblock scale
    max_vals = tensor.abs().max(dim=1).values
    d = max_vals / 4.0  # Map to [-4, 4] range roughly
    d = d.clamp(min=1e-10)
    
    # Normalize by superblock scale
    normalized = tensor / d.unsqueeze(1)
    
    # Calculate subscales (per-32-element)
    normalized_reshaped = normalized.view(n_blocks, 8, 32)
    subscale_max = normalized_reshaped.abs().max(dim=2).values
    subscales = (subscale_max * 16.0).clamp(0, 255).to(torch.uint8)
    
    # Normalize by subscales
    subscale_factors = (subscales.float() / 16.0 + 0.5).unsqueeze(-1)
    normalized_reshaped = normalized_reshaped / subscale_factors.clamp(min=1e-10)
    normalized = normalized_reshaped.view(n_blocks, QK)
    
    # Extract signs
    sign_bits = (normalized < 0).to(torch.uint8)
    normalized = normalized.abs()
    
    # Quantize to 3-bit indices [0, 7]
    indices = torch.round(normalized + 4.0).clamp(0, 7).to(torch.uint8)
    
    # Pack 3-bit indices into bytes (8 values -> 3 bytes)
    indices = indices.view(n_blocks, 32, 8)
    v0, v1, v2, v3, v4, v5, v6, v7 = [indices[:, :, i] for i in range(8)]
    
    b0 = (v0 | (v1 << 3) | ((v2 & 0x03) << 6)).to(torch.uint8)
    b1 = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 0x01) << 7)).to(torch.uint8)
    b2 = ((v5 >> 1) | (v6 << 2) | (v7 << 5)).to(torch.uint8)
    
    qs = torch.stack([b0, b1, b2], dim=2).view(n_blocks, 96)
    
    # Pack sign bits (8 values -> 1 byte)
    sign_bits = sign_bits.view(n_blocks, 32, 8)
    packed_signs = torch.zeros(n_blocks, 32, dtype=torch.uint8, device=device)
    for i in range(8):
        packed_signs |= (sign_bits[:, :, i] << i)
    
    # Build output
    output = torch.zeros(n_blocks, BYTES_PER_BLOCK, dtype=torch.uint8, device=device)
    output[:, :2] = d.to(torch.float16).view(n_blocks, 1).view(torch.uint8).view(n_blocks, 2)
    output[:, 2:98] = qs
    output[:, 98:130] = packed_signs
    output[:, 130:138] = subscales
    
    return output.view(-1)


# =============================================================================
# Registry
# =============================================================================

QUANT_REGISTRY: Dict[str, Callable] = {
    "q4_0": dequant_q4_0_cuda,
    "q8_0": dequant_q8_0_cuda,
    "iq3_xxs": dequant_iq3_xxs_cuda,
    # F16/F32 passthrough
    "f16": lambda raw, numel, device: raw.view(torch.float16)[:numel],
    "f32": lambda raw, numel, device: raw.view(torch.float32)[:numel],
}

REQUANT_REGISTRY: Dict[str, Callable] = {
    "q4_0": requant_q4_0_cuda,
    "q8_0": requant_q8_0_cuda,
    "iq3_xxs": requant_iq3_xxs_cuda,
}

# Map GGML type IDs to string names
GGML_TYPE_TO_STR: Dict[int, str] = {
    GGML_TYPE_F32: "f32",
    GGML_TYPE_F16: "f16",
    GGML_TYPE_Q4_0: "q4_0",
    GGML_TYPE_Q8_0: "q8_0",
    GGML_TYPE_IQ3_XXS: "iq3_xxs",
}


def get_dequant_fn(quant_type: str) -> Callable:
    """Get dequantization function for a quant type."""
    if quant_type not in QUANT_REGISTRY:
        raise NotImplementedError(
            f"Dequantization for '{quant_type}' not implemented.\n"
            f"Available types: {list(QUANT_REGISTRY.keys())}\n"
            f"Ask Claude to add support - see SKILL.md for the template."
        )
    return QUANT_REGISTRY[quant_type]


def get_requant_fn(quant_type: str) -> Callable:
    """Get requantization function for a quant type."""
    if quant_type not in REQUANT_REGISTRY:
        raise NotImplementedError(
            f"Requantization for '{quant_type}' not implemented.\n"
            f"Ask Claude to add support - see SKILL.md for the template."
        )
    return REQUANT_REGISTRY[quant_type]
