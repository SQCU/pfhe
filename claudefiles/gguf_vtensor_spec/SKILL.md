# GGUF VTensor Training Skill

## Purpose

This skill enables **training through quantized GGUF weights** using virtual tensors (vtensors) that:
1. Store weights in their original quantized format (saving VRAM)
2. Dequantize on-the-fly during forward pass
3. Pass gradients through via Straight-Through Estimator (STE) during backward pass
4. Optionally accumulate sub-quantization-floor updates via carry buffers

## Core Thesis

There is **no fundamental technical barrier** to treating GGUF quantized weights as trainable parameters. The barriers are purely social/epistemic:
- GGUF is *perceived* as inference-only because that's how llama.cpp uses it
- The format is just bytes + a spec; nothing prevents gradient computation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     QuantizedParameter                       │
├─────────────────────────────────────────────────────────────┤
│  raw_data: torch.Tensor (uint8)  # Quantized bytes on GPU   │
│  shape: Tuple[int, ...]          # Logical tensor shape     │
│  quant_type: str                 # "q4_0", "iq3_xxs", etc   │
│  carry: Optional[torch.Tensor]   # Accumulated micro-updates│
├─────────────────────────────────────────────────────────────┤
│  forward() → dequantized tensor (transient, not stored)     │
│  backward() → STE gradient passthrough                      │
│  step() → apply optimizer update (with optional carry)      │
└─────────────────────────────────────────────────────────────┘
```

## Implemented Quant Types

| Type | Status | Notes |
|------|--------|-------|
| q4_0 | ✅ Implemented | Google QAT models use this |
| q8_0 | ✅ Implemented | Reference 8-bit |
| iq3_xxs | ✅ Implemented | Proves arbitrary formats work |
| * | NotImplementedError | Ask Claude to add |

## Usage

### Basic Forward/Backward

```python
from gguf_vtensor import QuantizedParameter, load_quantized_model

# Load model with quantized weights
model = load_quantized_model("model.gguf", device="cuda")

# Forward pass dequantizes on-the-fly
output = model(input_ids)

# Backward pass uses STE - gradients flow through
loss = criterion(output, targets)
loss.backward()

# Gradients are now on the dequantized proxy
# Apply to quantized weights via requantization or carry accumulation
optimizer.step()
```

### With Carry Buffers (Sub-Quantization Updates)

```python
from gguf_vtensor import QuantizedParameter, CarryOptimizer

# Carry buffers accumulate updates that would be lost to quantization floor
# Uses Johnson-Lindenstrauss lemma for compressed representation
optimizer = CarryOptimizer(
    model.parameters(),
    lr=1e-4,
    carry_dim=64,  # JL projection dimension
)

# Training loop
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()  # Updates carry, periodically flushes to weights
```

## Adding New Quant Types

When you encounter `NotImplementedError` for a quant type:

1. Find the spec in `ggml-quants.h` or GGUF docs
2. Implement `dequant_{type}_cuda()` following the pattern in `dequant_kernels.py`
3. Add entry to `QUANT_REGISTRY`
4. Test with `test_quant_roundtrip()`

### Template for New Quant Type

```python
def dequant_NEWTYPE_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize NEWTYPE data on GPU.
    
    Block structure (N bytes for M elements):
    - [describe layout from ggml-quants.h]
    """
    # 1. Calculate block parameters
    BLOCK_SIZE = ...  # Elements per block
    BYTES_PER_BLOCK = ...
    n_blocks = numel // BLOCK_SIZE
    
    # 2. Move to GPU if needed
    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)
    
    # 3. Reshape to blocks
    raw = raw[:n_blocks * BYTES_PER_BLOCK].view(n_blocks, BYTES_PER_BLOCK)
    
    # 4. Extract scale(s) and quantized values
    # [type-specific unpacking]
    
    # 5. Dequantize: value = f(q, scale, ...)
    # [type-specific formula]
    
    # 6. Return as fp16
    return result.view(-1)[:numel].to(torch.float16)
```

## Files in This Skill

- `SKILL.md` - This file
- `vtensor.py` - Core QuantizedParameter and autograd Function
- `dequant_kernels.py` - Dequantization implementations
- `carry_optimizer.py` - JL-compressed carry buffer optimizer
- `test_backward.py` - Validates STE gradient flow
- `test_iq3_xxs.py` - Proves wacky formats are trainable

## The Carry Buffer Idea

Standard quantization floors destroy small updates:
```
weight_q4 = 0.5  # Stored as 4-bit
grad = 0.001     # Would update to 0.501
requant(0.501) = 0.5  # Update lost!
```

Carry buffers accumulate these micro-updates:
```
carry += grad * lr  # carry = 0.001
# ... many steps later ...
carry = 0.05  # Accumulated enough to matter
weight_new = requant(dequant(weight) + carry)  # Now it sticks
carry = residual  # Keep what didn't fit
```

**JL Compression**: The carry buffer can be much smaller than the weight tensor. By Johnson-Lindenstrauss, we can project to O(log(n)/ε²) dimensions while preserving distances within (1±ε). This means a 4096×4096 weight matrix's carry can be stored in ~1000 floats instead of 16M.

## References

- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- ggml-quants.h: https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h
- IQ quants paper: https://arxiv.org/abs/2310.08659 (importance-weighted quantization)
- JL lemma: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
