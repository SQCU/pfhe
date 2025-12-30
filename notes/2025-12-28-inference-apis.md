# Self-Vendored Inference APIs

**Date**: 2025-12-28
**Status**: Complete (core primitives)

---

## Motivation

We needed local LM inference for training and eval that doesn't depend on:
- vLLM's unstable API and CUDA version churn
- llama-cpp-python's build complexity
- Ollama (rejected)
- Any upstream that "can't code and spend 95% of their time dueling by hostile API deprecations"

Solution: Pure Python GGUF loading + PyTorch's native `flex_attention` for paged KV cache inference.

---

## Architecture

```
src/phfe/inference/
├── __init__.py           # Exports all APIs
├── gguf_loader.py        # GGUF parsing + dequantization
├── paged_attention.py    # PageTable, KVCache, flex_attention
└── engine.py             # (Legacy vLLM wrapper, optional)
```

---

## Component 1: GGUF Tensor Loader

**File**: `gguf_loader.py`

Pure Python GGUF file parser with block dequantization. No C dependencies.

### Supported Quantization Formats

| Format | Block Size | Bytes/Block | Status |
|--------|-----------|-------------|--------|
| Q4_0 | 32 | 18 (fp16 scale + nibbles) | ✅ |
| Q4_1 | 32 | 20 (fp16 scale + min + nibbles) | ✅ |
| Q8_0 | 32 | 34 (fp16 scale + int8s) | ✅ |
| F16 | 1 | 2 | ✅ |
| BF16 | 1 | 2 | ✅ |
| F32 | 1 | 4 | ✅ |
| Q4_K, Q5_K, Q6_K | - | - | ❌ (K-quants not yet) |

### Usage

```python
from phfe.inference import (
    tensor_loading_context,
    streaming_dequant_context,
    inspect_gguf,
    GGUFModelPatcher,
)

# Inspect without loading
info = inspect_gguf("model.gguf")
print(f"Tensors: {info['n_tensors']}, Arch: {info['metadata']['general.architecture']}")

# Load specific tensors
with tensor_loading_context("model.gguf") as loader:
    embeddings = loader.read_tensor("token_embd.weight", device="cuda", dtype=torch.float16)
    output_proj = loader.read_tensor("output.weight", device="cuda", dtype=torch.float16)

# Stream tensors in batches (memory-efficient)
with streaming_dequant_context("model.gguf", batch_size=8, device="cuda") as stream:
    for batch in stream.iter_batches():
        model.load_partial(batch)

# Patch weights into a PyTorch model
patcher = GGUFModelPatcher("model.gguf")
with patcher.patch_context(model, device="cuda", dtype=torch.float16) as stats:
    print(f"Loaded {stats['loaded']} tensors, skipped {stats['skipped']}")
```

### Q4_0 Dequantization

```
Block structure (18 bytes for 32 elements):
  - d: float16 scale factor (2 bytes)
  - qs: 16 bytes of packed 4-bit values (32 nibbles)

Dequantization:
  value = (nibble - 8) * d
```

---

## Component 2: Paged Attention Primitives

**File**: `paged_attention.py`

Adapted from logsnrcat's PageTable and KVTManager for pure causal LM.

### Key Classes

| Class | Purpose |
|-------|---------|
| `BlockManager` | Block allocation with prefix caching via content hashing |
| `PageTable` | Logical→Physical slot mapping for paged attention |
| `KVCacheManager` | Full KV cache manager with prefix caching |
| `PagedCausalAttention` | Attention layer using flex_attention |
| `RotaryEmbedding` | Standard RoPE |

### Prefix Caching

Sequences with identical prefixes share KV cache blocks:

```python
from phfe.inference import KVCacheManager

cache = KVCacheManager(
    num_blocks=1024,
    block_size=64,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    device="cuda",
)

# System prompt (shared across requests)
system_prompt = tokenizer.encode("You are a helpful assistant...")

# Different user queries, same prefix
req1_tokens = system_prompt + tokenizer.encode("What is 2+2?")
req2_tokens = system_prompt + tokenizer.encode("Explain quantum physics")

# Allocate - prefix blocks are shared!
new_blocks_1, hits_1 = cache.allocate_sequence(0, req1_tokens)
new_blocks_2, hits_2 = cache.allocate_sequence(1, req2_tokens)

print(f"Request 2 cache hits: {hits_2}")  # Reuses prefix blocks
print(f"Cache hit rate: {cache.cache_hit_rate:.1%}")
```

### Content Hashing for Prefix Matching

```python
# Blocks are identified by chain hashing:
# hash(block_N) = xxhash64(hash(block_N-1) || token_ids)

# Same tokens → same hash → block reuse
# Different tokens → different hash → new allocation
```

### Batched Inference with flex_attention

```python
from phfe.inference import KVCacheManager, create_causal_mask
from torch.nn.attention.flex_attention import flex_attention

# Build mask for batch
mask = cache.build_causal_mask([req_id_0, req_id_1])

# Get KV cache for layer
k_cache, v_cache = cache.get_kv_cache(layer_idx=0)

# Run attention
out = flex_attention(q, k_cache, v_cache, block_mask=mask)

# Update cache with new KV
slot_mapping = cache.get_slot_mapping([req_id_0, req_id_1])
cache.update_kv_cache(layer_idx, k_new, v_new, slot_mapping)
```

---

## Component 3: Memory Estimation

```python
from phfe.inference import estimate_kv_cache_memory

mem = estimate_kv_cache_memory(
    num_blocks=2048,
    block_size=128,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
)

print(f"KV Cache: {mem['kv_cache_gb']:.2f} GB")
print(f"Capacity: {mem['capacity_tokens']:,} tokens")
```

---

## Integration with Training

### For GKD (Generalized Knowledge Distillation)

```python
from phfe.inference import tensor_loading_context, KVCacheManager

# Load teacher from GGUF
with tensor_loading_context("teacher.gguf") as loader:
    teacher_embeddings = loader.read_tensor("token_embd.weight")
    teacher_output = loader.read_tensor("output.weight")
    # ... load all layers

# Setup KV cache for efficient batched inference
cache = KVCacheManager(...)

# Run teacher inference with prefix caching
for batch in dataloader:
    # Allocate sequences (prefix caching kicks in)
    for i, tokens in enumerate(batch.input_ids):
        cache.allocate_sequence(i, tokens.tolist())

    # Get teacher logits via cached forward pass
    teacher_logits = teacher_forward(batch, cache)

    # Compute GKD loss against student
    loss = compute_gkd_loss(student_logits, teacher_logits)
```

### For Eval

```python
from phfe.inference import GGUFReader, KVCacheManager

# Load model weights
with GGUFReader("model.gguf") as reader:
    model = build_model_from_gguf(reader)

# Setup cache
cache = KVCacheManager(...)

# Run eval with efficient batched inference
for problem in benchmark:
    tokens = tokenize(problem.question)
    cache.allocate_sequence(0, tokens)

    # Generate with KV caching
    output = generate(model, cache, tokens, max_new_tokens=256)

    cache.free_sequence(0)
```

---

## Test Results

```
Block Manager Prefix Caching:
  Seq 1: blocks [0, 1, 2], cache hits: 0
  Seq 2: blocks [0, 1, 3], cache hits: 2  ← Shared prefix!

KV Cache Manager:
  40% cache hit rate on test sequences

GGUF Dequantization Verification:
  Embedding:  mean=+0.000381, std=0.030, no NaN/Inf ✓
  LM Head:    mean=+0.000450, std=0.029, no NaN/Inf ✓
  Attn Q:     mean=+0.000018, std=0.053, no NaN/Inf ✓
  FFN Gate:   mean=-0.000044, std=0.038, no NaN/Inf ✓

Q4 vs Q8 Logit Divergence (via llama-cpp):
  Jensen-Shannon: 0.019 (low - distributions similar)
  Top-1 agreement: YES
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `gguf_loader.py` | ~700 | GGUF parsing, dequantization, model patching |
| `paged_attention.py` | ~500 | PageTable, KVCache, flex_attention primitives |

---

## Dependencies

```toml
# Core (already in phfe)
torch >= 2.4.0  # flex_attention support
numpy

# Optional (for reference comparison)
llama-cpp-python  # KL divergence verification
```

---

## Not Yet Implemented

1. **K-quants (Q4_K, Q5_K, Q6_K)**: More complex super-block structure
2. **Full model forward pass**: Currently just primitives, need to wire up full transformer
3. **Continuous batching**: Single-batch inference only
4. **Speculative decoding**: Not yet

---

## Reference

Adapted from:
- logsnrcat's `src/model.py` (PageTable, BlockMask, flex_attention)
- logsnrcat's `src/utils.py` (KVTManager, BlockManager)
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
