# PHFE - Posthumanity's First Exam

Benchmark suite measuring in-context learning vs memorization in language models.

## Installation

```bash
# Core install
uv pip install -e .

# With inference support (vLLM, transformers)
uv pip install -e ".[inference]"

# Full dev install
uv pip install -e ".[all]"
```

## Self-Vendored Inference APIs

We provide pure Python inference primitives that don't depend on vLLM's unstable API or llama-cpp-python's build complexity.

### GGUF Tensor Loading

Load quantized weights directly from GGUF files with pure Python dequantization.

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

**Supported quantization formats**: Q4_0, Q4_1, Q8_0, F16, BF16, F32

### Paged Attention with Prefix Caching

Efficient batched inference using PyTorch's `flex_attention` with automatic prefix caching.

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

### Batched Inference with flex_attention

```python
from phfe.inference import KVCacheManager
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

### Memory Estimation

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

## Architecture

```
src/phfe/
├── inference/
│   ├── gguf_loader.py       # GGUF parsing + dequantization
│   ├── paged_attention.py   # PageTable, KVCache, flex_attention
│   └── engine.py            # Legacy vLLM wrapper (optional)
├── data/                    # Benchmark data handling
├── tutors/                  # Tutor model interfaces
└── eval/                    # Evaluation harness
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4.0 (for flex_attention support)
- See `pyproject.toml` for full dependency list

## License

MIT
