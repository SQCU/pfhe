"""
Self-Vendored Inference Module

Local inference for testing ICR examples and format adherence.
Supports:
- vLLM for HuggingFace models (when installed)
- Direct GGUF loading with pure Python dequantization
- Streaming tensor loading for memory-efficient access
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for vLLM inference."""

    model_path: str  # HuggingFace ID or local GGUF path

    # Generation
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Batching
    batch_size: int = 4

    # Hardware
    gpu_memory_utilization: float = 0.85
    dtype: str = "auto"

    # Context
    max_model_len: Optional[int] = None  # None = use model default


@dataclass
class GenerationResult:
    """Result from generation."""

    prompt: str
    generated_text: str
    full_text: str
    finish_reason: str
    tokens_generated: int
    generation_time_ms: float
    logprobs: Optional[list[dict]] = None


# GGUF loading exports
from .gguf_loader import (
    GGUFReader,
    GGUFTensorInfo,
    GGUFHeader,
    GGUFModelPatcher,
    StreamingDequantizer,
    tensor_loading_context,
    streaming_dequant_context,
    load_gguf_tensors,
    inspect_gguf,
    print_gguf_info,
)

# Paged attention exports
from .paged_attention import (
    BlockManager,
    PageTable,
    KVCacheManager,
    PagedCausalAttention,
    RotaryEmbedding,
    create_causal_mask,
    estimate_kv_cache_memory,
)

# Model exports
from .model import (
    GGUFModel,
    ModelConfig,
    load_model,
    generate,
)

__all__ = [
    # Config
    "InferenceConfig",
    "GenerationResult",
    # GGUF
    "GGUFReader",
    "GGUFTensorInfo",
    "GGUFHeader",
    "GGUFModelPatcher",
    "StreamingDequantizer",
    "tensor_loading_context",
    "streaming_dequant_context",
    "load_gguf_tensors",
    "inspect_gguf",
    "print_gguf_info",
    # Paged Attention
    "BlockManager",
    "PageTable",
    "KVCacheManager",
    "PagedCausalAttention",
    "RotaryEmbedding",
    "create_causal_mask",
    "estimate_kv_cache_memory",
    # Model
    "GGUFModel",
    "ModelConfig",
    "load_model",
    "generate",
]
