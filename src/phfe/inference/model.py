"""
Transformer Model for GGUF Inference

Pure PyTorch transformer using our GGUF loader and paged attention primitives.
No llama-cpp dependency - just PyTorch + flex_attention.

Supports Gemma/Llama-style architectures:
- RMSNorm
- SwiGLU FFN (gate_proj, up_proj, down_proj)
- RoPE (Rotary Position Embeddings)
- Grouped Query Attention (GQA) optional

Usage:
    model = GGUFModel.from_gguf("model.gguf", device="cuda")
    output_ids = model.generate(input_ids, max_new_tokens=256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import math

from .gguf_loader import GGUFReader, inspect_gguf
from .paged_attention import KVCacheManager, RotaryEmbedding, create_causal_mask

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration extracted from GGUF metadata."""

    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int  # For GQA
    head_dim: int
    intermediate_dim: int
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    max_seq_len: int = 8192
    use_qk_norm: bool = False  # Gemma3
    use_post_attn_norm: bool = False  # Gemma3
    use_post_ffn_norm: bool = False  # Gemma3
    tie_embeddings: bool = True  # Share embed/lm_head weights

    @classmethod
    def from_gguf_metadata(
        cls,
        metadata: Dict[str, Any],
        vocab_size_override: Optional[int] = None,
    ) -> "ModelConfig":
        """Extract config from GGUF metadata."""
        # Try different naming conventions (llama.*, gemma.*, etc.)
        def get_val(keys: List[str], default=None):
            for k in keys:
                if k in metadata:
                    return metadata[k]
            return default

        arch = get_val(["general.architecture"], "llama")
        prefix = f"{arch}."

        hidden_dim = get_val([f"{prefix}embedding_length", "llama.embedding_length"], 4096)
        num_heads = get_val([f"{prefix}attention.head_count", "llama.attention.head_count"], 32)
        num_kv_heads = get_val([
            f"{prefix}attention.head_count_kv",
            "llama.attention.head_count_kv"
        ], num_heads)
        # head_dim can be explicitly set (key_length) or derived
        head_dim = get_val([
            f"{prefix}attention.key_length",
            f"{prefix}attention.head_dim",
        ], hidden_dim // num_heads)

        # Detect Gemma3 features
        is_gemma3 = arch == "gemma3"

        # Vocab size: prefer override (from embedding tensor) over metadata
        if vocab_size_override is not None:
            vocab_size = vocab_size_override
        else:
            vocab_size = get_val([f"{prefix}vocab_size", "llama.vocab_size"], 32000)

        return cls(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=get_val([f"{prefix}block_count", "llama.block_count"], 32),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=get_val([
                f"{prefix}feed_forward_length",
                "llama.feed_forward_length"
            ], hidden_dim * 4),
            rms_norm_eps=get_val([f"{prefix}attention.layer_norm_rms_epsilon"], 1e-6),
            rope_base=get_val([f"{prefix}rope.freq_base"], 10000.0),
            max_seq_len=get_val([f"{prefix}context_length"], 8192),
            use_qk_norm=is_gemma3,
            use_post_attn_norm=is_gemma3,
            use_post_ffn_norm=is_gemma3,
            tie_embeddings=is_gemma3,  # Gemma3 ties embeddings
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Llama/Gemma style)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional GQA."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # QK normalization (Gemma3)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, rms_norm_eps)

        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape

        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization (Gemma3)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rope(q, k, positions)

        # Expand KV for GQA if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Use KV cache if provided
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)

        # Output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_out)

        # Return new KV for caching
        return output, (k, v)


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_post_attn_norm = config.use_post_attn_norm
        self.use_post_ffn_norm = config.use_post_ffn_norm

        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = Attention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            max_seq_len=config.max_seq_len,
            use_qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Post-attention norm (Gemma3: residual normalization)
        if self.use_post_attn_norm:
            self.post_attention_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config.hidden_dim, config.intermediate_dim)

        # Post-FFN norm (Gemma3)
        if self.use_post_ffn_norm:
            self.post_ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(hidden_states, positions, mask, kv_cache)

        # Post-attention norm (Gemma3)
        if self.use_post_attn_norm:
            hidden_states = self.post_attention_norm(hidden_states)

        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Post-FFN norm (Gemma3)
        if self.use_post_ffn_norm:
            hidden_states = self.post_ffn_norm(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class GGUFModel(nn.Module):
    """
    Transformer model loaded from GGUF.

    Supports autoregressive generation with KV caching.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # LM head - may be tied to embeddings
        if config.tie_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    @classmethod
    def from_gguf(
        cls,
        path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "GGUFModel":
        """Load model from GGUF file."""
        logger.info(f"Loading model from {path}")

        # Get metadata and config
        info = inspect_gguf(path)

        # Get vocab_size from embedding tensor (not always in metadata)
        emb_info = info["tensors"].get("token_embd.weight")
        if emb_info:
            # Shape is (vocab_size, hidden_dim) after loading
            vocab_size = emb_info["shape"][0]
        else:
            vocab_size = None

        config = ModelConfig.from_gguf_metadata(info["metadata"], vocab_size_override=vocab_size)

        logger.info(f"Config: {config.num_layers} layers, {config.hidden_dim} hidden, "
                   f"{config.num_heads} heads, {config.vocab_size} vocab")

        # Create model
        model = cls(config)

        # Load weights
        with GGUFReader(path) as reader:
            loaded = 0
            skipped = 0

            for name, tensor_info in reader.tensors.items():
                # Map GGUF names to our parameter names
                param_name = cls._map_tensor_name(name)
                if param_name is None:
                    skipped += 1
                    continue

                # Get parameter
                try:
                    param = model.get_parameter(param_name)
                except (KeyError, AttributeError):
                    skipped += 1
                    continue

                # Load and assign
                tensor = reader.read_tensor(name, device=device, dtype=dtype)

                if tensor.shape != param.shape:
                    logger.warning(f"Shape mismatch for {name}: "
                                 f"got {tensor.shape}, expected {param.shape}")
                    skipped += 1
                    continue

                param.data.copy_(tensor)
                loaded += 1

            logger.info(f"Loaded {loaded} tensors, skipped {skipped}")

        return model.to(device=device, dtype=dtype)

    @staticmethod
    def _map_tensor_name(gguf_name: str) -> Optional[str]:
        """Map GGUF tensor name to PyTorch parameter name."""
        # Common mappings
        mappings = {
            "token_embd.weight": "embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "output_norm.weight": "norm.weight",
        }

        if gguf_name in mappings:
            return mappings[gguf_name]

        # Block tensors: blk.{i}.{component}
        import re
        match = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
        if match:
            layer_idx = match.group(1)
            component = match.group(2)

            component_map = {
                # Attention projections
                "attn_q.weight": f"layers.{layer_idx}.self_attn.q_proj.weight",
                "attn_k.weight": f"layers.{layer_idx}.self_attn.k_proj.weight",
                "attn_v.weight": f"layers.{layer_idx}.self_attn.v_proj.weight",
                "attn_output.weight": f"layers.{layer_idx}.self_attn.o_proj.weight",
                # FFN
                "ffn_gate.weight": f"layers.{layer_idx}.mlp.gate_proj.weight",
                "ffn_up.weight": f"layers.{layer_idx}.mlp.up_proj.weight",
                "ffn_down.weight": f"layers.{layer_idx}.mlp.down_proj.weight",
                # Norms (standard)
                "attn_norm.weight": f"layers.{layer_idx}.input_layernorm.weight",
                "ffn_norm.weight": f"layers.{layer_idx}.post_attention_layernorm.weight",
                # Gemma3: QK norms
                "attn_q_norm.weight": f"layers.{layer_idx}.self_attn.q_norm.weight",
                "attn_k_norm.weight": f"layers.{layer_idx}.self_attn.k_norm.weight",
                # Gemma3: Post-attention/FFN norms
                "post_attention_norm.weight": f"layers.{layer_idx}.post_attention_norm.weight",
                "post_ffw_norm.weight": f"layers.{layer_idx}.post_ffn_norm.weight",
            }

            return component_map.get(component)

        return None

    def get_parameter(self, name: str) -> nn.Parameter:
        """Get parameter by dot-separated name."""
        parts = name.split(".")
        module = self
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return getattr(module, parts[-1])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            positions: [seq_len] position indices (auto-generated if None)
            kv_cache: List of (k, v) tuples per layer

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_cache: Updated KV cache
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Auto-generate positions
        if positions is None:
            if kv_cache is not None and kv_cache[0][0] is not None:
                past_len = kv_cache[0][0].shape[2]
                positions = torch.arange(past_len, past_len + seq_len, device=device)
            else:
                positions = torch.arange(seq_len, device=device)

        # Causal mask
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)

            # Extend mask for past KV
            if kv_cache is not None and kv_cache[0][0] is not None:
                past_len = kv_cache[0][0].shape[2]
                past_mask = torch.zeros((seq_len, past_len), device=device)
                mask = torch.cat([past_mask, mask], dim=1)
        else:
            mask = None

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache else None
            hidden_states, new_kv = layer(hidden_states, positions, mask, layer_kv)
            new_kv_cache.append(new_kv)

        # Output
        hidden_states = self.norm(hidden_states)

        # LM head (tied or separate)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings: use transpose of embedding matrix
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        return logits, new_kv_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.

        Args:
            input_ids: [batch, seq_len] prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop_tokens: Token IDs that stop generation

        Returns:
            output_ids: [batch, seq_len + generated] full sequence
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if stop_tokens is None:
            stop_tokens = []

        # Initial forward pass (prefill)
        logits, kv_cache = self.forward(input_ids)

        # Sample first token
        next_token = self._sample(logits[:, -1, :], temperature, top_p, top_k)
        output_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            # Check stop condition
            if next_token.item() in stop_tokens:
                break

            # Single token forward with KV cache
            logits, kv_cache = self.forward(next_token.unsqueeze(1), kv_cache=kv_cache)

            # Sample
            next_token = self._sample(logits[:, -1, :], temperature, top_p, top_k)
            output_ids = torch.cat([output_ids, next_token.unsqueeze(1)], dim=1)

        return output_ids

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample from logits with temperature, top-p, top-k."""
        if temperature == 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(
    path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> GGUFModel:
    """Load a GGUF model."""
    return GGUFModel.from_gguf(path, device, dtype)


def generate(
    model: GGUFModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    **kwargs,
) -> torch.Tensor:
    """Generate from a model."""
    return model.generate(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
