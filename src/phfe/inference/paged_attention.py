"""
Paged Attention Primitives for Causal LM Inference

Adapted from logsnrcat's PageTable and KVTManager for pure text/causal LM.
Uses PyTorch's native flex_attention for efficient batched inference.

Key concepts:
- PageTable: Maps logical sequence positions to physical KV cache slots
- BlockManager: Handles block allocation with prefix caching via content hashing
- KVCacheManager: Holds the actual KV tensors and topology
- CausalBlockMask: Generates block masks for flex_attention

Usage:
    cache = KVCacheManager(num_blocks=1024, block_size=64, ...)

    # Allocate sequence (with prefix caching)
    cache.allocate_sequence(req_id=0, token_ids=[1, 2, 3, ...])

    # Get attention inputs for a batch
    inputs = cache.get_attention_inputs([req_id_0, req_id_1, ...], layer_idx=0)

    # Run attention with flex_attention
    out = flex_attention(q, inputs['k_cache'], inputs['v_cache'],
                         block_mask=inputs['block_mask'])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import xxhash
import numpy as np
import copy
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Block Manager (Prefix Caching via Content Hashing)
# =============================================================================

class Block:
    """A single block in the KV cache."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.content_hash = -1  # Hash of token content for prefix matching

    def link(self, content_hash: int):
        """Link this block to a specific content hash."""
        self.content_hash = content_hash

    def reset(self):
        """Reset block for reuse."""
        self.ref_count = 0
        self.content_hash = -1


class BlockManager:
    """
    Manages block allocation with prefix caching.

    When a sequence is allocated, blocks with matching content hashes
    are reused (prefix caching). This avoids recomputing KV for
    shared prefixes like system prompts.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block: Dict[int, int] = {}  # content_hash -> block_id
        self.free_blocks: deque = deque(range(num_blocks))

    def _allocate_block(self) -> Block:
        """Allocate a fresh block."""
        if not self.free_blocks:
            raise RuntimeError("KV cache OOM: no free blocks")
        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        block.reset()
        block.ref_count = 1
        return block

    def _free_block(self, block_id: int):
        """Return block to free pool."""
        block = self.blocks[block_id]
        if block.content_hash != -1:
            if self.hash_to_block.get(block.content_hash) == block_id:
                del self.hash_to_block[block.content_hash]
        block.reset()
        self.free_blocks.append(block_id)

    @staticmethod
    def compute_block_hash(token_ids: List[int], prefix_hash: int = -1) -> int:
        """
        Compute content hash for a block of tokens.

        The hash depends on:
        1. The token IDs in this block
        2. The hash of the previous block (chain hashing)

        This ensures that identical token sequences map to the same hash,
        enabling prefix caching.
        """
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, 'little'))
        token_bytes = np.array(token_ids, dtype=np.int64).tobytes()
        h.update(token_bytes)
        return h.intdigest()

    def allocate(self, token_ids: List[int]) -> Tuple[List[int], List[int], int]:
        """
        Allocate blocks for a token sequence.

        Returns:
            block_table: List of physical block IDs
            new_blocks: Block IDs that need KV computation (cache miss)
            cache_hits: Number of blocks reused from prefix cache
        """
        num_tokens = len(token_ids)
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        block_table = []
        new_blocks = []
        prefix_hash = -1
        cache_hits = 0

        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, num_tokens)
            chunk = token_ids[start:end]
            is_full = len(chunk) == self.block_size

            # Compute hash for full blocks only
            if is_full:
                current_hash = self.compute_block_hash(chunk, prefix_hash)
            else:
                current_hash = -1  # Partial blocks can't be cached

            # Check for cache hit
            cached_block_id = -1
            if current_hash != -1:
                cached_block_id = self.hash_to_block.get(current_hash, -1)

            if cached_block_id != -1:
                # Cache hit - reuse existing block
                self.blocks[cached_block_id].ref_count += 1
                block_table.append(cached_block_id)
                prefix_hash = current_hash
                cache_hits += 1
            else:
                # Cache miss - allocate new block
                block = self._allocate_block()
                if is_full:
                    block.link(current_hash)
                    self.hash_to_block[current_hash] = block.block_id
                    prefix_hash = current_hash
                else:
                    prefix_hash = -1
                block_table.append(block.block_id)
                new_blocks.append(block.block_id)

        return block_table, new_blocks, cache_hits

    def free(self, block_table: List[int]):
        """Release blocks, freeing those with ref_count=0."""
        for block_id in block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)


# =============================================================================
# Page Table (Logical to Physical Mapping)
# =============================================================================

class PageTable:
    """
    Maps logical sequence positions to physical KV cache slots.

    This enables paged attention where:
    - Logical: Contiguous sequence positions (0, 1, 2, ...)
    - Physical: Scattered blocks in the KV cache heap

    The page table is used by flex_attention's block_mask to translate
    logical attention patterns to physical memory access patterns.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        max_batch_size: int,
        max_seq_blocks: int,
        device: str = "cuda",
    ):
        self.block_size = block_size
        self.device = device

        # [batch_idx, logical_block] -> physical_block
        self.logical_to_physical = torch.full(
            (max_batch_size, max_seq_blocks), -1,
            dtype=torch.int32, device=device
        )

        # [batch_idx, physical_block] -> logical_block (for mask_mod)
        self.physical_to_logical = torch.full(
            (max_batch_size, num_blocks), -1,
            dtype=torch.int32, device=device
        )

    def set_mapping(self, batch_idx: int, block_table: List[int]):
        """Set page table mapping for a sequence."""
        for log_idx, phys_idx in enumerate(block_table):
            self.logical_to_physical[batch_idx, log_idx] = phys_idx
            self.physical_to_logical[batch_idx, phys_idx] = log_idx

    def clear_mapping(self, batch_idx: int):
        """Clear page table for a sequence."""
        self.logical_to_physical[batch_idx].fill_(-1)
        self.physical_to_logical[batch_idx].fill_(-1)

    def convert_to_physical_mask(
        self,
        logical_mask: BlockMask,
        batch_indices: torch.Tensor,
    ) -> BlockMask:
        """
        Convert a logical BlockMask to physical space.

        Args:
            logical_mask: Mask defined on logical sequence positions
            batch_indices: [B] tensor mapping kernel batch idx to request ID

        Returns:
            Physical BlockMask for paged KV cache access
        """
        B = batch_indices.size(0)

        # Get logical indices from mask
        kv_indices = logical_mask.kv_indices
        full_kv_indices = logical_mask.full_kv_indices

        # Expand if mask is shared across batch
        if kv_indices.size(0) == 1 and B > 1:
            kv_indices = kv_indices.expand(B, -1, -1, -1)
            full_kv_indices = full_kv_indices.expand(B, -1, -1, -1)

        # Gather physical block IDs
        active_tables = self.logical_to_physical[batch_indices.long()]
        pt_view = active_tables.view(B, 1, 1, -1)

        phys_kv_indices = torch.gather(
            pt_view.expand(-1, kv_indices.size(1), kv_indices.size(2), -1),
            3, kv_indices.long()
        )
        phys_full_kv_indices = torch.gather(
            pt_view.expand(-1, full_kv_indices.size(1), full_kv_indices.size(2), -1),
            3, full_kv_indices.long()
        )

        # Wrap mask_mod to translate physical -> logical for condition checks
        original_mod = logical_mask.mask_mod
        phys_to_log = self.physical_to_logical
        block_size = self.block_size

        def physical_mask_mod(b, h, q_idx, k_phys_idx):
            req_id = batch_indices[b]
            phys_block = k_phys_idx // block_size
            offset = k_phys_idx % block_size
            log_block = phys_to_log[req_id, phys_block]
            log_k_idx = log_block * block_size + offset
            return original_mod(b, h, q_idx, log_k_idx)

        # Create new mask with physical indices
        physical_mask = copy.copy(logical_mask)
        physical_mask.kv_indices = phys_kv_indices.int()
        physical_mask.full_kv_indices = phys_full_kv_indices.int()
        physical_mask.mask_mod = physical_mask_mod

        return physical_mask


# =============================================================================
# KV Cache Manager
# =============================================================================

class KVCacheManager:
    """
    Manages paged KV cache with prefix caching for causal LM inference.

    Features:
    - Paged memory: Non-contiguous block allocation
    - Prefix caching: Reuse KV for shared prefixes
    - Batch support: Multiple sequences with different lengths
    - flex_attention integration: Generates BlockMasks for efficient attention
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Block management
        self.block_manager = BlockManager(num_blocks, block_size)

        # Page table
        max_seq_blocks = num_blocks  # Conservative: any seq could use all blocks
        self.page_table = PageTable(
            num_blocks, block_size, max_batch_size, max_seq_blocks, device
        )

        # KV cache tensors: [layers, heads, num_blocks * block_size, head_dim]
        capacity = num_blocks * block_size
        self.k_cache = torch.zeros(
            (num_layers, num_heads, capacity, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, num_heads, capacity, head_dim),
            dtype=dtype, device=device
        )

        # Request tracking
        self.req_block_tables: Dict[int, List[int]] = {}
        self.req_lengths: Dict[int, int] = {}
        self.req_token_ids: Dict[int, List[int]] = {}

        # Stats
        self.total_cache_hits = 0
        self.total_allocations = 0

    def allocate_sequence(
        self,
        req_id: int,
        token_ids: List[int],
    ) -> Tuple[List[int], int]:
        """
        Allocate KV cache blocks for a new sequence.

        Returns:
            new_blocks: Block IDs that need KV computation
            cache_hits: Number of blocks reused from prefix cache
        """
        block_table, new_blocks, cache_hits = self.block_manager.allocate(token_ids)

        # Update tracking
        self.req_block_tables[req_id] = block_table
        self.req_lengths[req_id] = len(token_ids)
        self.req_token_ids[req_id] = list(token_ids)

        # Update page table
        batch_idx = req_id % self.page_table.logical_to_physical.size(0)
        self.page_table.set_mapping(batch_idx, block_table)

        # Stats
        self.total_cache_hits += cache_hits
        self.total_allocations += len(block_table)

        return new_blocks, cache_hits

    def extend_sequence(
        self,
        req_id: int,
        new_token_ids: List[int],
    ) -> List[int]:
        """
        Extend an existing sequence with new tokens.

        Returns:
            new_blocks: Block IDs that need KV computation
        """
        if req_id not in self.req_token_ids:
            raise KeyError(f"Request {req_id} not found")

        old_tokens = self.req_token_ids[req_id]
        full_tokens = old_tokens + new_token_ids

        # Re-allocate with full sequence (prefix caching handles reuse)
        old_table = self.req_block_tables[req_id]
        self.block_manager.free(old_table)

        block_table, new_blocks, _ = self.block_manager.allocate(full_tokens)

        # Update tracking
        self.req_block_tables[req_id] = block_table
        self.req_lengths[req_id] = len(full_tokens)
        self.req_token_ids[req_id] = full_tokens

        # Update page table
        batch_idx = req_id % self.page_table.logical_to_physical.size(0)
        self.page_table.set_mapping(batch_idx, block_table)

        return new_blocks

    def free_sequence(self, req_id: int):
        """Free all resources for a sequence."""
        if req_id not in self.req_block_tables:
            return

        block_table = self.req_block_tables[req_id]
        self.block_manager.free(block_table)

        batch_idx = req_id % self.page_table.logical_to_physical.size(0)
        self.page_table.clear_mapping(batch_idx)

        del self.req_block_tables[req_id]
        del self.req_lengths[req_id]
        del self.req_token_ids[req_id]

    def get_slot_mapping(self, req_ids: List[int]) -> torch.Tensor:
        """
        Get physical slot indices for all tokens in the batch.

        Used by update_kv_cache to scatter new KV values into the cache.
        """
        slots = []
        for req_id in req_ids:
            block_table = self.req_block_tables[req_id]
            length = self.req_lengths[req_id]

            for token_idx in range(length):
                block_idx = token_idx // self.block_size
                offset = token_idx % self.block_size
                physical_block = block_table[block_idx]
                slot = physical_block * self.block_size + offset
                slots.append(slot)

        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache tensors for a layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def update_kv_cache(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """
        Write new KV values into the cache.

        Args:
            k_new: [batch_tokens, num_heads, head_dim]
            v_new: [batch_tokens, num_heads, head_dim]
            slot_mapping: [batch_tokens] physical slot indices
        """
        # Transpose to [num_heads, batch_tokens, head_dim] for indexing
        k_src = k_new.transpose(0, 1)
        v_src = v_new.transpose(0, 1)

        # Scatter into cache
        # k_cache[layer]: [num_heads, capacity, head_dim]
        self.k_cache[layer_idx, :, slot_mapping, :] = k_src.to(self.dtype)
        self.v_cache[layer_idx, :, slot_mapping, :] = v_src.to(self.dtype)

    def build_causal_mask(
        self,
        req_ids: List[int],
        q_len: Optional[int] = None,
    ) -> BlockMask:
        """
        Build a causal attention BlockMask for the batch.

        Args:
            req_ids: List of request IDs in the batch
            q_len: Query length (if None, uses full sequence)

        Returns:
            BlockMask for flex_attention
        """
        # Calculate total KV length
        kv_len = sum(self.req_lengths[rid] for rid in req_ids)
        if q_len is None:
            q_len = kv_len

        device = self.device

        # Build sequence boundaries for batched masking
        seq_starts = []
        seq_ends = []
        cursor = 0
        for req_id in req_ids:
            length = self.req_lengths[req_id]
            seq_starts.append(cursor)
            seq_ends.append(cursor + length)
            cursor += length

        seq_starts_t = torch.tensor(seq_starts, device=device)
        seq_ends_t = torch.tensor(seq_ends, device=device)

        def causal_mask_mod(b, h, q_idx, kv_idx):
            # Find which sequence this position belongs to
            # For simplicity, assume flattened batch (b=0)
            # Causal: q can attend to kv if kv <= q
            return kv_idx <= q_idx

        mask = create_block_mask(
            causal_mask_mod,
            B=None,  # Broadcast across batch
            H=None,  # Broadcast across heads
            Q_LEN=q_len,
            KV_LEN=kv_len,
        )

        return mask

    @property
    def cache_hit_rate(self) -> float:
        if self.total_allocations == 0:
            return 0.0
        return self.total_cache_hits / self.total_allocations

    def stats(self) -> Dict[str, Any]:
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": self.block_manager.num_used_blocks,
            "free_blocks": self.block_manager.num_free_blocks,
            "active_sequences": len(self.req_block_tables),
            "cache_hit_rate": f"{self.cache_hit_rate:.2%}",
            "total_cache_hits": self.total_cache_hits,
            "total_allocations": self.total_allocations,
        }


# =============================================================================
# RoPE (Rotary Position Embeddings)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Standard RoPE for causal LM."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.

        Args:
            q, k: [batch, heads, seq_len, head_dim]
            positions: [seq_len] or [batch, seq_len] position indices
        """
        seq_len = q.shape[2]

        if positions.dim() == 1:
            cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[positions].unsqueeze(1)
            sin = self.sin_cached[positions].unsqueeze(1)

        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to tensor."""
        # Split into halves
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]

        # Rotate
        rotated = torch.cat([-x2, x1], dim=-1)

        return (x * cos) + (rotated * sin)


# =============================================================================
# Convenience: Causal LM Attention Layer
# =============================================================================

class PagedCausalAttention(nn.Module):
    """
    Causal attention layer with paged KV cache support.

    Uses flex_attention for efficient batched inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_mask: Optional[BlockMask] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional KV caching.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            positions: [seq_len] position indices
            kv_cache: (k_cache, v_cache) from KVCacheManager
            block_mask: BlockMask for attention
            slot_mapping: Physical slots for KV update

        Returns:
            output: [batch, seq_len, hidden_dim]
            new_kv: (k, v) to update cache
        """
        batch, seq_len, _ = hidden_states.shape

        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, positions)

        # Attention
        if kv_cache is not None:
            # Use cached KV for full sequence attention
            k_cache, v_cache = kv_cache
            # k_cache: [heads, capacity, head_dim]
            # Expand for batch: [batch, heads, capacity, head_dim]
            k_full = k_cache.unsqueeze(0).expand(batch, -1, -1, -1)
            v_full = v_cache.unsqueeze(0).expand(batch, -1, -1, -1)

            attn_out = flex_attention(q, k_full, v_full, block_mask=block_mask)
        else:
            # Standard attention without cache
            attn_out = flex_attention(q, k, v, block_mask=block_mask)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_out)

        # Return new KV for cache update
        # Flatten for slot_mapping: [batch*seq, heads, head_dim]
        k_new = k.transpose(1, 2).contiguous().view(batch * seq_len, self.num_heads, self.head_dim)
        v_new = v.transpose(1, 2).contiguous().view(batch * seq_len, self.num_heads, self.head_dim)

        return output, (k_new, v_new)


# =============================================================================
# Utility Functions
# =============================================================================

def create_causal_mask(seq_len: int) -> BlockMask:
    """Create a simple causal attention mask."""
    def causal_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx

    return create_block_mask(
        causal_mod,
        B=None, H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


def estimate_kv_cache_memory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Estimate KV cache memory usage."""
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    capacity = num_blocks * block_size

    # K and V caches
    kv_bytes = 2 * num_layers * num_heads * capacity * head_dim * bytes_per_element

    return {
        "kv_cache_gb": kv_bytes / 1024**3,
        "kv_cache_mb": kv_bytes / 1024**2,
        "capacity_tokens": capacity,
        "num_blocks": num_blocks,
    }
