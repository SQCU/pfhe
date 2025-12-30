"""
Streaming GGUF → CUDA Tensor Loading

Streams quantized data directly to GPU with on-device dequantization.
No RAM staging - reads file chunks → GPU → dequant → store.

Key design:
- Use existing GGUFReader for correct header parsing
- Small chunk reads → immediate GPU transfer
- Vectorized PyTorch ops for dequantization on GPU
- Tensors stored directly in model, no intermediate buffers
"""

import mmap
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator, Any
from dataclasses import dataclass
import logging

from .gguf_loader import GGUFReader, GGUFTensorInfo, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0

logger = logging.getLogger(__name__)


@dataclass
class TensorMeta:
    """Minimal tensor metadata for streaming."""
    shape: Tuple[int, ...]
    dtype: int
    offset: int
    nbytes: int


def _read_gguf_header(mm) -> Tuple[Dict[str, Any], Dict[str, TensorMeta], int]:
    """
    Parse GGUF header using GGUFReader, return metadata needed for streaming.

    We still need the mmap for direct byte access, but use GGUFReader for
    correct header parsing.
    """
    # This is a workaround - we need to get a path from the mmap
    # Since we can't easily get the path, we'll parse inline
    # But for now, just use a temporary approach with GGUFReader
    raise NotImplementedError("Use StreamingGGUFLoader.open() with path instead")

# Block sizes for dequantization
QK4_0 = 32
QK8_0 = 32


def dequant_q8_0_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize Q8_0 data on GPU.

    Block structure (34 bytes for 32 elements):
    - d: float16 scale (2 bytes)
    - qs: 32 x int8 values
    """
    n_blocks = numel // QK8_0
    bytes_per_block = 2 + QK8_0  # 34

    # Ensure raw is on GPU
    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)

    # Reshape to blocks
    raw = raw[:n_blocks * bytes_per_block].view(n_blocks, bytes_per_block)

    # Extract scale (first 2 bytes as fp16) - squeeze to 1D
    scale_bytes = raw[:, :2].contiguous()
    scales = scale_bytes.view(torch.float16).squeeze(-1).to(torch.float32)  # [n_blocks]

    # Extract quantized values (remaining 32 bytes as int8)
    qs = raw[:, 2:].contiguous().view(torch.int8).to(torch.float32)  # [n_blocks, 32]

    # Dequantize: value = q * scale
    result = qs * scales.unsqueeze(1)  # [n_blocks, 32]

    return result.view(-1)[:numel].to(torch.float16)


def dequant_q4_0_cuda(raw: torch.Tensor, numel: int, device: str) -> torch.Tensor:
    """
    Dequantize Q4_0 data on GPU.

    Block structure (18 bytes for 32 elements):
    - d: float16 scale (2 bytes)
    - qs: 16 bytes of packed nibbles (32 x 4-bit values)
    """
    n_blocks = numel // QK4_0
    bytes_per_block = 2 + QK4_0 // 2  # 18

    if raw.device.type != 'cuda':
        raw = raw.to(device, non_blocking=True)

    raw = raw[:n_blocks * bytes_per_block].view(n_blocks, bytes_per_block)

    # Extract scale - squeeze to 1D
    scale_bytes = raw[:, :2].contiguous()
    scales = scale_bytes.view(torch.float16).squeeze(-1).to(torch.float32)  # [n_blocks]

    # Extract packed nibbles
    qs = raw[:, 2:].contiguous()  # [n_blocks, 16]

    # Unpack nibbles to int8
    low = (qs & 0x0F).to(torch.int16) - 8  # Low nibbles
    high = ((qs >> 4) & 0x0F).to(torch.int16) - 8  # High nibbles

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low, high], dim=2).view(n_blocks, QK4_0).to(torch.float32)

    # Dequantize
    result = unpacked * scales.unsqueeze(1)  # [n_blocks, 32]

    return result.view(-1)[:numel].to(torch.float16)


class StreamingGGUFLoader:
    """
    Streams GGUF tensors directly to GPU with on-device dequantization.

    Usage:
        loader = StreamingGGUFLoader(path, device='cuda')
        for name, tensor in loader.stream_tensors():
            model.load_tensor(name, tensor)
    """

    def __init__(self, path: str, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        self.path = Path(path)
        self.device = device
        self.dtype = dtype
        self._mm = None
        self._tensor_metas = None
        self._data_offset = None

    def open(self):
        """Open file and parse header using GGUFReader."""
        self._file = open(self.path, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Use GGUFReader for correct header parsing
        self._reader = GGUFReader(self.path)
        self._reader.__enter__()

        # Convert to TensorMeta format
        self._tensor_metas = {}
        for name, info in self._reader.tensors.items():
            self._tensor_metas[name] = TensorMeta(
                shape=info.shape,
                dtype=info.dtype,
                offset=info.offset,
                nbytes=info.nbytes,
            )
        self._data_offset = self._reader.header.tensor_data_offset
        logger.info(f"Opened {self.path}: {len(self._tensor_metas)} tensors")

    def close(self):
        """Close file."""
        if hasattr(self, '_reader') and self._reader:
            self._reader.__exit__(None, None, None)
            self._reader = None
        if self._mm:
            self._mm.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def tensor_names(self):
        return list(self._tensor_metas.keys())

    def stream_tensor(self, name: str) -> torch.Tensor:
        """
        Stream a single tensor directly to GPU.

        Reads raw bytes, transfers to GPU, dequantizes on device.
        """
        meta = self._tensor_metas[name]
        offset = self._data_offset + meta.offset

        # Compute numel from shape
        numel = 1
        for d in meta.shape:
            numel *= d

        # Read raw bytes (this is fast - kernel page fault handles it)
        raw_bytes = self._mm[offset:offset + meta.nbytes]

        # Transfer to GPU as ByteTensor (non-blocking)
        raw_gpu = torch.frombuffer(raw_bytes, dtype=torch.uint8).to(
            self.device, non_blocking=True
        )

        # Dequantize on GPU based on dtype
        if meta.dtype == GGML_TYPE_F32:
            tensor = raw_gpu.view(torch.float32)
        elif meta.dtype == GGML_TYPE_F16:
            tensor = raw_gpu.view(torch.float16)
        elif meta.dtype == GGML_TYPE_BF16:
            tensor = raw_gpu.view(torch.bfloat16).to(torch.float16)
        elif meta.dtype == GGML_TYPE_Q8_0:
            tensor = dequant_q8_0_cuda(raw_gpu, numel, self.device)
        elif meta.dtype == GGML_TYPE_Q4_0:
            tensor = dequant_q4_0_cuda(raw_gpu, numel, self.device)
        else:
            raise ValueError(f"Unsupported dtype {meta.dtype} for tensor {name}")

        # Reshape and convert
        tensor = tensor.view(meta.shape)
        if self.dtype != tensor.dtype:
            tensor = tensor.to(self.dtype)

        return tensor

    def stream_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream all tensors, yielding (name, tensor) pairs."""
        for name in self._tensor_metas:
            try:
                tensor = self.stream_tensor(name)
                yield name, tensor
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    def stream_tensor_raw(self, name: str) -> torch.Tensor:
        """
        Stream raw quantized bytes directly to GPU WITHOUT dequantization.

        Returns the raw GGUF block data on GPU for use with quantized kernels.
        """
        meta = self._tensor_metas[name]
        offset = self._data_offset + meta.offset
        raw_bytes = self._mm[offset:offset + meta.nbytes]
        return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(self.device)

    def get_tensor_meta(self, name: str) -> TensorMeta:
        """Get metadata for a tensor."""
        return self._tensor_metas[name]


class QuantizedTensor:
    """
    Holds quantized GGUF blocks on GPU.

    Dequantizes on-the-fly during operations, keeping VRAM usage ~4x lower.
    This mirrors how efficient implementations (llama.cpp CUDA, exllama) work.
    """

    def __init__(
        self,
        raw_data: torch.Tensor,
        shape: Tuple[int, ...],
        quant_type: int,
        device: str = 'cuda',
    ):
        """
        Args:
            raw_data: Raw quantized bytes on GPU
            shape: Logical tensor shape (dequantized)
            quant_type: GGML quantization type (Q4_0, Q8_0, etc.)
        """
        self.raw_data = raw_data
        self.shape = shape
        self.quant_type = quant_type
        self.device = device

        self.numel = 1
        for d in shape:
            self.numel *= d

    def dequantize(self) -> torch.Tensor:
        """Full dequantization - use sparingly, prefer fused ops."""
        if self.quant_type == GGML_TYPE_Q8_0:
            return dequant_q8_0_cuda(self.raw_data, self.numel, self.device).view(self.shape)
        elif self.quant_type == GGML_TYPE_Q4_0:
            return dequant_q4_0_cuda(self.raw_data, self.numel, self.device).view(self.shape)
        elif self.quant_type == GGML_TYPE_F16:
            return self.raw_data.view(torch.float16).view(self.shape)
        elif self.quant_type == GGML_TYPE_F32:
            return self.raw_data.view(torch.float32).view(self.shape)
        else:
            raise ValueError(f"Unsupported quant type: {self.quant_type}")

    @property
    def nbytes(self) -> int:
        return self.raw_data.numel()

    @property
    def compression_ratio(self) -> float:
        """Ratio of quantized size to fp16 size."""
        fp16_bytes = self.numel * 2
        return fp16_bytes / self.nbytes


def quantized_matmul_q8_0(
    x: torch.Tensor,
    w_quant: torch.Tensor,
    w_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Fused quantized matmul: x @ W^T where W is Q8_0 quantized.

    Dequantizes W on-the-fly during matmul computation.
    For true efficiency, this should be a custom CUDA kernel that
    dequantizes within the matmul tile loop. This version dequantizes
    first but keeps weights in quantized form in VRAM.

    Args:
        x: Input activations [batch, seq, hidden] or [batch, hidden]
        w_quant: Quantized weight bytes on GPU
        w_shape: Logical weight shape (out_features, in_features)

    Returns:
        x @ W^T with shape [..., out_features]
    """
    out_features, in_features = w_shape
    numel = out_features * in_features

    # Dequantize weight (TODO: fuse into kernel)
    w = dequant_q8_0_cuda(w_quant, numel, str(x.device))
    w = w.view(out_features, in_features)

    # Standard matmul
    return torch.nn.functional.linear(x.to(w.dtype), w)


def quantized_matmul_q4_0(
    x: torch.Tensor,
    w_quant: torch.Tensor,
    w_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Fused quantized matmul: x @ W^T where W is Q4_0 quantized.
    """
    out_features, in_features = w_shape
    numel = out_features * in_features

    # Dequantize weight (TODO: fuse into kernel)
    w = dequant_q4_0_cuda(w_quant, numel, str(x.device))
    w = w.view(out_features, in_features)

    return torch.nn.functional.linear(x.to(w.dtype), w)


class QuantizedLinear(torch.nn.Module):
    """
    Linear layer with quantized weights stored on GPU.

    Weights are kept in quantized form, dequantized during forward pass.
    This uses ~4x less VRAM than FP16 weights for Q4_0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        raw_weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type

        # Store quantized weight as buffer (not parameter - saves memory)
        self.register_buffer('weight_quant', raw_weight)

        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (self.out_features, self.in_features)

        if self.quant_type == GGML_TYPE_Q8_0:
            out = quantized_matmul_q8_0(x, self.weight_quant, shape)
        elif self.quant_type == GGML_TYPE_Q4_0:
            out = quantized_matmul_q4_0(x, self.weight_quant, shape)
        else:
            raise ValueError(f"Unsupported quant type: {self.quant_type}")

        if self.bias is not None:
            out = out + self.bias

        return out

    @property
    def weight_bytes(self) -> int:
        return self.weight_quant.numel()

    @property
    def equivalent_fp16_bytes(self) -> int:
        return self.in_features * self.out_features * 2


def test_streaming_load(model_path: str, device: str = 'cuda'):
    """Test streaming GGUF loading."""
    import time

    print(f"Testing streaming load: {model_path}")

    with StreamingGGUFLoader(model_path, device=device) as loader:
        print(f"Found {len(loader.tensor_names)} tensors")

        total_bytes = 0
        total_tensors = 0
        start = time.perf_counter()

        for name, tensor in loader.stream_tensors():
            total_bytes += tensor.numel() * tensor.element_size()
            total_tensors += 1

            if total_tensors <= 5:
                print(f"  {name}: {tensor.shape} {tensor.dtype} "
                      f"mean={tensor.float().mean():.4f}")

            # Free tensor immediately to avoid GPU memory buildup
            del tensor

        elapsed = time.perf_counter() - start
        print(f"\nLoaded {total_tensors} tensors, "
              f"{total_bytes/1e9:.2f}GB in {elapsed:.1f}s "
              f"({total_bytes/elapsed/1e9:.2f} GB/s)")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/f/dox/ai/text/models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    test_streaming_load(path)
