"""
GGUF Tensor Loader

Pure Python GGUF file parser and dequantizer.
No vLLM, no llama-cpp-python - just struct parsing and torch ops.

References:
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Q4_0 block structure: 32 elements, fp16 scale + 16 bytes of nibbles
"""

import struct
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# GGUF Constants
# =============================================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGML Types (from ggml.h)
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
GGML_TYPE_BF16 = 30

# Block sizes for quantized types
QK4_0 = 32  # Elements per block for Q4_0
QK4_1 = 32
QK5_0 = 32
QK5_1 = 32
QK8_0 = 32
QK8_1 = 32

# Type info: (block_size, bytes_per_block)
QUANT_INFO = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_BF16: (1, 2),
    GGML_TYPE_Q4_0: (QK4_0, 2 + QK4_0 // 2),  # fp16 scale + 16 bytes nibbles
    GGML_TYPE_Q4_1: (QK4_1, 2 + 2 + QK4_1 // 2),  # fp16 scale + fp16 min + nibbles
    GGML_TYPE_Q8_0: (QK8_0, 2 + QK8_0),  # fp16 scale + 32 bytes
}

GGUF_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_BF16: "BF16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
}

# Metadata value types
GGUF_METADATA_VALUE_TYPE_UINT8 = 0
GGUF_METADATA_VALUE_TYPE_INT8 = 1
GGUF_METADATA_VALUE_TYPE_UINT16 = 2
GGUF_METADATA_VALUE_TYPE_INT16 = 3
GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_INT32 = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL = 7
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9
GGUF_METADATA_VALUE_TYPE_UINT64 = 10
GGUF_METADATA_VALUE_TYPE_INT64 = 11
GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GGUFTensorInfo:
    """Metadata for a single tensor in the GGUF file."""
    name: str
    n_dims: int
    dims: Tuple[int, ...]
    dtype: int  # GGML type
    offset: int  # Offset from start of tensor data section

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def shape(self) -> Tuple[int, ...]:
        """PyTorch-style shape (reversed from GGML)."""
        return tuple(reversed(self.dims))

    @property
    def dtype_name(self) -> str:
        return GGUF_TYPE_NAMES.get(self.dtype, f"UNKNOWN({self.dtype})")

    @property
    def nbytes(self) -> int:
        """Calculate byte size based on quantization type."""
        if self.dtype not in QUANT_INFO:
            raise ValueError(f"Unknown dtype {self.dtype} for size calculation")

        block_size, bytes_per_block = QUANT_INFO[self.dtype]
        n_blocks = (self.numel + block_size - 1) // block_size
        return n_blocks * bytes_per_block


@dataclass
class GGUFHeader:
    """GGUF file header."""
    magic: int
    version: int
    n_tensors: int
    n_kv: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, GGUFTensorInfo] = field(default_factory=dict)
    tensor_data_offset: int = 0


# =============================================================================
# GGUF Parser
# =============================================================================

class GGUFReader:
    """
    Memory-mapped GGUF file reader.

    Usage:
        with GGUFReader("model.gguf") as reader:
            tensor = reader.read_tensor("model.layers.0.self_attn.q_proj.weight")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self._header: Optional[GGUFHeader] = None
        self._cursor = 0

    def __enter__(self) -> "GGUFReader":
        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()
        return self

    def __exit__(self, *args):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    @property
    def header(self) -> GGUFHeader:
        if self._header is None:
            raise RuntimeError("File not opened - use context manager")
        return self._header

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.header.metadata

    @property
    def tensors(self) -> Dict[str, GGUFTensorInfo]:
        return self.header.tensors

    def _read_bytes(self, n: int) -> bytes:
        """Read n bytes from current position."""
        data = self._mmap[self._cursor:self._cursor + n]
        self._cursor += n
        return data

    def _read_u8(self) -> int:
        return struct.unpack("<B", self._read_bytes(1))[0]

    def _read_i8(self) -> int:
        return struct.unpack("<b", self._read_bytes(1))[0]

    def _read_u16(self) -> int:
        return struct.unpack("<H", self._read_bytes(2))[0]

    def _read_i16(self) -> int:
        return struct.unpack("<h", self._read_bytes(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._read_bytes(4))[0]

    def _read_i32(self) -> int:
        return struct.unpack("<i", self._read_bytes(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack("<Q", self._read_bytes(8))[0]

    def _read_i64(self) -> int:
        return struct.unpack("<q", self._read_bytes(8))[0]

    def _read_f32(self) -> float:
        return struct.unpack("<f", self._read_bytes(4))[0]

    def _read_f64(self) -> float:
        return struct.unpack("<d", self._read_bytes(8))[0]

    def _read_string(self) -> str:
        """Read length-prefixed string."""
        length = self._read_u64()
        data = self._read_bytes(length)
        return data.decode("utf-8")

    def _read_metadata_value(self, value_type: int) -> Any:
        """Read a metadata value based on its type."""
        if value_type == GGUF_METADATA_VALUE_TYPE_UINT8:
            return self._read_u8()
        elif value_type == GGUF_METADATA_VALUE_TYPE_INT8:
            return self._read_i8()
        elif value_type == GGUF_METADATA_VALUE_TYPE_UINT16:
            return self._read_u16()
        elif value_type == GGUF_METADATA_VALUE_TYPE_INT16:
            return self._read_i16()
        elif value_type == GGUF_METADATA_VALUE_TYPE_UINT32:
            return self._read_u32()
        elif value_type == GGUF_METADATA_VALUE_TYPE_INT32:
            return self._read_i32()
        elif value_type == GGUF_METADATA_VALUE_TYPE_UINT64:
            return self._read_u64()
        elif value_type == GGUF_METADATA_VALUE_TYPE_INT64:
            return self._read_i64()
        elif value_type == GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return self._read_f32()
        elif value_type == GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return self._read_f64()
        elif value_type == GGUF_METADATA_VALUE_TYPE_BOOL:
            return self._read_u8() != 0
        elif value_type == GGUF_METADATA_VALUE_TYPE_STRING:
            return self._read_string()
        elif value_type == GGUF_METADATA_VALUE_TYPE_ARRAY:
            arr_type = self._read_u32()
            arr_len = self._read_u64()
            return [self._read_metadata_value(arr_type) for _ in range(arr_len)]
        else:
            raise ValueError(f"Unknown metadata value type: {value_type}")

    def _parse_header(self):
        """Parse GGUF header, metadata, and tensor info."""
        self._cursor = 0

        # Magic and version
        magic = self._read_u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

        version = self._read_u32()
        if version < 2 or version > 3:
            raise ValueError(f"Unsupported GGUF version: {version}")

        n_tensors = self._read_u64()
        n_kv = self._read_u64()

        self._header = GGUFHeader(
            magic=magic,
            version=version,
            n_tensors=n_tensors,
            n_kv=n_kv,
        )

        # Parse metadata
        for _ in range(n_kv):
            key = self._read_string()
            value_type = self._read_u32()
            value = self._read_metadata_value(value_type)
            self._header.metadata[key] = value

        # Parse tensor info
        for _ in range(n_tensors):
            name = self._read_string()
            n_dims = self._read_u32()
            dims = tuple(self._read_u64() for _ in range(n_dims))
            dtype = self._read_u32()
            offset = self._read_u64()

            self._header.tensors[name] = GGUFTensorInfo(
                name=name,
                n_dims=n_dims,
                dims=dims,
                dtype=dtype,
                offset=offset,
            )

        # Align to 32 bytes for tensor data
        alignment = 32
        self._header.tensor_data_offset = (self._cursor + alignment - 1) // alignment * alignment

    def read_tensor_raw(self, name: str) -> bytes:
        """Read raw tensor bytes without dequantization."""
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")

        info = self.tensors[name]
        start = self._header.tensor_data_offset + info.offset
        end = start + info.nbytes
        return bytes(self._mmap[start:end])

    def read_tensor(
        self,
        name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Read and dequantize a tensor.

        Args:
            name: Tensor name in the GGUF file
            device: Target device
            dtype: Target dtype (dequantized output)

        Returns:
            Dequantized tensor
        """
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")

        info = self.tensors[name]
        raw_data = self.read_tensor_raw(name)

        # Dequantize based on type
        if info.dtype == GGML_TYPE_F32:
            tensor = _decode_f32(raw_data, info.numel)
        elif info.dtype == GGML_TYPE_F16:
            tensor = _decode_f16(raw_data, info.numel)
        elif info.dtype == GGML_TYPE_BF16:
            tensor = _decode_bf16(raw_data, info.numel)
        elif info.dtype == GGML_TYPE_Q4_0:
            tensor = _dequantize_q4_0(raw_data, info.numel)
        elif info.dtype == GGML_TYPE_Q4_1:
            tensor = _dequantize_q4_1(raw_data, info.numel)
        elif info.dtype == GGML_TYPE_Q8_0:
            tensor = _dequantize_q8_0(raw_data, info.numel)
        else:
            raise NotImplementedError(
                f"Dequantization not implemented for {info.dtype_name}"
            )

        # Reshape and convert
        tensor = tensor.reshape(info.shape)
        tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    def tensor_iterator(
        self,
        pattern: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over tensors, yielding (name, tensor) pairs.

        Args:
            pattern: Optional glob pattern to filter tensor names
            device: Target device
            dtype: Target dtype
        """
        import fnmatch

        for name in self.tensors:
            if pattern is not None and not fnmatch.fnmatch(name, pattern):
                continue
            yield name, self.read_tensor(name, device=device, dtype=dtype)


# =============================================================================
# Dequantization Functions
# =============================================================================

def _decode_f32(data: bytes, numel: int) -> torch.Tensor:
    """Decode F32 tensor."""
    arr = np.frombuffer(data, dtype=np.float32)[:numel]
    return torch.from_numpy(arr.copy())


def _decode_f16(data: bytes, numel: int) -> torch.Tensor:
    """Decode F16 tensor."""
    arr = np.frombuffer(data, dtype=np.float16)[:numel]
    return torch.from_numpy(arr.copy()).float()


def _decode_bf16(data: bytes, numel: int) -> torch.Tensor:
    """Decode BF16 tensor."""
    # BF16 is stored as uint16, need to reinterpret
    arr = np.frombuffer(data, dtype=np.uint16)[:numel]
    # Convert to float32 via bit manipulation
    # BF16: sign(1) + exp(8) + mantissa(7)
    # F32:  sign(1) + exp(8) + mantissa(23)
    arr32 = arr.astype(np.uint32) << 16
    return torch.from_numpy(arr32.view(np.float32).copy())


def _dequantize_q4_0(data: bytes, numel: int) -> torch.Tensor:
    """
    Dequantize Q4_0 tensor.

    Q4_0 block structure (18 bytes per 32 elements):
        - d: float16 scale factor (2 bytes)
        - qs: 16 bytes of packed 4-bit quantized values (32 nibbles)

    Dequantization: value = (nibble - 8) * d
    """
    block_size = QK4_0
    bytes_per_block = 2 + block_size // 2  # 18 bytes
    n_blocks = (numel + block_size - 1) // block_size

    expected_bytes = n_blocks * bytes_per_block
    if len(data) < expected_bytes:
        raise ValueError(f"Data too short: {len(data)} < {expected_bytes}")

    # Pre-allocate output
    output = np.zeros(n_blocks * block_size, dtype=np.float32)

    for block_idx in range(n_blocks):
        block_start = block_idx * bytes_per_block

        # Read scale (fp16)
        d = np.frombuffer(data[block_start:block_start + 2], dtype=np.float16)[0]
        d = float(d)

        # Read quantized values (16 bytes = 32 nibbles)
        qs = np.frombuffer(
            data[block_start + 2:block_start + bytes_per_block],
            dtype=np.uint8
        )

        # Unpack nibbles (vectorized)
        out_start = block_idx * block_size
        # Extract low and high nibbles as int16 to avoid overflow
        low_nibbles = (qs.astype(np.int16) & 0x0F) - 8
        high_nibbles = ((qs.astype(np.int16) >> 4) & 0x0F) - 8

        # Interleave: [l0, h0, l1, h1, ...]
        values = np.empty(block_size, dtype=np.float32)
        values[0::2] = low_nibbles * d
        values[1::2] = high_nibbles * d
        output[out_start:out_start + block_size] = values

    return torch.from_numpy(output[:numel])


def _dequantize_q4_1(data: bytes, numel: int) -> torch.Tensor:
    """
    Dequantize Q4_1 tensor.

    Q4_1 block structure (20 bytes per 32 elements):
        - d: float16 scale factor (2 bytes)
        - m: float16 minimum value (2 bytes)
        - qs: 16 bytes of packed 4-bit quantized values

    Dequantization: value = nibble * d + m
    """
    block_size = QK4_1
    bytes_per_block = 2 + 2 + block_size // 2  # 20 bytes
    n_blocks = (numel + block_size - 1) // block_size

    output = np.zeros(n_blocks * block_size, dtype=np.float32)

    for block_idx in range(n_blocks):
        block_start = block_idx * bytes_per_block

        # Read scale and min (fp16)
        d = float(np.frombuffer(data[block_start:block_start + 2], dtype=np.float16)[0])
        m = float(np.frombuffer(data[block_start + 2:block_start + 4], dtype=np.float16)[0])

        # Read quantized values
        qs = np.frombuffer(
            data[block_start + 4:block_start + bytes_per_block],
            dtype=np.uint8
        )

        # Unpack nibbles (vectorized)
        out_start = block_idx * block_size
        low_nibbles = qs.astype(np.int16) & 0x0F
        high_nibbles = (qs.astype(np.int16) >> 4) & 0x0F

        values = np.empty(block_size, dtype=np.float32)
        values[0::2] = low_nibbles * d + m
        values[1::2] = high_nibbles * d + m
        output[out_start:out_start + block_size] = values

    return torch.from_numpy(output[:numel])


def _dequantize_q8_0(data: bytes, numel: int) -> torch.Tensor:
    """
    Dequantize Q8_0 tensor.

    Q8_0 block structure (34 bytes per 32 elements):
        - d: float16 scale factor (2 bytes)
        - qs: 32 bytes of signed 8-bit quantized values

    Dequantization: value = qs * d
    """
    block_size = QK8_0
    bytes_per_block = 2 + block_size  # 34 bytes
    n_blocks = (numel + block_size - 1) // block_size

    output = np.zeros(n_blocks * block_size, dtype=np.float32)

    for block_idx in range(n_blocks):
        block_start = block_idx * bytes_per_block

        # Read scale (fp16)
        d = float(np.frombuffer(data[block_start:block_start + 2], dtype=np.float16)[0])

        # Read quantized values (signed int8)
        qs = np.frombuffer(
            data[block_start + 2:block_start + bytes_per_block],
            dtype=np.int8
        )

        # Dequantize
        out_start = block_idx * block_size
        output[out_start:out_start + block_size] = qs.astype(np.float32) * d

    return torch.from_numpy(output[:numel])


# =============================================================================
# Context Managers for Efficient Loading
# =============================================================================

@contextmanager
def tensor_loading_context(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Iterator[GGUFReader]:
    """
    Context manager for efficient tensor loading.

    Usage:
        with tensor_loading_context("model.gguf") as loader:
            weight = loader.read_tensor("model.layers.0.weight")
    """
    with GGUFReader(path) as reader:
        yield reader


@contextmanager
def streaming_dequant_context(
    path: str | Path,
    batch_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Iterator["StreamingDequantizer"]:
    """
    Context manager for streaming dequantization to GPU.

    Loads tensors in batches to minimize peak memory.

    Usage:
        with streaming_dequant_context("model.gguf", batch_size=4) as stream:
            for batch in stream.iter_batches():
                model.load_partial(batch)
    """
    streamer = StreamingDequantizer(path, batch_size, device, dtype)
    streamer.open()
    try:
        yield streamer
    finally:
        streamer.close()


class StreamingDequantizer:
    """
    Streams dequantized tensors to GPU in batches.
    Useful for large models where full dequantization exceeds VRAM.
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.path = Path(path)
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._reader: Optional[GGUFReader] = None

    def open(self):
        self._reader = GGUFReader(self.path)
        self._reader.__enter__()

    def close(self):
        if self._reader:
            self._reader.__exit__(None, None, None)
            self._reader = None

    @property
    def tensor_names(self) -> List[str]:
        if self._reader is None:
            raise RuntimeError("Streamer not opened")
        return list(self._reader.tensors.keys())

    def iter_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield batches of dequantized tensors."""
        if self._reader is None:
            raise RuntimeError("Streamer not opened")

        names = self.tensor_names
        for i in range(0, len(names), self.batch_size):
            batch_names = names[i:i + self.batch_size]
            batch = {}
            for name in batch_names:
                batch[name] = self._reader.read_tensor(
                    name, device=self.device, dtype=self.dtype
                )
            yield batch

    def iter_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield individual tensors one at a time."""
        if self._reader is None:
            raise RuntimeError("Streamer not opened")

        for name in self.tensor_names:
            yield name, self._reader.read_tensor(
                name, device=self.device, dtype=self.dtype
            )


# =============================================================================
# Model Patcher
# =============================================================================

class GGUFModelPatcher:
    """
    Patches PyTorch model weights from GGUF file.

    Handles name mapping between GGUF and PyTorch naming conventions.
    """

    def __init__(
        self,
        gguf_path: str | Path,
        name_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            gguf_path: Path to GGUF file
            name_mapping: Dict mapping GGUF names to PyTorch names.
                          If None, attempts automatic mapping.
        """
        self.gguf_path = Path(gguf_path)
        self.name_mapping = name_mapping or {}
        self._reader: Optional[GGUFReader] = None

    def _map_name(self, gguf_name: str) -> str:
        """Map GGUF tensor name to PyTorch parameter name."""
        if gguf_name in self.name_mapping:
            return self.name_mapping[gguf_name]

        # Common transformations
        # GGUF: "model.layers.0.self_attn.q_proj.weight"
        # Some models use: "transformer.h.0.attn.c_attn.weight"

        # Default: strip "model." prefix if present
        if gguf_name.startswith("model."):
            return gguf_name[6:]

        return gguf_name

    @contextmanager
    def patch_context(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        strict: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Context manager that patches model weights from GGUF.

        Args:
            model: PyTorch model to patch
            device: Device to load tensors to
            dtype: Target dtype
            strict: If True, raise error on missing parameters

        Yields:
            Stats dict with loaded/skipped counts
        """
        stats = {
            "loaded": 0,
            "skipped": 0,
            "missing": [],
            "unexpected": [],
        }

        model_params = dict(model.named_parameters())

        with GGUFReader(self.gguf_path) as reader:
            self._reader = reader

            for gguf_name in reader.tensors:
                torch_name = self._map_name(gguf_name)

                if torch_name not in model_params:
                    stats["unexpected"].append(gguf_name)
                    stats["skipped"] += 1
                    continue

                # Load and assign
                param = model_params[torch_name]
                tensor = reader.read_tensor(gguf_name, device=device, dtype=dtype)

                # Handle shape mismatches
                if tensor.shape != param.shape:
                    logger.warning(
                        f"Shape mismatch for {torch_name}: "
                        f"GGUF {tensor.shape} vs Model {param.shape}"
                    )
                    stats["skipped"] += 1
                    continue

                # Assign weight
                with torch.no_grad():
                    param.copy_(tensor)
                stats["loaded"] += 1

            # Check for missing
            gguf_mapped = {self._map_name(n) for n in reader.tensors}
            for torch_name in model_params:
                if torch_name not in gguf_mapped:
                    stats["missing"].append(torch_name)

            if strict and stats["missing"]:
                raise ValueError(
                    f"Missing parameters in GGUF: {stats['missing'][:10]}..."
                )

            yield stats
            self._reader = None


# =============================================================================
# Convenience Functions
# =============================================================================

def load_gguf_tensors(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Load all tensors from GGUF file.

    Warning: May use significant memory for large models.
    Prefer tensor_loading_context for memory-efficient access.
    """
    tensors = {}
    with GGUFReader(path) as reader:
        for name in reader.tensors:
            tensors[name] = reader.read_tensor(name, device=device, dtype=dtype)
    return tensors


def inspect_gguf(path: str | Path) -> Dict[str, Any]:
    """
    Return GGUF file info without loading tensors.
    """
    with GGUFReader(path) as reader:
        tensor_info = {}
        for name, info in reader.tensors.items():
            tensor_info[name] = {
                "shape": info.shape,
                "dtype": info.dtype_name,
                "numel": info.numel,
                "nbytes": info.nbytes,
            }

        return {
            "version": reader.header.version,
            "n_tensors": reader.header.n_tensors,
            "metadata": reader.header.metadata,
            "tensors": tensor_info,
        }


def print_gguf_info(path: str | Path):
    """Print human-readable GGUF file info."""
    info = inspect_gguf(path)

    print(f"GGUF File: {path}")
    print(f"Version: {info['version']}")
    print(f"Tensors: {info['n_tensors']}")
    print()

    # Key metadata
    meta = info["metadata"]
    print("Metadata:")
    for key in ["general.architecture", "general.name", "general.file_type"]:
        if key in meta:
            print(f"  {key}: {meta[key]}")
    print()

    # Tensor summary
    print("Tensors (first 20):")
    for i, (name, tinfo) in enumerate(info["tensors"].items()):
        if i >= 20:
            print(f"  ... and {len(info['tensors']) - 20} more")
            break
        print(f"  {name}: {tinfo['shape']} ({tinfo['dtype']}, {tinfo['nbytes'] / 1024 / 1024:.2f} MB)")
