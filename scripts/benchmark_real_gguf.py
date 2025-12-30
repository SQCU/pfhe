"""
Benchmark: Real GGUF Model Memory Analysis

Load actual GGUF checkpoints and compare:
- Quantized storage (file/GPU)
- Dequantized size (if loaded to fp16)
- Theoretical optimizer state requirements
"""

import sys
sys.path.insert(0, 'src')

import torch
import gc
import os
from pathlib import Path

from phfe.inference.gguf_cuda import StreamingGGUFLoader
from phfe.inference.gguf_loader import GGUFReader, GGUF_TYPE_NAMES


def get_memory_stats():
    """Get current GPU memory stats."""
    torch.cuda.synchronize()
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_memory():
    """Reset GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def analyze_gguf(path: str) -> dict:
    """Analyze a GGUF file without fully loading it."""
    with GGUFReader(path) as reader:
        # Count tensors by type
        type_counts = {}
        type_sizes = {}
        total_elements = 0

        for name, tensor_info in reader.tensors.items():
            qtype = tensor_info.dtype
            type_name = GGUF_TYPE_NAMES.get(qtype, f"UNKNOWN_{qtype}")

            elements = 1
            for dim in tensor_info.shape:
                elements *= dim
            total_elements += elements

            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            type_sizes[type_name] = type_sizes.get(type_name, 0) + elements

        num_tensors = len(reader.tensors)

    file_size = os.path.getsize(path)

    return {
        'path': path,
        'file_size_gb': file_size / 1e9,
        'num_tensors': num_tensors,
        'total_elements': total_elements,
        'fp16_size_gb': total_elements * 2 / 1e9,
        'fp32_size_gb': total_elements * 4 / 1e9,
        'type_counts': type_counts,
        'type_sizes': type_sizes,
    }


def benchmark_model(path: str, load_to_gpu: bool = True, max_gpu_gb: float = 20.0):
    """Benchmark a GGUF model."""
    print(f"\n{'='*70}")
    print(f"Model: {Path(path).name}")
    print(f"{'='*70}")

    # Analyze without loading
    info = analyze_gguf(path)

    print(f"\nFile Analysis:")
    print(f"  File size: {info['file_size_gb']:.2f} GB")
    print(f"  Tensors: {info['num_tensors']}")
    print(f"  Total elements: {info['total_elements'] / 1e9:.2f}B")
    print(f"  FP16 equivalent: {info['fp16_size_gb']:.2f} GB")
    print(f"  Compression: {info['fp16_size_gb'] / info['file_size_gb']:.1f}x vs FP16")

    print(f"\nTensor types:")
    for qtype, count in sorted(info['type_counts'].items()):
        elements = info['type_sizes'][qtype]
        print(f"  {qtype}: {count} tensors, {elements/1e6:.1f}M elements")

    # Memory analysis
    print(f"\nMemory Analysis (theoretical):")

    # Quantized storage
    quant_storage = info['file_size_gb']
    print(f"  Quantized weights: {quant_storage:.2f} GB")

    # FP16 storage (if dequantized)
    fp16_storage = info['fp16_size_gb']
    print(f"  FP16 weights: {fp16_storage:.2f} GB")

    # Optimizer state for training
    # AdamW: 2x params (m, v) in FP32
    adam_state = info['fp32_size_gb'] * 2
    print(f"  AdamW state (FP32 m,v): {adam_state:.2f} GB")

    # Carry optimizer: 3x params (m, v, carry) in FP32
    carry_state = info['fp32_size_gb'] * 3
    print(f"  CarryOpt state (m,v,carry): {carry_state:.2f} GB")

    # Total for different scenarios
    print(f"\nTotal Memory Scenarios:")
    print(f"  Inference (FP16):        {fp16_storage:.2f} GB")
    print(f"  Inference (Quantized):   {quant_storage:.2f} GB  ({fp16_storage/quant_storage:.1f}x savings)")
    print(f"  Training FP16 + AdamW:   {fp16_storage + adam_state:.2f} GB")
    print(f"  Training Q4 + CarryOpt:  {quant_storage + carry_state:.2f} GB")

    # Compare training scenarios
    fp16_train = fp16_storage + adam_state
    quant_train = quant_storage + carry_state

    if quant_train < fp16_train:
        print(f"  → Quantized training saves {fp16_train/quant_train:.1f}x")
    else:
        print(f"  → Quantized training uses {quant_train/fp16_train:.1f}x MORE")

    # Load to GPU if requested (skip if too large)
    if load_to_gpu and info['fp16_size_gb'] <= max_gpu_gb:
        print(f"\nActual GPU Loading:")
        reset_memory()

        try:
            with StreamingGGUFLoader(path, device='cuda') as loader:
                mem_before = get_memory_stats()
                tensors = {}
                loaded = 0
                skipped = 0
                for name, tensor in loader.stream_tensors():
                    tensors[name] = tensor
                    loaded += 1
                torch.cuda.synchronize()
                mem_after = get_memory_stats()

                print(f"  Loaded {loaded} tensors (skipped unsupported types)")
                print(f"  GPU memory (dequantized fp16): {mem_after['allocated_gb']:.2f} GB")

                # Clean up
                del tensors
                reset_memory()

        except Exception as e:
            print(f"  Failed to load: {e}")
            import traceback
            traceback.print_exc()
    elif load_to_gpu:
        print(f"\n  Skipping GPU load: {info['fp16_size_gb']:.1f}GB exceeds {max_gpu_gb}GB limit")

    return info


def main():
    models_dir = Path("/mnt/f/dox/ai/text/models")

    # Models to test (in order of size)
    test_models = [
        "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",    # 1.8GB - Q8_0 supported
        "gemma-3-4b-it-Q8_0.gguf",                     # 3.9GB - Q8_0 supported
        "gemma-3-27b-it-q4_0.gguf",                    # 17GB - Q4_0 supported
    ]

    print("=" * 70)
    print("GGUF MODEL MEMORY ANALYSIS")
    print("Comparing quantized vs FP16 storage and training requirements")
    print("=" * 70)

    results = []
    for model_name in test_models:
        model_path = models_dir / model_name
        if model_path.exists():
            info = benchmark_model(str(model_path), load_to_gpu=True)
            results.append(info)
        else:
            print(f"\nSkipping {model_name} (not found)")

    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY: When does quantized training win?")
        print("=" * 70)

        print(f"\n{'Model':<45} {'FP16 Train':>12} {'Q4 Train':>12} {'Winner':>10}")
        print("-" * 80)

        for info in results:
            name = Path(info['path']).name[:42]
            fp16_train = info['fp16_size_gb'] + info['fp32_size_gb'] * 2
            quant_train = info['file_size_gb'] + info['fp32_size_gb'] * 3

            if quant_train < fp16_train:
                winner = f"Q4 ({fp16_train/quant_train:.1f}x)"
            else:
                winner = f"FP16 ({quant_train/fp16_train:.1f}x)"

            print(f"{name:<45} {fp16_train:>10.1f}GB {quant_train:>10.1f}GB {winner:>10}")

        print("\n" + "=" * 70)
        print("ANALYSIS: Memory per parameter")
        print("=" * 70)
        print("\nTraining memory breakdown (per parameter):")
        print("  FP16 training:  2B (weight) + 8B (Adam m,v FP32) = 10 bytes/param")
        print("  Q4_0 training:  0.5B (weight) + 12B (m,v,carry FP32) = 12.5 bytes/param")
        print("  Q8_0 training:  1B (weight) + 12B (m,v,carry FP32) = 13 bytes/param")
        print("\n→ Quantized training NEVER saves memory (optimizer state dominates)")
        print("\nInference memory:")
        print("  FP16: 2 bytes/param")
        print("  Q4_0: 0.5 bytes/param (4x savings)")
        print("  Q8_0: 1 byte/param (2x savings)")
        print("\n→ Quantized inference saves 2-4x memory")
        print("\nConclusion: Use quantization for INFERENCE, not training.")
        print("For training, use mixed-precision (FP16 weights + FP32 opt state).")


if __name__ == "__main__":
    main()
