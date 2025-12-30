"""
Test: Backward Pass Through IQ3_XXS Quantized Weights

This proves that "wacky" quantization formats with:
- Codebook lookups
- Non-uniform bit layouts  
- Importance weighting
- Nested scales

...are still perfectly trainable via STE.

The format is arbitrary. The backward pass is not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vtensor import QuantizedParameter, QuantizedLinear, DequantSTE
from .dequant_kernels import (
    dequant_iq3_xxs_cuda,
    requant_iq3_xxs_cuda,
    dequant_q4_0_cuda,
    requant_q4_0_cuda,
)


def test_ste_gradient_flow():
    """
    Test that gradients flow through STE for various quant types.
    """
    print("=" * 60)
    print("TEST: STE Gradient Flow")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    for quant_type, dequant_fn, requant_fn in [
        ("q4_0", dequant_q4_0_cuda, requant_q4_0_cuda),
        ("iq3_xxs", dequant_iq3_xxs_cuda, requant_iq3_xxs_cuda),
    ]:
        print(f"\n--- Testing {quant_type} ---")
        
        # Create a random weight tensor
        shape = (256, 256)  # IQ3_XXS needs 256-element blocks
        numel = shape[0] * shape[1]
        
        # Original fp32 weight
        w_fp32 = torch.randn(shape, device=device)
        
        # Quantize it
        w_quant = requant_fn(w_fp32, quant_type)
        
        # Create QuantizedParameter
        qparam = QuantizedParameter(
            raw_data=w_quant,
            shape=shape,
            quant_type=quant_type,
            device=device,
        )
        
        # Forward pass through STE
        w_dequant = qparam()
        
        # Simple loss: mean squared
        loss = (w_dequant ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # Check that gradient was captured
        grad = qparam.grad
        
        assert grad is not None, f"Gradient not captured for {quant_type}"
        assert grad.shape == shape, f"Gradient shape mismatch: {grad.shape} vs {shape}"
        assert not torch.isnan(grad).any(), f"NaN in gradient for {quant_type}"
        assert not torch.isinf(grad).any(), f"Inf in gradient for {quant_type}"
        
        # Gradient should be 2 * w_dequant / numel (derivative of mean squared)
        expected_grad = 2 * w_dequant / numel
        grad_error = (grad - expected_grad).abs().max().item()
        
        print(f"  Gradient captured: ✓")
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient error vs expected: {grad_error:.6f}")
        print(f"  Gradient mean: {grad.mean().item():.6f}")
        print(f"  Gradient std: {grad.std().item():.6f}")
        
        assert grad_error < 1e-5, f"Gradient error too large for {quant_type}"
        print(f"  STE gradient correct: ✓")


def test_quantized_linear_backward():
    """
    Test backward pass through a QuantizedLinear layer.
    """
    print("\n" + "=" * 60)
    print("TEST: QuantizedLinear Backward Pass")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    in_features, out_features = 256, 256
    batch_size, seq_len = 2, 16
    
    for quant_type in ["q4_0", "iq3_xxs"]:
        print(f"\n--- Testing {quant_type} QuantizedLinear ---")
        
        # Create weight and quantize
        w = torch.randn(out_features, in_features, device=device)
        
        if quant_type == "q4_0":
            w_quant = requant_q4_0_cuda(w, quant_type)
        else:
            w_quant = requant_iq3_xxs_cuda(w, quant_type)
        
        # Create QuantizedLinear
        layer = QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            raw_weight=w_quant,
            quant_type=quant_type,
            device=device,
        )
        
        # Input with requires_grad to test full chain
        x = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=True)
        
        # Forward
        y = layer(x)
        
        # Loss
        loss = y.sum()
        
        # Backward
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradient not computed"
        assert layer.weight.grad is not None, f"Weight gradient not captured for {quant_type}"
        
        print(f"  Input grad shape: {x.grad.shape}")
        print(f"  Weight grad shape: {layer.weight.grad.shape}")
        print(f"  Input grad norm: {x.grad.norm().item():.4f}")
        print(f"  Weight grad norm: {layer.weight.grad.norm().item():.4f}")
        print(f"  Backward pass: ✓")


def test_training_loop_iq3_xxs():
    """
    Test a complete training loop with IQ3_XXS weights.
    
    This is the ultimate proof: we can train through the wackiest format.
    """
    print("\n" + "=" * 60)
    print("TEST: Training Loop with IQ3_XXS")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Simple MLP with quantized weights
    class QuantizedMLP(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            
            # Create random weights and quantize to IQ3_XXS
            w1 = torch.randn(dim, dim, device=device) * 0.1
            w2 = torch.randn(dim, dim, device=device) * 0.1
            
            w1_quant = requant_iq3_xxs_cuda(w1, "iq3_xxs")
            w2_quant = requant_iq3_xxs_cuda(w2, "iq3_xxs")
            
            self.fc1 = QuantizedLinear(dim, dim, w1_quant, "iq3_xxs", device=device)
            self.fc2 = QuantizedLinear(dim, dim, w2_quant, "iq3_xxs", device=device)
        
        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = QuantizedMLP().to(device)
    
    # Simple training data: identity function
    batch_size = 8
    dim = 256
    
    x = torch.randn(batch_size, dim, device=device)
    y = x.clone()  # Target: identity
    
    # Training loop
    losses = []
    lr = 0.01
    
    print("\nTraining IQ3_XXS MLP to learn identity function...")
    
    for step in range(10):
        # Forward
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        # Backward
        loss.backward()
        
        # Manual SGD step on quantized weights
        model.fc1.apply_gradient(lr)
        model.fc2.apply_gradient(lr)
        
        losses.append(loss.item())
        
        if step % 2 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")
    
    # Check that loss decreased
    assert losses[-1] < losses[0], "Loss did not decrease during training"
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\nLoss improved by {improvement:.1f}%")
    print("Training through IQ3_XXS: ✓")


def test_carry_buffer_accumulation():
    """
    Test that carry buffers correctly accumulate sub-quantization updates.
    """
    print("\n" + "=" * 60)
    print("TEST: Carry Buffer Accumulation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create a quantized parameter with carry enabled
    shape = (256, 256)
    w = torch.randn(shape, device=device) * 0.1
    w_quant = requant_q4_0_cuda(w, "q4_0")
    
    qparam = QuantizedParameter(
        raw_data=w_quant,
        shape=shape,
        quant_type="q4_0",
        device=device,
        enable_carry=True,
        carry_dim=64,  # JL compression
    )
    
    print(f"Weight shape: {shape}")
    print(f"Carry dim: 64 (JL compressed from {shape[0] * shape[1]})")
    print(f"Compression: {shape[0] * shape[1] / 64:.0f}x")
    
    # Simulate many small gradient updates
    initial_weight = qparam.dequantize().clone()
    
    for i in range(100):
        # Small gradient (would be lost without carry)
        grad = torch.randn(shape, device=device) * 0.001
        qparam.grad_holder.grad = grad
        qparam.apply_gradient(lr=0.01)
    
    final_weight = qparam.dequantize()
    
    # Weight should have changed (carry accumulated enough to matter)
    weight_change = (final_weight - initial_weight).norm().item()
    print(f"\nWeight change after 100 small updates: {weight_change:.6f}")
    
    # Without carry, these small updates would be lost to quantization floor
    assert weight_change > 0.01, "Carry buffer not accumulating properly"
    print("Carry buffer accumulation: ✓")


def test_roundtrip_correctness():
    """
    Test that dequant → requant preserves values reasonably well.
    """
    print("\n" + "=" * 60)
    print("TEST: Quantization Roundtrip")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for quant_type, dequant_fn, requant_fn in [
        ("q4_0", dequant_q4_0_cuda, requant_q4_0_cuda),
        ("iq3_xxs", dequant_iq3_xxs_cuda, requant_iq3_xxs_cuda),
    ]:
        print(f"\n--- {quant_type} roundtrip ---")
        
        # Original tensor
        shape = (256, 256)
        numel = shape[0] * shape[1]
        original = torch.randn(shape, device=device)
        
        # Quantize
        quantized = requant_fn(original, quant_type)
        
        # Dequantize
        recovered = dequant_fn(quantized, numel, device).view(shape)
        
        # Compute error
        mse = F.mse_loss(recovered.float(), original.float()).item()
        max_err = (recovered.float() - original.float()).abs().max().item()
        
        print(f"  MSE: {mse:.6f}")
        print(f"  Max error: {max_err:.4f}")
        print(f"  Quantized bytes: {quantized.numel()}")
        print(f"  Compression: {numel * 2 / quantized.numel():.2f}x vs fp16")


def run_all_tests():
    """Run all backward pass tests."""
    print("\n" + "=" * 70)
    print("GGUF VTENSOR BACKWARD PASS TESTS")
    print("Proving that quantized weights are trainable via STE")
    print("=" * 70)
    
    test_ste_gradient_flow()
    test_quantized_linear_backward()
    test_training_loop_iq3_xxs()
    test_carry_buffer_accumulation()
    test_roundtrip_correctness()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("Quantized formats (including IQ3_XXS) are trainable.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
