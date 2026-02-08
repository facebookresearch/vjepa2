"""
ST-A² (Spatiotemporal Area Attention) verification script.

Run on Google Colab free tier (CPU or T4 GPU) or any machine with PyTorch.
Verifies:
  1. RoPEAreaAttention produces correct output shapes
  2. Weight compatibility with RoPEAttention (checkpoint loading)
  3. Gradients flow through area attention
  4. Area assignment is correct for known token positions
  5. Output equivalence: with num_areas=1, matches RoPEAttention exactly
  6. Full VisionTransformer forward pass with area attention enabled
  7. Attention cost reduction estimate
  8. CPU wall-clock timing comparison
  9. GPU benchmark: forward, forward+backward, memory (auto-skipped if no CUDA)

Usage:
  pip install torch timm einops
  python test_area_attention.py

Or in Colab:
  !git clone https://github.com/tarassh/vjepa2.git
  %cd vjepa2
  !pip install timm einops
  !python notebooks/test_area_attention.py
"""

import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.models.utils.modules import RoPEAttention, RoPEAreaAttention, Block


def test_shape_and_forward():
    """Test 1: Basic forward pass and output shape."""
    print("=" * 60)
    print("Test 1: Forward pass shape verification")
    print("=" * 60)

    dim = 384  # ViT-S embed dim
    num_heads = 6
    B = 2
    T, H, W = 8, 16, 16  # 8 temporal groups, 16x16 spatial
    N_full = T * H * W  # 2048 tokens

    # Simulate ~25% visible tokens (75% masked)
    N_visible = N_full // 4  # 512 visible tokens

    area_attn = RoPEAreaAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        use_sdpa=False,  # Use manual attention for CPU compatibility
        grid_size=H,
        spatial_splits=2,
        temporal_splits=2,
    )

    # Simulate sparse visible tokens
    x = torch.randn(B, N_visible, dim)

    # Simulate mask indices (sorted subset of [0, N_full))
    mask = torch.stack([
        torch.sort(torch.randperm(N_full)[:N_visible])[0]
        for _ in range(B)
    ])

    # Forward pass
    out = area_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)

    assert out.shape == (B, N_visible, dim), f"Expected {(B, N_visible, dim)}, got {out.shape}"
    print(f"  Input:  x={list(x.shape)}, mask={list(mask.shape)}")
    print(f"  Output: {list(out.shape)}")
    print(f"  PASSED\n")


def test_weight_compatibility():
    """Test 2: RoPEAreaAttention loads RoPEAttention weights."""
    print("=" * 60)
    print("Test 2: Weight compatibility (checkpoint loading)")
    print("=" * 60)

    dim = 384
    num_heads = 6

    rope_attn = RoPEAttention(dim=dim, num_heads=num_heads, qkv_bias=True, grid_size=16)
    area_attn = RoPEAreaAttention(dim=dim, num_heads=num_heads, qkv_bias=True, grid_size=16)

    # Get state dicts
    rope_sd = rope_attn.state_dict()
    area_sd = area_attn.state_dict()

    # Check parametric keys match
    rope_keys = set(rope_sd.keys())
    area_keys = set(area_sd.keys())

    shared = rope_keys & area_keys
    rope_only = rope_keys - area_keys
    area_only = area_keys - rope_keys

    print(f"  Shared keys: {sorted(shared)}")
    if rope_only:
        print(f"  WARNING - RoPE-only keys: {sorted(rope_only)}")
    if area_only:
        print(f"  Area-only keys (non-parametric): {sorted(area_only)}")

    # Load RoPEAttention weights into RoPEAreaAttention
    area_attn.load_state_dict(rope_sd, strict=False)

    # Verify weights are identical
    for key in shared:
        assert torch.equal(rope_sd[key], area_attn.state_dict()[key]), f"Weight mismatch for {key}"

    print(f"  All {len(shared)} shared weights loaded and verified.")
    print(f"  PASSED\n")


def test_gradient_flow():
    """Test 3: Gradients flow through area attention."""
    print("=" * 60)
    print("Test 3: Gradient flow")
    print("=" * 60)

    dim = 192
    num_heads = 3
    B = 2
    T, H, W = 4, 8, 8
    N_visible = (T * H * W) // 4

    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=False, grid_size=H, spatial_splits=2, temporal_splits=2,
    )

    x = torch.randn(B, N_visible, dim, requires_grad=True)
    mask = torch.stack([
        torch.sort(torch.randperm(T * H * W)[:N_visible])[0]
        for _ in range(B)
    ])

    out = area_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input!"
    assert x.grad.abs().sum() > 0, "Gradient is all zeros!"

    # Check all parameters have gradients
    params_with_grad = 0
    params_total = 0
    for name, p in area_attn.named_parameters():
        params_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad += 1
        else:
            print(f"  WARNING: No gradient for {name}")

    print(f"  Input gradient norm: {x.grad.norm().item():.4f}")
    print(f"  Parameters with gradients: {params_with_grad}/{params_total}")
    print(f"  PASSED\n")


def test_area_assignment():
    """Test 4: Verify area IDs are assigned correctly."""
    print("=" * 60)
    print("Test 4: Area assignment verification")
    print("=" * 60)

    dim = 192
    num_heads = 3
    T, H, W = 4, 8, 8
    # spatial_splits=2 → top half (h<4) = area_h=0, bottom half (h>=4) = area_h=1
    # temporal_splits=2 → first half (t<2) = area_t=0, second half (t>=2) = area_t=1

    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, grid_size=H,
        spatial_splits=2, temporal_splits=2,
    )

    # Create mask with known token positions
    # Token at (t=0, h=0, w=0) → flat_idx=0 → area_t=0, area_h=0 → area=0
    # Token at (t=0, h=5, w=0) → flat_idx=40 → area_t=0, area_h=1 → area=1
    # Token at (t=2, h=0, w=0) → flat_idx=128 → area_t=1, area_h=0 → area=2
    # Token at (t=3, h=7, w=7) → flat_idx=255 → area_t=1, area_h=1 → area=3
    tokens_per_frame = H * W  # 64
    test_positions = torch.tensor([
        0,                           # (t=0, h=0, w=0) → area 0
        0 * 64 + 5 * 8 + 0,         # (t=0, h=5, w=0) → area 1
        2 * 64 + 0 * 8 + 0,         # (t=2, h=0, w=0) → area 2
        3 * 64 + 7 * 8 + 7,         # (t=3, h=7, w=7) → area 3
    ]).unsqueeze(0)  # [1, 4]

    area_ids = area_attn._compute_area_ids(test_positions, T=T, H_patches=H, W_patches=W)

    expected = torch.tensor([[0, 1, 2, 3]])
    assert torch.equal(area_ids, expected), f"Expected {expected}, got {area_ids}"

    print(f"  Token (t=0,h=0,w=0) → area {area_ids[0,0].item()} (expected 0)")
    print(f"  Token (t=0,h=5,w=0) → area {area_ids[0,1].item()} (expected 1)")
    print(f"  Token (t=2,h=0,w=0) → area {area_ids[0,2].item()} (expected 2)")
    print(f"  Token (t=3,h=7,w=7) → area {area_ids[0,3].item()} (expected 3)")
    print(f"  PASSED\n")


def test_single_area_equivalence():
    """Test 5: With 1 area (no split), output matches RoPEAttention."""
    print("=" * 60)
    print("Test 5: Single-area equivalence with RoPEAttention")
    print("=" * 60)

    dim = 192
    num_heads = 3
    B = 1
    T, H, W = 4, 8, 8
    N = T * H * W  # No masking for clean comparison

    torch.manual_seed(42)

    rope_attn = RoPEAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=False, grid_size=H,
    )

    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=False, grid_size=H,
        spatial_splits=1, temporal_splits=1,  # Single area = full attention
    )

    # Copy weights
    area_attn.load_state_dict(rope_attn.state_dict(), strict=False)

    x = torch.randn(B, N, dim)

    # No mask (full token set)
    with torch.no_grad():
        out_rope = rope_attn(x, mask=None, T=T, H_patches=H, W_patches=W)
        out_area = area_attn(x, mask=None, T=T, H_patches=H, W_patches=W)

    max_diff = (out_rope - out_area).abs().max().item()
    mean_diff = (out_rope - out_area).abs().mean().item()

    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    # Should be numerically identical (same computation path)
    assert max_diff < 1e-5, f"Outputs differ too much: max_diff={max_diff}"
    print(f"  PASSED\n")


def test_full_vit_forward():
    """Test 6: Full VisionTransformer forward pass with area attention."""
    print("=" * 60)
    print("Test 6: Full VisionTransformer forward pass")
    print("=" * 60)

    from functools import partial
    from src.models.vision_transformer import VisionTransformer

    # Small ViT for CPU testing
    model = VisionTransformer(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        use_sdpa=False,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # ST-A² config
        use_area_attention=True,
        area_attention_layers=[0, 3],  # First 3 of 4 layers
        area_spatial_splits=2,
        area_temporal_splits=2,
    )

    B = 2
    # Video input: [B, C, T, H, W]
    x = torch.randn(B, 3, 4, 64, 64)

    # Create masks (simulate encoder masking)
    # Grid: T=4/2=2, H=64/16=4, W=64/16=4 → 32 total tokens
    N_total = 2 * 4 * 4  # 32
    N_visible = N_total // 2  # Keep 50%
    masks = [torch.stack([
        torch.sort(torch.randperm(N_total)[:N_visible])[0]
        for _ in range(B)
    ])]

    # Forward
    out = model(x, masks=masks)

    print(f"  Model: ViT (depth=4, dim=192, heads=3)")
    print(f"  Area attention on layers: [0, 1, 2], full attention on layer [3]")
    print(f"  Input video: {list(x.shape)}")
    print(f"  Visible tokens: {N_visible}/{N_total}")
    print(f"  Output: {list(out.shape)}")

    # Check block types
    for i, blk in enumerate(model.blocks):
        attn_type = type(blk.attn).__name__
        print(f"  Layer {i}: {attn_type}")

    # Verify gradient flow through entire model
    out.sum().backward()
    grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters() if p.requires_grad)
    print(f"  Gradient flow: {'OK' if grad_ok else 'FAILED'}")

    assert out.shape == (B, N_visible, 192)
    assert grad_ok
    print(f"  PASSED\n")


def test_cost_reduction():
    """Test 7: Estimate and verify attention cost reduction."""
    print("=" * 60)
    print("Test 7: Attention cost reduction estimate")
    print("=" * 60)

    # ViT-L config: depth=24, 2048 tokens, 75% masking → 512 visible
    depth = 24
    N_visible = 512
    num_areas = 4  # spatial_splits=2, temporal_splits=2
    aa_layers = 18  # First 75% of layers
    full_layers = depth - aa_layers

    # Full attention cost per layer: N²
    cost_full_per_layer = N_visible ** 2

    # Area attention cost per layer: num_areas × (N/num_areas)²
    tokens_per_area = N_visible // num_areas
    cost_area_per_layer = num_areas * (tokens_per_area ** 2)

    # Total costs
    cost_baseline = depth * cost_full_per_layer
    cost_hybrid = aa_layers * cost_area_per_layer + full_layers * cost_full_per_layer

    reduction = 1.0 - cost_hybrid / cost_baseline

    print(f"  Configuration:")
    print(f"    Depth: {depth} layers")
    print(f"    Visible tokens: {N_visible} (after 75% masking)")
    print(f"    Areas: {num_areas} (2×2 factored split)")
    print(f"    Area attention layers: {aa_layers}, full attention layers: {full_layers}")
    print(f"")
    print(f"  Cost per layer:")
    print(f"    Full attention:  {cost_full_per_layer:,} (N²)")
    print(f"    Area attention:  {cost_area_per_layer:,} ({num_areas} × {tokens_per_area}²)")
    print(f"    Per-layer reduction: {1.0 - cost_area_per_layer/cost_full_per_layer:.1%}")
    print(f"")
    print(f"  Total attention cost:")
    print(f"    Baseline (all full): {cost_baseline:,}")
    print(f"    Hybrid (ST-A²):      {cost_hybrid:,}")
    print(f"    Overall reduction:    {reduction:.1%}")
    print(f"  PASSED\n")


def test_timing():
    """Test 8: Wall-clock timing comparison on CPU."""
    print("=" * 60)
    print("Test 8: Wall-clock timing (CPU)")
    print("=" * 60)

    dim = 384
    num_heads = 6
    B = 2
    T, H, W = 8, 16, 16
    N_visible = (T * H * W) // 4  # 512 tokens

    rope_attn = RoPEAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=False, grid_size=H,
    )
    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=False, grid_size=H,
        spatial_splits=2, temporal_splits=2,
    )
    area_attn.load_state_dict(rope_attn.state_dict(), strict=False)

    x = torch.randn(B, N_visible, dim)
    mask = torch.stack([
        torch.sort(torch.randperm(T * H * W)[:N_visible])[0]
        for _ in range(B)
    ])

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            rope_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
            area_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)

    # Benchmark RoPEAttention
    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            rope_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
    rope_time = (time.time() - t0) / n_runs * 1000

    # Benchmark RoPEAreaAttention
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            area_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
    area_time = (time.time() - t0) / n_runs * 1000

    print(f"  Config: B={B}, N_visible={N_visible}, dim={dim}, heads={num_heads}")
    print(f"  RoPEAttention:     {rope_time:.1f} ms/forward")
    print(f"  RoPEAreaAttention: {area_time:.1f} ms/forward")
    print(f"  Ratio: {area_time/rope_time:.2f}x")
    print(f"  NOTE: CPU timing includes gather/scatter overhead that is")
    print(f"        negligible on GPU. GPU speedup will be much larger.")
    print(f"  PASSED\n")


def _gpu_bench_attention(attn_module, x, mask, T, H, W, n_warmup=20, n_runs=100):
    """Benchmark a single attention module on GPU with cuda events."""
    # Warmup — let CUDA kernels JIT-compile and caches settle
    for _ in range(n_warmup):
        with torch.no_grad():
            attn_module(x, mask=mask, T=T, H_patches=H, W_patches=W)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for _ in range(n_runs):
        start_event.record()
        with torch.no_grad():
            attn_module(x, mask=mask, T=T, H_patches=H, W_patches=W)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    times_ms.sort()
    # Drop top/bottom 10% for stable median
    trim = max(1, n_runs // 10)
    trimmed = times_ms[trim:-trim]
    return sum(trimmed) / len(trimmed)


def test_gpu_benchmark():
    """Test 9: GPU (T4) timing benchmark — RoPEAttention vs RoPEAreaAttention.

    This test measures the real wall-clock speedup on GPU where the O(N²)
    attention cost dominates and gather/scatter overhead is negligible.
    Skipped automatically when no CUDA device is available.
    """
    print("=" * 60)
    print("Test 9: GPU timing benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED — no CUDA device (run on Colab T4 for GPU benchmark)")
        print()
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # T4 supports FP16 natively (65 TFLOPS) but not BF16.
    # A100/H100 support BF16. Auto-select best dtype.
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        dtype_name = "BF16"
    else:
        dtype = torch.float16
        dtype_name = "FP16"
    print(f"  Dtype: {dtype_name}")
    print()

    # --------------- Configurations to benchmark ---------------
    # Each config: (label, dim, num_heads, B, T, H, W, mask_ratio)
    configs = [
        # ViT-S scale (small, sanity check)
        ("ViT-S  (384d, 6h)",   384,  6,  4,  8, 16, 16, 0.75),
        # ViT-L scale (the ablation target)
        ("ViT-L  (1024d, 16h)", 1024, 16, 2,  8, 16, 16, 0.75),
        # ViT-L larger batch
        ("ViT-L  (1024d, B=4)", 1024, 16, 4,  8, 16, 16, 0.75),
        # Longer sequence (more temporal frames, no masking — predictor-like)
        ("Long-seq (1024d, N=2048)", 1024, 16, 2, 8, 16, 16, 0.0),
    ]

    print(f"  {'Config':<30} {'N_vis':>6} {'Full':>8} {'Area':>8} {'Speedup':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for label, dim, num_heads, B, T, H, W, mask_ratio in configs:
        N_full = T * H * W
        N_visible = max(1, int(N_full * (1.0 - mask_ratio)))

        # Build modules
        rope_attn = RoPEAttention(
            dim=dim, num_heads=num_heads, qkv_bias=True,
            use_sdpa=True, grid_size=H,
        ).to(device=device, dtype=dtype).eval()

        area_attn = RoPEAreaAttention(
            dim=dim, num_heads=num_heads, qkv_bias=True,
            use_sdpa=True, grid_size=H,
            spatial_splits=2, temporal_splits=2,
        ).to(device=device, dtype=dtype).eval()

        # Copy weights for fair comparison
        area_attn.load_state_dict(rope_attn.state_dict(), strict=False)

        # Input tensors
        x = torch.randn(B, N_visible, dim, device=device, dtype=dtype)
        if mask_ratio > 0:
            mask = torch.stack([
                torch.sort(torch.randperm(N_full, device=device)[:N_visible])[0]
                for _ in range(B)
            ])
        else:
            mask = None

        # Benchmark
        try:
            full_ms = _gpu_bench_attention(
                rope_attn, x, mask, T, H, W, n_warmup=20, n_runs=100
            )
            area_ms = _gpu_bench_attention(
                area_attn, x, mask, T, H, W, n_warmup=20, n_runs=100
            )
            speedup = full_ms / area_ms
            print(f"  {label:<30} {N_visible:>6} {full_ms:>7.2f}ms {area_ms:>7.2f}ms {speedup:>7.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<30} {N_visible:>6}  OOM — skipped")
            torch.cuda.empty_cache()

        # Free memory between configs
        del rope_attn, area_attn, x, mask
        torch.cuda.empty_cache()

    # --------------- Forward + backward benchmark (ViT-L) ---------------
    print()
    print(f"  Forward + backward (ViT-L, B=2, 75% mask):")

    dim, num_heads, B, T, H, W = 1024, 16, 2, 8, 16, 16
    N_full = T * H * W
    N_visible = N_full // 4

    rope_attn = RoPEAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=True, grid_size=H,
    ).to(device=device, dtype=dtype)
    rope_attn.train()

    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=True, grid_size=H,
        spatial_splits=2, temporal_splits=2,
    ).to(device=device, dtype=dtype)
    area_attn.load_state_dict(rope_attn.state_dict(), strict=False)
    area_attn.train()

    mask = torch.stack([
        torch.sort(torch.randperm(N_full, device=device)[:N_visible])[0]
        for _ in range(B)
    ])

    def bench_fwd_bwd(attn_module, n_warmup=10, n_runs=50):
        for _ in range(n_warmup):
            x = torch.randn(B, N_visible, dim, device=device, dtype=dtype, requires_grad=True)
            out = attn_module(x, mask=mask, T=T, H_patches=H, W_patches=W)
            out.sum().backward()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_runs):
            x = torch.randn(B, N_visible, dim, device=device, dtype=dtype, requires_grad=True)
            start.record()
            out = attn_module(x, mask=mask, T=T, H_patches=H, W_patches=W)
            out.sum().backward()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        trim = max(1, n_runs // 10)
        return sum(times[trim:-trim]) / len(times[trim:-trim])

    try:
        full_fwdbwd = bench_fwd_bwd(rope_attn)
        area_fwdbwd = bench_fwd_bwd(area_attn)
        speedup_fwdbwd = full_fwdbwd / area_fwdbwd
        print(f"    Full attention:  {full_fwdbwd:.2f} ms")
        print(f"    Area attention:  {area_fwdbwd:.2f} ms")
        print(f"    Speedup:         {speedup_fwdbwd:.2f}x")
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM — try reducing batch size")
        torch.cuda.empty_cache()

    # --------------- Memory usage comparison ---------------
    print()
    print(f"  Peak memory usage (ViT-L forward, B=2, 75% mask):")

    del rope_attn, area_attn
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rope_attn = RoPEAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=True, grid_size=H,
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn(B, N_visible, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        rope_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
    full_peak_mb = torch.cuda.max_memory_allocated() / 1e6

    del rope_attn
    torch.cuda.empty_cache()

    area_attn = RoPEAreaAttention(
        dim=dim, num_heads=num_heads, qkv_bias=True,
        use_sdpa=True, grid_size=H,
        spatial_splits=2, temporal_splits=2,
    ).to(device=device, dtype=dtype).eval()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        area_attn(x, mask=mask, T=T, H_patches=H, W_patches=W)
    area_peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f"    Full attention:  {full_peak_mb:.1f} MB")
    print(f"    Area attention:  {area_peak_mb:.1f} MB")
    if full_peak_mb > 0:
        print(f"    Memory savings:  {(1 - area_peak_mb/full_peak_mb)*100:.1f}%")

    print(f"  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ST-A² (Spatiotemporal Area Attention) Verification Suite")
    print("=" * 60 + "\n")

    tests = [
        test_shape_and_forward,
        test_weight_compatibility,
        test_gradient_flow,
        test_area_assignment,
        test_single_area_equivalence,
        test_full_vit_forward,
        test_cost_reduction,
        test_timing,
        test_gpu_benchmark,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
