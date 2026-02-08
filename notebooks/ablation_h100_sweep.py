#!/usr/bin/env python3
"""
ST-A² Multi-Resolution Sweep: Area Attention vs Baseline
=========================================================

Runs a multi-resolution ablation comparing baseline (full attention)
vs ST-A² (area attention) on the real V-JEPA 2 ViT-L model.

Usage:
    # Clone and run (on Lambda Labs / any GPU machine):
    git clone -b feat/st-a2-area-attention https://github.com/tarassh/vjepa2.git
    cd vjepa2
    pip install timm
    python notebooks/ablation_h100_sweep.py

Resolution sweep:
    256px/16f  →  2,048 tokens, ~512 visible, batch=4
    384px/16f  →  4,608 tokens, ~1,152 visible, batch=2
    256px/64f  →  8,192 tokens, ~2,048 visible, batch=1
    384px/64f  → 18,432 tokens, ~4,608 visible, batch=1
"""

import os
import sys
import copy
import time
import gc
import traceback

import torch
import torch.nn.functional as F
import numpy as np

# Ensure repo root is on path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)

from app.vjepa.utils import init_video_model
from src.masks.multiseq_multiblock3d import _MaskGenerator
from src.masks.utils import apply_masks
from src.models.utils.modules import Block, RoPEAttention, RoPEAreaAttention


# ──────────────────────────────────────────────────────────────────
# GPU Detection
# ──────────────────────────────────────────────────────────────────

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
gpu_mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9

if torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
    dtype_str = "bfloat16"
else:
    DTYPE = torch.float16
    dtype_str = "float16"

print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Dtype: {dtype_str}")
print(f"Compute capability: {props.major}.{props.minor}")
print()


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

MODEL_CFG = dict(
    model_name="vit_large",
    patch_size=16,
    tubelet_size=2,
    pred_depth=12,
    pred_embed_dim=384,
    pred_num_heads=12,
    num_steps=100,
    warmup_steps=10,
    lr=5.25e-4,
    weight_decay=0.04,
    loss_exp=1.0,
    ema_momentum=0.999,
)

# (label, crop_size, num_frames, batch_size)
SWEEP_RESOLUTIONS = [
    ("256px-16f", 256, 16, 4),
    ("384px-16f", 384, 16, 2),
    ("256px-64f", 256, 64, 1),
    ("384px-64f", 384, 64, 1),
]

MASK_CFGS = [
    dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
         aspect_ratio=(0.75, 1.5), num_blocks=8, max_temporal_keep=1.0),
    dict(spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0),
         aspect_ratio=(0.75, 1.5), num_blocks=2, max_temporal_keep=1.0),
]

# Build all configs
ALL_CONFIGS = {}
for label, crop, frames, batch in SWEEP_RESOLUTIONS:
    H = W = crop // MODEL_CFG["patch_size"]
    T = frames // MODEL_CFG["tubelet_size"]
    total = H * W * T
    visible = total // 4

    base = {
        **MODEL_CFG,
        "crop_size": crop,
        "num_frames": frames,
        "batch_size": batch,
        "total_tokens": total,
        "visible_tokens": visible,
        "resolution_label": label,
    }

    ALL_CONFIGS[(label, "baseline")] = {**base, "use_area_attention": False}
    ALL_CONFIGS[(label, "st_a2")] = {
        **base,
        "use_area_attention": True,
        "area_attention_layers": [0, 18],
        "area_spatial_splits": 2,
        "area_temporal_splits": 2,
        "area_residual_scale": 1.0,
    }

print(f"Sweep: {len(SWEEP_RESOLUTIONS)} resolutions x 2 configs = {len(ALL_CONFIGS)} runs")
print(f"Steps per run: {MODEL_CFG['num_steps']}")
print()
print(f"{'Label':<12} {'Crop':>5} {'Frames':>6} {'Batch':>5} {'Total':>7} {'Visible':>7} {'Per-Area':>8}")
print("-" * 60)
for label, crop, frames, batch in SWEEP_RESOLUTIONS:
    H = crop // 16
    T = frames // 2
    total = H * H * T
    visible = total // 4
    per_area = visible // 4
    print(f"{label:<12} {crop:>5} {frames:>6} {batch:>5} {total:>7} {visible:>7} {per_area:>8}")
print()


# ──────────────────────────────────────────────────────────────────
# Data Utilities
# ──────────────────────────────────────────────────────────────────

def make_mask_generators(cfg):
    generators = []
    for m in MASK_CFGS:
        gen = _MaskGenerator(
            crop_size=cfg["crop_size"],
            num_frames=cfg["num_frames"],
            spatial_patch_size=cfg["patch_size"],
            temporal_patch_size=cfg["tubelet_size"],
            spatial_pred_mask_scale=m["spatial_scale"],
            temporal_pred_mask_scale=m["temporal_scale"],
            aspect_ratio=m["aspect_ratio"],
            npred=m["num_blocks"],
            max_context_frames_ratio=m["max_temporal_keep"],
        )
        generators.append(gen)
    return generators


def make_synthetic_batch(cfg, mask_generators):
    B = cfg["batch_size"]
    T = cfg["num_frames"]
    H = W = cfg["crop_size"]
    clip = torch.randn(B, 3, T, H, W, device=device)
    all_masks_enc = []
    all_masks_pred = []
    for gen in mask_generators:
        masks_enc, masks_pred = gen(B)
        all_masks_enc.append(masks_enc.to(device))
        all_masks_pred.append(masks_pred.to(device))
    return [clip], [all_masks_enc], [all_masks_pred]


# ──────────────────────────────────────────────────────────────────
# Model Builder
# ──────────────────────────────────────────────────────────────────

def build_models(cfg, use_activation_checkpointing=True):
    num_mask_tokens = len(MASK_CFGS)
    encoder, predictor = init_video_model(
        device=device,
        patch_size=cfg["patch_size"],
        max_num_frames=cfg["num_frames"],
        tubelet_size=cfg["tubelet_size"],
        model_name=cfg["model_name"],
        crop_size=cfg["crop_size"],
        pred_depth=cfg["pred_depth"],
        pred_num_heads=cfg["pred_num_heads"],
        pred_embed_dim=cfg["pred_embed_dim"],
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=True,
        use_sdpa=True,
        use_rope=True,
        use_activation_checkpointing=use_activation_checkpointing,
        use_area_attention=cfg["use_area_attention"],
        area_attention_layers=cfg.get("area_attention_layers"),
        area_spatial_splits=cfg.get("area_spatial_splits", 2),
        area_temporal_splits=cfg.get("area_temporal_splits", 2),
        area_residual_scale=cfg.get("area_residual_scale", 1.0),
    )
    target_encoder = copy.deepcopy(encoder)
    target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda")
    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    pred_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"  Encoder: {enc_params:.1f}M params, Predictor: {pred_params:.1f}M params")
    return encoder, predictor, target_encoder, optimizer, scaler


# ──────────────────────────────────────────────────────────────────
# Training Step
# ──────────────────────────────────────────────────────────────────

def train_step(encoder, predictor, target_encoder, optimizer, scaler,
               clips, masks_enc, masks_pred, loss_exp=1.0, momentum=0.999):
    def forward_target(c):
        with torch.no_grad():
            h = target_encoder(c)
            h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
            return h

    def forward_context(c):
        z = encoder(c, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        return z

    def loss_fn(z, h):
        h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
        loss, n = 0, 0
        for zi, hi in zip(z, h):
            for zij, hij in zip(zi, hi):
                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                n += 1
        loss /= n
        return loss

    with torch.amp.autocast("cuda", dtype=DTYPE):
        h = forward_target(clips)
        z = forward_context(clips)
        loss = loss_fn(z, h)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    with torch.no_grad():
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    return float(loss)


# ──────────────────────────────────────────────────────────────────
# Run Ablation
# ──────────────────────────────────────────────────────────────────

def run_ablation(name, cfg):
    label = cfg["resolution_label"]
    attn_type = "ST-A\u00b2" if cfg["use_area_attention"] else "Baseline"
    print(f"\n{'='*70}")
    print(f"Running: {label} / {attn_type}")
    print(f"  Resolution: {cfg['crop_size']}px, {cfg['num_frames']}f, batch={cfg['batch_size']}")
    print(f"  Tokens: {cfg['total_tokens']} total, ~{cfg['visible_tokens']} visible")
    print(f"  Area attention: {cfg['use_area_attention']}")
    print(f"{'='*70}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    encoder, predictor, target_encoder, optimizer, scaler = build_models(cfg)
    mask_generators = make_mask_generators(cfg)

    num_steps = cfg["num_steps"]
    losses = []
    step_times_ms = []

    print("  Warmup (3 steps)...")
    for _ in range(3):
        clips, masks_enc, masks_pred = make_synthetic_batch(cfg, mask_generators)
        _ = train_step(encoder, predictor, target_encoder, optimizer, scaler,
                       clips, masks_enc, masks_pred,
                       loss_exp=cfg["loss_exp"], momentum=cfg["ema_momentum"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"  Training ({num_steps} steps)...")
    for step in range(num_steps):
        clips, masks_enc, masks_pred = make_synthetic_batch(cfg, mask_generators)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss = train_step(encoder, predictor, target_encoder, optimizer, scaler,
                          clips, masks_enc, masks_pred,
                          loss_exp=cfg["loss_exp"], momentum=cfg["ema_momentum"])
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        losses.append(loss)
        step_times_ms.append(elapsed_ms)
        if (step + 1) % 25 == 0 or step == 0:
            print(f"    Step {step+1:4d}/{num_steps}: "
                  f"loss={np.mean(losses[-25:]):.4f}, time={np.mean(step_times_ms[-25:]):.1f}ms")

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    del encoder, predictor, target_encoder, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "losses": losses,
        "step_times_ms": step_times_ms,
        "peak_mem_mb": peak_mem_mb,
        "avg_step_ms": np.mean(step_times_ms),
        "final_loss": np.mean(losses[-20:]),
        "throughput_steps_sec": 1000.0 / np.mean(step_times_ms),
        "resolution_label": cfg["resolution_label"],
        "visible_tokens": cfg["visible_tokens"],
    }
    print(f"  Done. loss={result['final_loss']:.4f}, "
          f"time={result['avg_step_ms']:.1f}ms, "
          f"mem={result['peak_mem_mb']:.0f}MB")
    return result


# ──────────────────────────────────────────────────────────────────
# Per-Layer Profiling
# ──────────────────────────────────────────────────────────────────

def profile_encoder(cfg, num_runs=20):
    attn_type = "ST-A\u00b2" if cfg["use_area_attention"] else "Baseline"
    label = cfg["resolution_label"]
    print(f"\nProfiling: {label} / {attn_type}")

    torch.cuda.empty_cache()
    gc.collect()

    encoder, predictor = init_video_model(
        device=device,
        patch_size=cfg["patch_size"],
        max_num_frames=cfg["num_frames"],
        tubelet_size=cfg["tubelet_size"],
        model_name=cfg["model_name"],
        crop_size=cfg["crop_size"],
        pred_depth=cfg["pred_depth"],
        pred_num_heads=cfg["pred_num_heads"],
        pred_embed_dim=cfg["pred_embed_dim"],
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=len(MASK_CFGS),
        zero_init_mask_tokens=True,
        use_sdpa=True,
        use_rope=True,
        use_activation_checkpointing=False,
        use_area_attention=cfg["use_area_attention"],
        area_attention_layers=cfg.get("area_attention_layers"),
        area_spatial_splits=cfg.get("area_spatial_splits", 2),
        area_temporal_splits=cfg.get("area_temporal_splits", 2),
        area_residual_scale=cfg.get("area_residual_scale", 1.0),
    )
    encoder.eval()

    blocks = encoder.backbone.blocks
    num_layers = len(blocks)
    layer_timings = [{"attn": [], "mlp": []} for _ in range(num_layers)]

    original_forwards = []
    for i, block in enumerate(blocks):
        original_forwards.append(block.forward)

        def make_profiled_forward(block_ref, layer_idx):
            def profiled_forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if isinstance(block_ref.attn, (RoPEAttention, RoPEAreaAttention)):
                    y = block_ref.attn(block_ref.norm1(x), mask=mask, attn_mask=attn_mask,
                                       T=T, H_patches=H_patches, W_patches=W_patches)
                else:
                    y = block_ref.attn(block_ref.norm1(x), mask=mask, attn_mask=attn_mask)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                x_out = x + block_ref.drop_path(y)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                x_out = x_out + block_ref.drop_path(block_ref.mlp(block_ref.norm2(x_out)))
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                layer_timings[layer_idx]["attn"].append((t1 - t0) * 1000)
                layer_timings[layer_idx]["mlp"].append((t3 - t2) * 1000)
                return x_out
            return profiled_forward

        block.forward = make_profiled_forward(block, i)

    mask_generators = make_mask_generators(cfg)

    print("  Warmup (3 runs)...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        for _ in range(3):
            clips, masks_enc, _ = make_synthetic_batch(cfg, mask_generators)
            _ = encoder(clips, masks_enc)
    for lt in layer_timings:
        lt["attn"].clear()
        lt["mlp"].clear()

    print(f"  Profiling ({num_runs} forward passes)...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        for _ in range(num_runs):
            clips, masks_enc, _ = make_synthetic_batch(cfg, mask_generators)
            _ = encoder(clips, masks_enc)

    for i, block in enumerate(blocks):
        block.forward = original_forwards[i]

    profile_data = []
    total_attn_ms = 0
    total_mlp_ms = 0
    for i in range(num_layers):
        attn_ms = np.mean(layer_timings[i]["attn"])
        mlp_ms = np.mean(layer_timings[i]["mlp"])
        attn_type_name = type(blocks[i].attn).__name__
        total_attn_ms += attn_ms
        total_mlp_ms += mlp_ms
        profile_data.append({
            "layer": i, "attn_type": attn_type_name,
            "attn_ms": attn_ms, "mlp_ms": mlp_ms,
            "total_ms": attn_ms + mlp_ms,
            "attn_pct": attn_ms / (attn_ms + mlp_ms) * 100,
        })

    del encoder, predictor
    torch.cuda.empty_cache()
    gc.collect()

    total_ms = total_attn_ms + total_mlp_ms
    print(f"  Encoder: {total_ms:.1f}ms "
          f"(attn={total_attn_ms:.1f}ms [{total_attn_ms/total_ms*100:.0f}%], "
          f"mlp={total_mlp_ms:.1f}ms [{total_mlp_ms/total_ms*100:.0f}%])")
    return profile_data


# ──────────────────────────────────────────────────────────────────
# MAIN: Execute Sweep
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PHASE 1: ABLATION SWEEP (Training)")
    print("=" * 70)

    results = {}
    skipped = []

    for res_label, crop, frames, batch in SWEEP_RESOLUTIONS:
        for config_name in ["baseline", "st_a2"]:
            key = (res_label, config_name)
            cfg = ALL_CONFIGS[key]
            try:
                results[key] = run_ablation(config_name, cfg)
            except torch.cuda.OutOfMemoryError:
                print(f"\n  \u274c OOM: {res_label}/{config_name} — skipping")
                skipped.append(key)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"\n  \u274c Error: {res_label}/{config_name} — {e}")
                traceback.print_exc()
                skipped.append(key)
                torch.cuda.empty_cache()
                gc.collect()

    print(f"\n{'='*70}")
    print(f"Sweep complete! {len(results)} runs succeeded, {len(skipped)} skipped.")
    if skipped:
        print(f"Skipped: {skipped}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Per-Layer Profiling
    # ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  PHASE 2: PER-LAYER PROFILING")
    print("=" * 70)

    profile_results = {}
    for key in results:
        cfg = ALL_CONFIGS[key]
        try:
            profile_results[key] = profile_encoder(cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"  \u274c OOM during profiling: {key} — skipping")
            torch.cuda.empty_cache()
            gc.collect()

    print("\nAll profiling complete!")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Summary Table
    # ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("  ST-A\u00b2 MULTI-RESOLUTION SWEEP RESULTS")
    print("=" * 100)
    print(f"  GPU: {gpu_name} ({gpu_mem_gb:.1f} GB), dtype: {dtype_str}")
    print(f"  Model: {MODEL_CFG['model_name']}, steps: {MODEL_CFG['num_steps']}")
    print()

    header = (f"{'Resolution':<12} {'Visible':>7} {'Batch':>5} "
              f"{'BL Time':>9} {'ST Time':>9} {'\u0394 Time':>9} "
              f"{'BL Loss':>9} {'ST Loss':>9} {'\u0394 Loss':>9} "
              f"{'BL Mem':>8} {'ST Mem':>8}")
    print(header)
    print("-" * 100)

    for res_label, crop, frames, batch in SWEEP_RESOLUTIONS:
        bl_key = (res_label, "baseline")
        st_key = (res_label, "st_a2")

        if bl_key not in results or st_key not in results:
            H = crop // 16
            T = frames // 2
            vis = H * H * T // 4
            print(f"{res_label:<12} {vis:>7} {batch:>5}   SKIPPED (OOM)")
            continue

        bl = results[bl_key]
        st = results[st_key]
        vis = bl["visible_tokens"]

        dt = (st["avg_step_ms"] - bl["avg_step_ms"]) / bl["avg_step_ms"] * 100
        dl = (st["final_loss"] - bl["final_loss"]) / bl["final_loss"] * 100

        speed_marker = "\u2705" if dt <= 0 else ""

        print(f"{res_label:<12} {vis:>7} {batch:>5} "
              f"{bl['avg_step_ms']:>8.1f}ms {st['avg_step_ms']:>8.1f}ms {dt:>+8.1f}% "
              f"{bl['final_loss']:>9.4f} {st['final_loss']:>9.4f} {dl:>+8.1f}% "
              f"{bl['peak_mem_mb']:>7.0f}M {st['peak_mem_mb']:>7.0f}M "
              f"{speed_marker}")

    print("=" * 100)
    print()
    print("\u2705 = ST-A\u00b2 is FASTER than baseline")
    print("Negative \u0394 Time = ST-A\u00b2 faster, Negative \u0394 Loss = ST-A\u00b2 converges better")

    # ──────────────────────────────────────────────────────────────
    # PHASE 4: Per-Layer Profiling Table (highest resolution)
    # ──────────────────────────────────────────────────────────────

    best_res = None
    for res_label, _, _, _ in reversed(SWEEP_RESOLUTIONS):
        if (res_label, "baseline") in profile_results and (res_label, "st_a2") in profile_results:
            best_res = res_label
            break

    if best_res:
        print(f"\n{'='*90}")
        print(f"  PER-LAYER PROFILING: {best_res}")
        print(f"{'='*90}")

        for config_name, label in [("baseline", "BASELINE"), ("st_a2", "ST-A\u00b2")]:
            key = (best_res, config_name)
            data = profile_results[key]
            total_attn = sum(d["attn_ms"] for d in data)
            total_mlp = sum(d["mlp_ms"] for d in data)
            total = total_attn + total_mlp

            print(f"\n  {label}")
            print(f"  {'Layer':<6} {'Type':<22} {'Attn(ms)':>9} {'MLP(ms)':>9} {'Total(ms)':>10} {'Attn%':>7}")
            print(f"  {'-'*68}")
            for d in data:
                print(f"  {d['layer']:<6} {d['attn_type']:<22} {d['attn_ms']:>9.2f} {d['mlp_ms']:>9.2f} "
                      f"{d['total_ms']:>10.2f} {d['attn_pct']:>6.1f}%")
            print(f"  {'-'*68}")
            print(f"  {'TOTAL':<6} {'':<22} {total_attn:>9.2f} {total_mlp:>9.2f} "
                  f"{total:>10.2f} {total_attn/total*100:>6.1f}%")

        bl_data = profile_results[(best_res, "baseline")]
        st_data = profile_results[(best_res, "st_a2")]
        bl_attn = sum(d["attn_ms"] for d in bl_data)
        st_attn = sum(d["attn_ms"] for d in st_data)
        bl_total = sum(d["total_ms"] for d in bl_data)
        st_total = sum(d["total_ms"] for d in st_data)

        print(f"\n  Encoder total: Baseline={bl_total:.1f}ms, ST-A\u00b2={st_total:.1f}ms "
              f"({(st_total-bl_total)/bl_total*100:+.1f}%)")
        print(f"  Attention:     Baseline={bl_attn:.1f}ms, ST-A\u00b2={st_attn:.1f}ms "
              f"({(st_attn-bl_attn)/bl_attn*100:+.1f}%)")

    # ──────────────────────────────────────────────────────────────
    # PHASE 5: CSV Export
    # ──────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  SAVING RESULTS")
    print(f"{'='*70}")

    # Per-step CSV
    csv_lines = ["resolution,config,crop_size,num_frames,batch_size,visible_tokens,step,loss,step_time_ms"]
    for key, r in results.items():
        res_label, config_name = key
        cfg = ALL_CONFIGS[key]
        for i in range(len(r["losses"])):
            csv_lines.append(
                f"{res_label},{config_name},{cfg['crop_size']},{cfg['num_frames']},"
                f"{cfg['batch_size']},{cfg['visible_tokens']},{i+1},"
                f"{r['losses'][i]:.6f},{r['step_times_ms'][i]:.2f}"
            )
    out_path = os.path.join(repo_root, "sweep_results.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  Per-step metrics: {out_path} ({len(csv_lines)-1} rows)")

    # Summary CSV
    summary_lines = [
        "resolution,config,model,crop_size,num_frames,batch_size,"
        "total_tokens,visible_tokens,use_area_attention,num_steps,"
        "dtype,gpu,final_loss,avg_step_ms,peak_mem_mb,throughput_steps_sec"
    ]
    for key, r in results.items():
        res_label, config_name = key
        cfg = ALL_CONFIGS[key]
        summary_lines.append(
            f"{res_label},{config_name},{cfg['model_name']},{cfg['crop_size']},"
            f"{cfg['num_frames']},{cfg['batch_size']},{cfg['total_tokens']},"
            f"{cfg['visible_tokens']},{cfg['use_area_attention']},{cfg['num_steps']},"
            f"{dtype_str},{gpu_name},{r['final_loss']:.6f},{r['avg_step_ms']:.2f},"
            f"{r['peak_mem_mb']:.0f},{r['throughput_steps_sec']:.4f}"
        )
    out_path = os.path.join(repo_root, "sweep_summary.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary: {out_path}")

    print(f"\n{'='*70}")
    print("  ALL DONE!")
    print(f"{'='*70}")
