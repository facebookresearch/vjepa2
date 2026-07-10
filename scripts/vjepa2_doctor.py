#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Validate V-JEPA 2.1 checkpoint metadata against a training YAML config.

Addresses silent failures when continuing pretraining with mismatched crop_size,
pred_depth, or encoder keys (see facebookresearch/vjepa2#163, #149).

Usage:
  python scripts/vjepa2_doctor.py --checkpoint path/to.ckpt --config path/to.yaml
  python scripts/vjepa2_doctor.py --checkpoint path/to.ckpt --config path/to.yaml --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def _nested_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def count_predictor_blocks(predictor_state: dict[str, Any]) -> int | None:
    indices: set[int] = set()
    for key in predictor_state:
        if "blocks." in key or "predictor_blocks." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ("blocks", "predictor_blocks") and i + 1 < len(parts):
                    if parts[i + 1].isdigit():
                        indices.add(int(parts[i + 1]))
    if not indices:
        return None
    return max(indices) + 1


def infer_embed_dim(state: dict[str, Any]) -> int | None:
    for key, tensor in state.items():
        if "pos_embed" in key and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[-1])
    return None


def infer_crop_size(state: dict[str, Any], *, patch_size: int = 16) -> int | None:
    """Infer spatial crop from encoder pos_embed patch grid (image or tubelets)."""
    for key, tensor in state.items():
        if "pos_embed" not in key or not hasattr(tensor, "shape"):
            continue
        if len(tensor.shape) != 3:
            continue
        num_tokens = int(tensor.shape[1])
        # drop cls token if present (common N+1 layouts)
        for n in (num_tokens, num_tokens - 1):
            side = int(math.isqrt(n))
            if side * side == n and side > 0:
                return side * patch_size
    return None


def inspect_checkpoint(path: Path, *, patch_size: int = 16) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"expected dict checkpoint at {path}")

    encoder = ckpt.get("ema_encoder") or ckpt.get("encoder") or {}
    predictor = ckpt.get("predictor") or {}

    return {
        "path": str(path),
        "top_level_keys": sorted(ckpt.keys()),
        "has_ema_encoder": "ema_encoder" in ckpt,
        "has_target_encoder": "target_encoder" in ckpt,
        "pred_depth": count_predictor_blocks(predictor if isinstance(predictor, dict) else {}),
        "embed_dim": infer_embed_dim(encoder if isinstance(encoder, dict) else {}),
        "crop_size": infer_crop_size(encoder if isinstance(encoder, dict) else {}, patch_size=patch_size),
    }


def run_doctor(
    checkpoint: Path,
    config: Path,
    *,
    patch_size: int | None = None,
) -> dict[str, Any]:
    cfg = _load_yaml(config)
    patch = patch_size or int(_nested_get(cfg, "model", "patch_size", default=16) or 16)
    meta = inspect_checkpoint(checkpoint, patch_size=patch)

    expected_crop = _nested_get(cfg, "data", "crop_size") or cfg.get("crop_size")
    expected_depth = _nested_get(cfg, "model", "pred_depth") or cfg.get("pred_depth")
    expected_embed = _nested_get(cfg, "model", "embed_dim") or cfg.get("embed_dim")

    gates: list[dict[str, Any]] = []

    if expected_crop is not None and meta["crop_size"] is not None:
        ok = int(expected_crop) == int(meta["crop_size"])
        gates.append(
            {
                "name": "crop_size",
                "passed": ok,
                "detail": f"config={expected_crop} checkpoint≈{meta['crop_size']} (from pos_embed)",
            }
        )
    else:
        gates.append(
            {
                "name": "crop_size",
                "passed": True,
                "detail": "skipped — missing crop_size in config or could not infer from checkpoint",
                "skipped": True,
            }
        )

    if expected_depth is not None and meta["pred_depth"] is not None:
        ok = int(expected_depth) == int(meta["pred_depth"])
        gates.append(
            {
                "name": "pred_depth",
                "passed": ok,
                "detail": f"config={expected_depth} checkpoint blocks={meta['pred_depth']}",
            }
        )
    else:
        gates.append(
            {
                "name": "pred_depth",
                "passed": True,
                "detail": "skipped — missing pred_depth in config or checkpoint",
                "skipped": True,
            }
        )

    if expected_embed is not None and meta["embed_dim"] is not None:
        ok = int(expected_embed) == int(meta["embed_dim"])
        gates.append(
            {
                "name": "embed_dim",
                "passed": ok,
                "detail": f"config={expected_embed} checkpoint={meta['embed_dim']}",
            }
        )
    else:
        gates.append(
            {
                "name": "embed_dim",
                "passed": True,
                "detail": "skipped — missing embed_dim in config or checkpoint",
                "skipped": True,
            }
        )

    if meta["has_ema_encoder"] and not meta["has_target_encoder"]:
        gates.append(
            {
                "name": "checkpoint_keys",
                "passed": True,
                "detail": "checkpoint uses ema_encoder (distilled release) — map to target_encoder when loading",
                "warn": True,
            }
        )

    active = [g for g in gates if not g.get("skipped")]
    passed = all(g["passed"] for g in active) if active else True

    return {
        "checkpoint": meta,
        "config": str(config.resolve()),
        "gates": gates,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 checkpoint/config doctor")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_doctor(args.checkpoint, args.config)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"V-JEPA 2.1 doctor — {'PASS' if report['passed'] else 'FAIL'}")
        for g in report["gates"]:
            mark = "PASS" if g["passed"] else "FAIL"
            if g.get("skipped"):
                mark = "SKIP"
            print(f"  [{mark}] {g['name']}: {g['detail']}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
