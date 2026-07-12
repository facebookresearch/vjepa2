"""Tests for scripts/vjepa2_doctor.py"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from vjepa2_doctor import (  # noqa: E402
    _get_config_value,
    infer_crop_size,
    infer_embed_dim,
    inspect_checkpoint,
    run_doctor,
)


def _write_ckpt(path: Path, *, crop_tokens: int, pred_blocks: int, embed_dim: int = 64):
    n = crop_tokens * crop_tokens
    encoder = {"pos_embed": torch.zeros(1, n, embed_dim)}
    predictor = {}
    for i in range(pred_blocks):
        predictor[f"blocks.{i}.weight"] = torch.zeros(1)
    ckpt = {"ema_encoder": encoder, "encoder": encoder, "predictor": predictor}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _write_cfg(path: Path, *, crop_size: int, pred_depth: int, embed_dim: int = 64):
    data = {
        "data": {"crop_size": crop_size},
        "model": {"pred_depth": pred_depth, "embed_dim": embed_dim, "patch_size": 16},
    }
    path.write_text(yaml.dump(data), encoding="utf-8")


def test_infer_crop_size_from_pos_embed():
    state = {"pos_embed": torch.zeros(1, 24 * 24, 32)}
    assert infer_crop_size(state, patch_size=16) == 384


def test_infer_crop_size_returns_first_valid_pos_embed():
    encoder = {"pos_embed": torch.zeros(1, 16 * 16, 32)}
    predictor = {"predictor.pos_embed": torch.zeros(1, 24 * 24, 32)}
    state = {**encoder, **predictor}
    assert infer_crop_size(state, patch_size=16) == 256


def test_infer_embed_dim_from_pos_embed():
    state = {"blocks.0.pos_embed": torch.zeros(1, 16 * 16, 128)}
    assert infer_embed_dim(state) == 128


def test_get_config_value_preserves_zero():
    cfg = {"data": {"crop_size": 0}, "model": {"pred_depth": 0}}
    assert _get_config_value(cfg, "data", "crop_size", flat_key="crop_size") == 0
    assert _get_config_value(cfg, "model", "pred_depth", flat_key="pred_depth") == 0


def test_doctor_passes_matching_config(tmp_path):
    ckpt = tmp_path / "ok.pt"
    cfg = tmp_path / "ok.yaml"
    _write_ckpt(ckpt, crop_tokens=24, pred_blocks=12)
    _write_cfg(cfg, crop_size=384, pred_depth=12)
    report = run_doctor(ckpt, cfg)
    assert report["passed"]
    assert all(g["passed"] for g in report["gates"] if not g.get("skipped"))


def test_doctor_fails_crop_mismatch(tmp_path):
    ckpt = tmp_path / "bad.pt"
    cfg = tmp_path / "bad.yaml"
    _write_ckpt(ckpt, crop_tokens=24, pred_blocks=12)  # 384
    _write_cfg(cfg, crop_size=256, pred_depth=12)
    report = run_doctor(ckpt, cfg)
    assert not report["passed"]
    crop = next(g for g in report["gates"] if g["name"] == "crop_size")
    assert not crop["passed"]


def test_doctor_fails_pred_depth_mismatch(tmp_path):
    ckpt = tmp_path / "depth.pt"
    cfg = tmp_path / "depth.yaml"
    _write_ckpt(ckpt, crop_tokens=16, pred_blocks=12)
    _write_cfg(cfg, crop_size=256, pred_depth=24)
    report = run_doctor(ckpt, cfg)
    assert not report["passed"]


def test_doctor_rejects_non_integer_crop_size(tmp_path):
    ckpt = tmp_path / "ck.pt"
    cfg = tmp_path / "cfg.yaml"
    _write_ckpt(ckpt, crop_tokens=16, pred_blocks=4)
    cfg.write_text(
        yaml.dump({"data": {"crop_size": "384px"}, "model": {"pred_depth": 4, "patch_size": 16}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="crop_size"):
        run_doctor(ckpt, cfg)


def test_inspect_checkpoint_keys(tmp_path):
    ckpt = tmp_path / "keys.pt"
    _write_ckpt(ckpt, crop_tokens=16, pred_blocks=4)
    meta = inspect_checkpoint(ckpt)
    assert meta["has_ema_encoder"]
    assert meta["pred_depth"] == 4
