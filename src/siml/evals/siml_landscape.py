"""
Fig.9-style (Δx, Δy) energy landscape sweep with Δz = 0.

Reads config YAML (same keys you already use), builds EnergyWithGate with
your base Eq.(5) energy, then evaluates a grid of actions:
  a = [dx, dy, 0, 0, 0, 0, grip]

Outputs:
  - CSV: dx, dy, energy, surprise (if SIML ready), ok_bit (if SIML ready)
  - PNG: heatmap via OpenCV (no matplotlib dependency)

Run:
  python -m siml.evals.siml_landscape --config siml/configs/eval/siml_landscape.yaml --binary_mode off
  python -m siml.evals.siml_landscape --config siml/configs/eval/siml_landscape.yaml --binary_mode fepgate --tau 0.3
"""
from __future__ import annotations
import os, sys, argparse, csv
from typing import Any, Dict
import numpy as np
import cv2

_THIS = os.path.abspath(os.path.dirname(__file__))
_SRC  = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import yaml
except Exception:
    yaml = None

from siml.energy_gate import EnergyWithGate
from siml.siml_agent_io import SimlContext
from siml.repo_base_energy import make_base_energy_from_repo

def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not yaml: return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SIML Fig.9 landscape sweep (Δx,Δy, Δz=0)")
    ap.add_argument("--config", type=str, default="", help="Path to a YAML config.")
    ap.add_argument("--binary_mode", type=str, default=None, choices=["off", "grip", "fepgate"])
    ap.add_argument("--gate_hard", action="store_true")
    ap.add_argument("--lambda_penalty", type=float, default=None)
    ap.add_argument("--agent_memory", type=str, default=None)
    ap.add_argument("--surprise_entry", type=str, default=None, help="module.path:callable")
    ap.add_argument("--tau", type=float, default=None)

    # optional z_k / z_g latents overrides
    ap.add_argument("--z_k", type=str, default=None, help=".npy latent for start frame")
    ap.add_argument("--z_g", type=str, default=None, help=".npy latent for goal image")

    return ap.parse_args()

def _merge_cli(cfg: Dict[str, Any], ns: argparse.Namespace) -> Dict[str, Any]:
    def _ovr(k, v): 
        if v is not None: cfg[k] = v
    for k in ("binary_mode","gate_hard","lambda_penalty","agent_memory","surprise_entry","tau"):
        _ovr(k, getattr(ns, k))
    if ns.z_k: cfg.setdefault("latents", {})["z_k_path"] = ns.z_k
    if ns.z_g: cfg.setdefault("latents", {})["z_g_path"] = ns.z_g
    return cfg

def _load_latent(path: str, fallback: float) -> np.ndarray:
    if path and os.path.exists(path):
        arr = np.load(path)
        return arr.astype(np.float32)
    # fallback latent (constant tensor) so code runs; replace with real latents
    return np.full((16,16,1408), fallback, dtype=np.float32)

def _write_png_from_grid(values: np.ndarray, out_png: str):
    v = values.copy()
    v -= v.min()
    vmax = v.max() + 1e-12
    v = (v / vmax * 255.0).astype(np.uint8)
    v = cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, v)

def main():
    ns = _parse_args()
    cfg = _merge_cli(_load_yaml(ns.config), ns)

    # Pull sweep spec
    sweep = cfg.get("sweep", {})
    dx_min = float(sweep.get("dx_min", -0.15))
    dx_max = float(sweep.get("dx_max",  0.15))
    dy_min = float(sweep.get("dy_min", -0.15))
    dy_max = float(sweep.get("dy_max",  0.15))
    steps  = int(sweep.get("steps", 61))

    # IO
    io = cfg.get("io", {})
    out_dir = io.get("out_dir", "siml/results/landscape")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"landscape_{cfg.get('binary_mode','off')}.csv")
    out_png = os.path.join(out_dir, f"landscape_{cfg.get('binary_mode','off')}.png")

    # Build SIML context and base Eq.(5) energy
    siml_ctx = SimlContext.from_cfg(cfg) if hasattr(SimlContext, "from_cfg") else SimlContext(cfg.get("agent_memory","./agent_memory.bin"))
    base_predictor = make_base_energy_from_repo(cfg)
    energy = EnergyWithGate(
        base_predictor=base_predictor,
        mode=cfg.get("binary_mode","off"),
        lambda_penalty=float(cfg.get("lambda_penalty",0.5)),
        hard_gate=bool(cfg.get("gate_hard", False)),
        siml_ctx=siml_ctx,
        tau=cfg.get("tau", None),
    )

    # Latents
    lat = cfg.get("latents", {})
    z_k = _load_latent(lat.get("z_k_path"), fallback=0.0)
    z_g = _load_latent(lat.get("z_g_path"), fallback=1.0)
    s_k = cfg.get("s_k", {"grip": 0})

    dxs = np.linspace(dx_min, dx_max, steps, dtype=np.float32)
    dys = np.linspace(dy_min, dy_max, steps, dtype=np.float32)
    grid = np.zeros((steps, steps), dtype=np.float32)

    # CSV header
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dx","dy","energy","surprise","ok_bit"])
        for iy, dy in enumerate(dys):
            for ix, dx in enumerate(dxs):
                a_seq = [[float(dx), float(dy), 0.0, 0.0, 0.0, 0.0, 0.0]]  # Δz = 0 ; grip=0
                e = float(energy.score(a_seq, z_k, s_k, z_g))
                grid[iy, ix] = e
                ok_bit, surpr = (1, float("nan"))
                if cfg.get("binary_mode") == "fepgate" and siml_ctx.ready() and (cfg.get("tau") is not None):
                    from siml.siml_gate import fep_ok
                    ok_bit, surpr = fep_ok(a_seq, siml_ctx, s_k, float(cfg["tau"]))
                w.writerow([f"{dx:.6f}", f"{dy:.6f}", f"{e:.8f}", f"{surpr:.8f}", ok_bit])

    _write_png_from_grid(grid, out_png)
    print(f"[siml_landscape] Wrote {out_csv}")
    print(f"[siml_landscape] Wrote {out_png}")

if __name__ == "__main__":
    main()
