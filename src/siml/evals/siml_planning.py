# src/siml/evals/siml_planning.py
import os, sys, argparse
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None

_THIS = os.path.abspath(os.path.dirname(__file__))
_SRC  = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from siml.energy_gate import EnergyWithGate
from siml.siml_agent_io import SimlContext
from siml.repo_base_energy import make_base_energy_from_repo
from siml.planner_hook import run_mpc_with_energy

def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not yaml: return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SIML one-bit gate runner")
    ap.add_argument("--config", type=str, default="", help="Path to a YAML config.")
    # Common knobs
    ap.add_argument("--binary_mode", type=str, default=None, choices=["off", "grip", "fepgate"])
    ap.add_argument("--gate_hard", action="store_true")
    ap.add_argument("--lambda_penalty", type=float, default=None)
    ap.add_argument("--agent_memory", type=str, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--cem_samples", type=int, default=None)
    ap.add_argument("--cem_iters", type=int, default=None)
    ap.add_argument("--action_l1_ball", type=float, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    # NEW: explicit entrypoints
    ap.add_argument("--predictor_entry", type=str, default=None, help='e.g. "siml.adapters:make_predictor"')
    ap.add_argument("--mpc_entry", type=str, default=None, help='e.g. "siml.adapters:run_mpc"')
    return ap.parse_args()

def _merge_cli(cfg: Dict[str, Any], ns: argparse.Namespace) -> Dict[str, Any]:
    def _ovr(k, v): 
        if v is not None: cfg[k] = v
    for k in ("binary_mode","gate_hard","lambda_penalty","agent_memory","tau",
              "horizon","cem_samples","cem_iters","action_l1_ball","out_dir",
              "predictor_entry","mpc_entry"):
        _ovr(k, getattr(ns, k))
    return cfg

def main():
    ns  = _parse_args()
    cfg = _merge_cli(_load_yaml(ns.config), ns)

    cfg.setdefault("binary_mode", "off")
    cfg.setdefault("gate_hard", False)
    cfg.setdefault("lambda_penalty", 0.5)
    cfg.setdefault("agent_memory", "./agent_memory.bin")
    cfg.setdefault("tau", None)

    print("[SIML runner] Resolved config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")

    # Build SIML context (prefers from_cfg so we can use surprise_entry)
    if hasattr(SimlContext, "from_cfg"):
        siml_ctx = SimlContext.from_cfg(cfg)
    else:
        siml_ctx = SimlContext(cfg.get("agent_memory", "./agent_memory.bin"))

    base_predictor = make_base_energy_from_repo(cfg)
    energy = EnergyWithGate(
        base_predictor=base_predictor,
        mode=cfg["binary_mode"],
        lambda_penalty=cfg["lambda_penalty"],
        hard_gate=cfg["gate_hard"],
        siml_ctx=siml_ctx,
        tau=cfg["tau"],
    )
    run_mpc_with_energy(energy, cfg)

if __name__ == "__main__":
    main()
