# src/siml/evals/reach_eval.py
"""
Closed-loop reaching evals (JEPA-only vs FEP-gate).
Outputs:
  siml/results/reaching/off/{metrics.csv,summary.json,traces/epXXXXX.csv}
  siml/results/reaching/fepgate/{metrics.csv,summary.json,traces/epXXXXX.csv}
"""

from __future__ import annotations
import os, sys, time, csv, json, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import importlib

_THIS = os.path.abspath(os.path.dirname(__file__))         # .../src/siml/evals
_SRC  = os.path.abspath(os.path.join(_THIS, "..", ".."))   # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import yaml
except Exception:
    yaml = None

from siml.repo_base_energy import make_base_energy_from_repo
from siml.energy_gate import EnergyWithGate
from siml.siml_agent_io import SimlContext
from siml.envs.pickplace_obs import PickPlaceObs, advance_state_once, DEFAULT_L1_RADIUS

def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not yaml:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_entrypoint(entry: str):
    if ":" not in entry:
        raise ValueError(f"Entrypoint must be 'module.path:callable', got {entry!r}")
    mod_name, func_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"Entrypoint {entry!r} is not callable")
    return fn

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _np_load_or_const(path: Optional[str], fill: float) -> np.ndarray:
    if path and os.path.exists(path):
        return np.load(path).astype(np.float32)
    return np.full((16, 16, 1408), fill, dtype=np.float32)

def _action_4_from_seq(a_seq: List[List[float]]) -> List[float]:
    if not a_seq:
        return [0.0, 0.0, 0.0, 0.0]
    vec = a_seq[0]
    if len(vec) >= 7:
        return [float(vec[0]), float(vec[1]), float(vec[2]), float(vec[6])]
    out = [0.0, 0.0, 0.0, 0.0]
    for i in range(min(4, len(vec))):
        out[i] = float(vec[i])
    return out

@dataclass
class EpMetrics:
    ep: int
    steps: int
    final_err_m: float
    monotonicity: float
    steps_to_tol: int
    avg_latency_ms: float
    mode: str

def _episode_states(cfg: Dict[str, Any]) -> Tuple[PickPlaceObs, np.ndarray, np.ndarray]:
    sk = cfg.get("s_k", None)
    if sk is None:
        rng = np.random.default_rng()
        sk = {
            "ee":   [float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.03, 0.03)), 0.05],
            "cube": [0.05, -0.10, 0.05],
            "pad":  [0.00, -0.10, 0.05],
            "grip": 0,
        }
    obs0 = PickPlaceObs.from_s_k(sk)
    lat = cfg.get("latents", {})
    z_k = _np_load_or_const(lat.get("z_k_path"), fill=0.0)
    z_g = _np_load_or_const(lat.get("z_g_path"), fill=1.0)
    return obs0, z_k, z_g

def _summarize(metrics: List[EpMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {}
    arr = np.array([[m.final_err_m, m.monotonicity, m.steps_to_tol, m.avg_latency_ms] for m in metrics], dtype=np.float32)
    def s(col):
        return {"mean": float(np.nanmean(arr[:, col])), "std": float(np.nanstd(arr[:, col]))}
    return {"N": len(metrics), "final_err_m": s(0), "monotonicity": s(1), "steps_to_tol": s(2), "avg_latency_ms": s(3)}

def run_eval(cfg: Dict[str, Any], mode: str, out_dir: str) -> Dict[str, Any]:
    _ensure_dir(out_dir); _ensure_dir(os.path.join(out_dir, "traces"))

    episodes = int(cfg.get("episodes", 100))
    steps_per_ep = int(cfg.get("reach_steps", cfg.get("steps_per_ep", 6)))
    tol_m = float(cfg.get("goal_tolerance_m", 0.04))
    horizon = int(cfg.get("horizon", 1))
    cem_cfg = cfg.get("cem", {"samples": 128, "iters": 5})
    action_dim = int(cfg.get("action_dim", 7))
    l1_r = float(cfg.get("action_l1_ball", DEFAULT_L1_RADIUS))

    siml_ctx = SimlContext.from_cfg(cfg) if hasattr(SimlContext, "from_cfg") else SimlContext(cfg.get("agent_memory", "./agent_memory.bin"))

    base_predictor = make_base_energy_from_repo(cfg)
    energy = EnergyWithGate(
        base_predictor=base_predictor,
        mode=("fepgate" if mode == "fepgate" else "off"),
        lambda_penalty=float(cfg.get("lambda_penalty", 0.5)),
        hard_gate=bool(cfg.get("gate_hard", False)),
        siml_ctx=siml_ctx,
        tau=cfg.get("tau", None),
    )

    mpc_entry = cfg.get("mpc_entry", "siml.evals.scaffold:run_mpc")
    run_mpc = _load_entrypoint(mpc_entry)

    met_path = os.path.join(out_dir, "metrics.csv")
    all_metrics: List[EpMetrics] = []
    with open(met_path, "w", newline="", encoding="utf-8") as fmet:
        mwr = csv.writer(fmet)
        mwr.writerow(["ep","mode","steps","final_err_m","monotonicity","steps_to_tol","avg_latency_ms"])

        from siml.agent_bridge import get_siml_backend
        cfg["_siml_ctx_obj"] = get_siml_backend(cfg)
        
        for ep in range(episodes):
            obs, z_k, z_g = _episode_states(cfg)
            s_k = {"ee": obs.ee.tolist(), "cube": obs.cube.tolist(), "pad": obs.pad.tolist(), "grip": int(obs.grip_bool)}

            errs: List[float] = []; dts_ms: List[float] = []
            for _t in range(steps_per_ep):
                step_cfg = dict(cfg)
                step_cfg.update({
                    "horizon": horizon,
                    "cem": cem_cfg,
                    "action_dim": action_dim,
                    "action_l1_ball": l1_r,
                    "io": {"out_dir": None, "save_mp4": False, "save_csv": False},
                })
                # expose siml context
                ctx = step_cfg.setdefault("__siml_ctx", {})
                ctx["z_k"] = z_k
                ctx["s_k"] = s_k

                score_fn = lambda a: energy.score(a, z_k, s_k, z_g)

                t0 = time.perf_counter()
                try:
                    best = run_mpc(score_fn, step_cfg)
                except TypeError:
                    best = run_mpc(score_fn=score_fn, cfg=step_cfg)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                dts_ms.append(dt_ms)

                # accept either 'a_seq' or 'a', handle numpy vs list
                A = best.get("a_seq", None)
                if A is None:
                    A = best.get("a", None)
                if A is None:
                    A_list = []
                elif isinstance(A, np.ndarray):
                    A_list = A.tolist()
                else:
                    A_list = A
                a4 = _action_4_from_seq(A_list)

                obs = advance_state_once(obs, a4, l1_radius=l1_r)
                s_k = {"ee": obs.ee.tolist(), "cube": obs.cube.tolist(), "pad": obs.pad.tolist(), "grip": int(obs.grip_bool)}
                errs.append(float(np.linalg.norm(obs.ee - obs.pad, ord=2)))

            final_err = float(errs[-1]) if errs else float("nan")
            decreases = sum(1 for i in range(1, len(errs)) if errs[i] <= errs[i-1])
            monot = float(decreases) / max(1, len(errs) - 1)
            steps_to = next((i+1 for i, e in enumerate(errs) if e <= tol_m), steps_per_ep+1)
            avg_lat = float(np.mean(dts_ms)) if dts_ms else float("nan")

            m = EpMetrics(ep=ep, steps=steps_per_ep, final_err_m=final_err,
                          monotonicity=monot, steps_to_tol=steps_to,
                          avg_latency_ms=avg_lat, mode=mode)
            all_metrics.append(m)
            mwr.writerow([m.ep, m.mode, m.steps, f"{m.final_err_m:.6f}",
                          f"{m.monotonicity:.3f}", m.steps_to_tol, f"{m.avg_latency_ms:.2f}"])

            tr_path = os.path.join(out_dir, "traces", f"ep{ep:05d}.csv")
            with open(tr_path, "w", newline="", encoding="utf-8") as ftr:
                tw = csv.writer(ftr)
                tw.writerow(["t","err_m","latency_ms"])
                for t, (e, dt) in enumerate(zip(errs, dts_ms)):
                    tw.writerow([t, f"{e:.6f}", f"{dt:.2f}"])

            if (ep + 1) % max(1, episodes // 10) == 0:
                print(f"[reach_eval:{mode}] {ep+1}/{episodes} eps done "
                      f"(final_err={final_err:.3f} m, avg_lat={avg_lat:.1f} ms)", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fs:
        json.dump(_summarize(all_metrics), fs, indent=2)

    return {"metrics_csv": met_path, "summary_json": os.path.join(out_dir, "summary.json")}

# CLI
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Closed-loop reaching evals (JEPA-only vs FEP-gate)")
    ap.add_argument("--config", type=str, default="siml/configs/eval/siml_reaching.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--only", type=str, choices=["off","fepgate","both"], default="both")
    return ap.parse_args()

def _merge_cli(cfg: Dict[str, Any], ns: argparse.Namespace) -> Dict[str, Any]:
    def _ovr(k, v):
        if v is not None:
            cfg[k] = v
    _ovr("episodes", ns.episodes)
    if ns.steps is not None:
        cfg["reach_steps"] = ns.steps
    _ovr("tau", ns.tau)
    if ns.out_dir is not None:
        cfg.setdefault("io", {})["out_dir"] = ns.out_dir
    return cfg

def main():
    ns = _parse_args()
    cfg = _merge_cli(_load_yaml(ns.config), ns)
    print("[reach_eval] Resolved config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")
    root = cfg.get("io", {}).get("out_dir", "results/reaching")

    if ns.only in ("off","both"):
        print("[reach_eval] Running JEPA-only (off)…", flush=True)
        run_eval(cfg, mode="off", out_dir=os.path.join(root, "off"))
    if ns.only in ("fepgate","both"):
        print("[reach_eval] Running FEP-gate…", flush=True)
        run_eval(cfg, mode="fepgate", out_dir=os.path.join(root, "fepgate"))

if __name__ == "__main__":
    main()
