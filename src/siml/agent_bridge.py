# src/siml/agent_bridge.py
from __future__ import annotations
import json, os, subprocess, tempfile
from typing import Any, Dict, Optional
import numpy as np

__all__ = ["get_siml_backend", "SimlBackend"]

_singleton: Optional["SimlBackend"] = None

def _ensure_float32(x): return np.asarray(x, dtype=np.float32)

def _run_robust(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode == 0:
        return p.stdout, p.stderr
    # try removing '--mode' if binary doesn't support it
    if "--mode" in cmd and ("unexpected argument '--mode'" in p.stderr or "found argument '--mode'" in p.stderr):
        i = cmd.index("--mode"); cmd2 = cmd[:i] + cmd[i+2:]
        p2 = subprocess.run(cmd2, capture_output=True, text=True, env=env)
        if p2.returncode == 0:
            return p2.stdout, p2.stderr
        raise subprocess.CalledProcessError(p2.returncode, cmd2, output=p2.stdout, stderr=p2.stderr)
    raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)

class _CliShim:
    """
    CLI shim to SIML runtime.
    Assumes the binary discovers memory via SIML_MEMORY env var (no flags).
    """
    def __init__(self, bin_path: str, mem_path: str, mode: str, autosave_every: int):
        self.bin = bin_path
        self.mem = mem_path
        self.mode = mode
        self.autosave_every = autosave_every
        self._steps = 0
        os.makedirs(os.path.dirname(self.mem), exist_ok=True)
        os.environ.setdefault("SIML_MEMORY", os.path.abspath(self.mem))

    def surprise(self, a_seq: np.ndarray, z_k: np.ndarray, s_k: Optional[np.ndarray]) -> float:
        # We only need z_k (obs) and optionally a single-step action (a0)
        a0 = _ensure_float32(a_seq[0]) if a_seq is not None else np.zeros((7,), np.float32)
        z  = _ensure_float32(z_k)
        with tempfile.TemporaryDirectory() as td:
            obs_p = os.path.join(td, "obs.npy")
            act_p = os.path.join(td, "act.npy")
            np.save(obs_p, z)
            np.save(act_p, a0)

            # Prefer stdout JSON
            cmd = [self.bin, "surprise", "--obs", obs_p, "--act", act_p]
            out, _ = _run_robust(cmd)

            out = out.strip()
            if out.startswith("{"):
                js = json.loads(out)
            else:
                # fallback to --out file
                out_p = os.path.join(td, "out.json")
                cmd2 = [self.bin, "surprise", "--obs", obs_p, "--act", act_p, "--out", out_p, "--mode", self.mode]
                _run_robust(cmd2)
                with open(out_p, "r") as f:
                    js = json.load(f)
            return float(js.get("surprise", 1.0))

    def step_update(self, obs: np.ndarray, act: np.ndarray) -> None:
        self._steps += 1
        with tempfile.TemporaryDirectory() as td:
            obs_p = os.path.join(td, "obs.npy")
            act_p = os.path.join(td, "act.npy")
            np.save(obs_p, _ensure_float32(obs))
            np.save(act_p, _ensure_float32(act))
            cmd = [self.bin, "step", "--obs", obs_p, "--act", act_p, "--mode", "online"]
            _run_robust(cmd)

    def stats(self) -> Dict[str, Any]:
        return {"mode": "cli", "steps": self._steps}

class SimlBackend:
    def __init__(self, cfg: Dict[str, Any]):
        s = cfg.get("siml", {})
        if s.get("backend", "mock") != "cli":
            raise RuntimeError("Only 'cli' backend is supported in this setup.")
        bin_path = s.get("runtime_bin") or os.getenv("SIML_RUNTIME_BIN")
        if not bin_path:
            raise RuntimeError("siml.runtime_bin not set")
        mem = s.get("agent_memory", "siml/memory/vjepa2_ac.mem")
        mode = s.get("memory_mode", "online")
        autosave = int(s.get("autosave_every", 0))
        self.impl = _CliShim(bin_path, mem, mode, autosave)

    def surprise(self, a_seq, z_k, s_k): return self.impl.surprise(a_seq, z_k, s_k)
    def step_update(self, obs, act):      return self.impl.step_update(obs, act)
    def stats(self):                      return self.impl.stats()

def get_siml_backend(cfg: Dict[str, Any]) -> SimlBackend:
    global _singleton
    if _singleton is None:
        _singleton = SimlBackend(cfg)
    return _singleton  # type: ignore
