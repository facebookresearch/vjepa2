from __future__ import annotations
import json, os, subprocess, tempfile
from typing import Any, Dict, Optional
import numpy as np

def _ensure_f32(x):
    return np.asarray(x, dtype=np.float32)

def _robust_run(cmd):
    def _exec(a):
        p = subprocess.run(a, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    rc, out, err = _exec(cmd)
    if rc == 0:
        return out

    # Drop flags some builds don’t support
    if "--mode" in cmd and ("unexpected argument '--mode'" in err or "found argument '--mode'" in err):
        i = cmd.index("--mode"); cmd = cmd[:i] + cmd[i+2:]
        rc, out, err = _exec(cmd)
        if rc == 0: return out
    if "--act" in cmd and ("unexpected argument '--act'" in err or "found argument '--act'" in err):
        i = cmd.index("--act"); cmd = cmd[:i] + cmd[i+2:]
        rc, out, err = _exec(cmd)
        if rc == 0: return out

    raise subprocess.CalledProcessError(rc, cmd, out + "\n" + err)

def _json_or_outfile(bin_path, subcmd, obs_p, act_p=None, mode=None, out_p=None):
    # Prefer JSON printed to stdout; fall back to writing --out
    cmd = [bin_path, subcmd, "--obs", obs_p]
    if act_p is not None: cmd += ["--act", act_p]
    if mode is not None:  cmd += ["--mode", mode]
    try:
        out = _robust_run(cmd).strip()
        if out.startswith("{"):
            return json.loads(out)
    except subprocess.CalledProcessError:
        pass

    if out_p is None:
        raise RuntimeError("SIML CLI didn’t print JSON and no --out provided")

    cmd = [bin_path, subcmd, "--obs", obs_p, "--out", out_p]
    if act_p is not None: cmd += ["--act", act_p]
    if mode is not None:  cmd += ["--mode", mode]
    _ = _robust_run(cmd)
    with open(out_p, "r") as f:
        return json.load(f)

def _normalize_surprise(raw: float) -> float:
    """
    Map arbitrary-scale surprise → [0,1], monotone & identity near 0.
    """
    if np.isnan(raw) or raw < 0:
        return 1.0
    return float(1.0 - np.exp(-float(raw)))  # small x ~ x, large x -> 1

class _CliBackend:
    def __init__(self, bin_path: str, mem_path: str, mode: str, autosave_every: int):
        self.bin = bin_path
        self.mem = mem_path
        self.mode = mode
        self.autosave_every = autosave_every
        os.environ.setdefault("SIML_MEMORY", os.path.abspath(mem_path))
        os.makedirs(os.path.dirname(mem_path), exist_ok=True)

    def surprise(self, a_seq, z_k, s_k):
        a = _ensure_f32(a_seq)[0] if a_seq is not None else np.zeros((7,), np.float32)
        z = _ensure_f32(z_k)
        with tempfile.TemporaryDirectory() as td:
            op = os.path.join(td, "obs.npy"); np.save(op, z)
            ap = os.path.join(td, "act.npy"); np.save(ap, a)
            js = _json_or_outfile(self.bin, "surprise", op, act_p=ap, mode=self.mode, out_p=os.path.join(td,"out.json"))
            return _normalize_surprise(float(js.get("surprise", 1.0)))

    def step_update(self, obs, act):
        with tempfile.TemporaryDirectory() as td:
            op = os.path.join(td, "obs.npy"); np.save(op, _ensure_f32(obs))
            ap = os.path.join(td, "act.npy"); np.save(ap, _ensure_f32(act))
            _json_or_outfile(self.bin, "step", op, act_p=ap, mode="online", out_p=os.path.join(td,"out.json"))
            print("[cli_bridge] step ok")

def get_siml_backend(cfg: Dict[str, Any]):
    s = cfg.get("siml", {})
    if s.get("backend", "cli") != "cli":
        raise RuntimeError("Only 'cli' backend supported here.")
    bin_path = s.get("runtime_bin") or os.getenv("SIML_RUNTIME_BIN")
    if not bin_path: raise RuntimeError("siml.runtime_bin not set")
    mem      = s.get("agent_memory", "siml/memory/vjepa2_ac.mem")
    mode     = s.get("memory_mode", "online")
    autosave = int(s.get("autosave_every", 0))
    return _CliBackend(bin_path, mem, mode, autosave)

# Optional one-line updater used by MPC scaffold (best-effort).
def siml_online_step(*, z_observed, z_k, cfg: Dict[str, Any]):
    try:
        siml = get_siml_backend(cfg)
        siml.step_update(z_observed, np.zeros((7,), np.float32))
    except Exception as e:
        print(f"[cli_bridge] step skipped: {e}")
