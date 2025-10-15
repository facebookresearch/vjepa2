# src/siml/evals/scaffold.py
from __future__ import annotations
import math, importlib
from typing import Any, Dict
import numpy as np

# Optional online SIML step (best-effort; never blocks)
try:
    from siml.surprise.cli_bridge import siml_online_step
except Exception:
    def siml_online_step(**_):  # type: ignore
        pass

def _load_entrypoint(entry: str):
    if ":" not in entry:
        raise ValueError(f'Entrypoint must be "module.path:callable", got {entry!r}')
    mod, name = entry.split(":", 1)
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise TypeError(f"{entry!r} is not callable")
    return fn

# --------- Minimal action-aware predictor that returns z_pred (3-arg signature) ----------
class _LatentTiltPredictor:
    """
    Returns z_pred with a tiny, action-dependent tilt on channel 0.
    Signature matches repo_base_energy: (a_seq, z_k, s_k) -> z_pred.
    """
    def __call__(self, a_seq, z_k, s_k):
        z_k = np.asarray(z_k, dtype=np.float32)
        # normalize a_seq to [H,D], take first step
        a = np.asarray(a_seq, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, ...]
        dx = float(a[0, 0]) if a.shape[1] >= 1 else 0.0
        dy = float(a[0, 1]) if a.shape[1] >= 2 else 0.0

        z_pred = z_k.copy()
        # ensure last dim exists; assume (..., C) with C>=1; if flat, just add to the whole array
        if z_pred.ndim >= 1 and z_pred.shape[-1] >= 1:
            z_pred[..., 0] = z_pred[..., 0] + (6.0 * dx - 6.0 * dy) * 1e-2
        else:
            z_pred[...] = z_pred[...] + (6.0 * dx - 6.0 * dy) * 1e-2
        return z_pred

def make_predictor(cfg: Dict[str, Any]):
    # Swap to your real V-JEPA2 AC forecaster later; keep same 3-arg signature.
    return _LatentTiltPredictor()

# ------------------ CEM with L1-ball constraint ------------------
def _sample_l1_ball(dim: int, r: float, n: int) -> np.ndarray:
    u = np.random.laplace(loc=0.0, scale=1.0, size=(n, dim)).astype(np.float32)
    norms = np.maximum(np.sum(np.abs(u), axis=1, keepdims=True), 1e-6)
    u = u / norms
    rad = (np.random.rand(n, 1).astype(np.float32) ** (1.0 / dim)) * r
    return u * rad

def run_mpc(score_fn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {"a": HxD numpy array, "E": best_energy}.
    Also performs a non-blocking SIML online step using z_k as a surrogate observation.
    """
    H   = int(cfg.get("horizon", 1))
    D   = int(cfg.get("action_dim", 7))
    r   = float(cfg.get("action_l1_ball", 0.075))
    cem = cfg.get("cem", {}) or {}
    S   = int(cem.get("samples", 128))
    I   = int(cem.get("iters", 5))
    topk = max(8, S // 4)

    mu  = np.zeros((H, D), dtype=np.float32)
    sig = np.full((H, D), 0.5 * r, dtype=np.float32)

    best_E = float("inf")
    best_A = None

    for _ in range(I):
        A = np.random.normal(loc=mu, scale=sig, size=(S, H, D)).astype(np.float32)
        # project steps into L1-ball
        for s in range(S):
            for h in range(H):
                if np.sum(np.abs(A[s, h])) > r:
                    A[s, h] = _sample_l1_ball(D, r, 1)[0]

        Es = np.empty((S,), dtype=np.float32)
        for s in range(S):
            Es[s] = float(score_fn(A[s]))

        # --- SIML gate on elites (soft/hard), applied BEFORE selecting best/incumbent ---
        try:
            from siml.siml_gate import fep_ok
            tau = cfg.get("tau", None)
            if tau is not None and cfg.get("_siml_ctx_obj", None) is not None:
                # candidate elite set based on current base energy
                elite_idx = np.argsort(Es)[:topk]

                # pull context for the gate (only 4D signature used)
                ctx = cfg.get("__siml_ctx", {})
                s_k = ctx.get("s_k", None)

                lam  = float(cfg.get("lambda_penalty", 0.5))
                hard = bool(cfg.get("gate_hard", False))
                for i_el in elite_idx:
                    a_vec = A[i_el, 0]                # first step only
                    grip  = a_vec[6] if a_vec.shape[0] >= 7 else 0.0
                    a4    = [float(a_vec[0]), float(a_vec[1]), float(a_vec[2]), float(grip)]
                    ok, _ = fep_ok([a4], cfg["_siml_ctx_obj"], s_k, float(tau))
                    if not ok:
                        Es[i_el] = Es[i_el] + (1e9 if hard else lam)
        except Exception as e:
            print(f"[scaffold.run_mpc] SIML gate on elites skipped: {e}")


        i = int(np.argmin(Es))
        if Es[i] < best_E:
            best_E = float(Es[i])
            best_A = A[i].copy()

        elite = A[np.argsort(Es)[:topk]]
        mu  = np.mean(elite, axis=0)
        sig = np.std(elite, axis=0) + 1e-6

    if best_A is None:
        best_A = np.zeros((H, D), dtype=np.float32)

    print(f"[scaffold.run_mpc] Best energy: {best_E:.6f}")

    # best-effort online SIML step (surrogate: z_observed = z_k)
    try:
        ctx  = cfg.get("__siml_ctx", {})
        z_k  = ctx.get("z_k", None)
        s_k  = ctx.get("s_k", None)
        if z_k is not None:
            predictor = _load_entrypoint(cfg.get("predictor_entry","siml.evals.scaffold:make_predictor"))(cfg)
            # best_A is HxD; if your predictor expects one step, keep best_A[0]
            z_pred_star = predictor(best_A, z_k, s_k)
            # use the *observed* latent surrogate as z_pred_star
            siml_online_step(z_observed=z_pred_star, z_k=z_k, cfg=cfg)
    except Exception as e:
        print(f"[scaffold.run_mpc] SIML online update skipped: {e}")
    
    return {"a": best_A, "E": best_E}
