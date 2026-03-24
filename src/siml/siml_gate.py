"""
Gate helpers for EnergyWithGate.
"""

from typing import Tuple

def fep_ok(a_seq, siml_ctx, s_k, tau: float) -> Tuple[int, float]:
    """
    Threshold SIML surprise at tau.
    Returns:
      (ok_bit, surprise_value)
      ok_bit = 1 if surprise <= tau else 0
    """
    try:
        surpr = float(siml_ctx.surprise(a_seq, s_k, extras=None))
    except Exception:
        # When SIML isn't ready, treat as always OK (no-op gating). Surpr = NaN-like.
        return 1, float("nan")
    ok = 1 if surpr <= float(tau) else 0
    return ok, surpr

def grip_bit(s_k, extras=None) -> int:
    """
    Extract {0,1} from current state. If env packs gripper as s_k['grip'],
    use its boolean value; otherwise default to 0.
    """
    try:
        return int(bool(s_k.get("grip", 0)))
    except Exception:
        return 0