# src/siml/energy_gate.py
from __future__ import annotations
from typing import Any, Callable, Optional, Sequence, Union

_BIG = 1e9

_Array = Union["list[float]", "list[list[float]]"]  # runtime-only; no typing deps

def _to_list2(a_seq: Any) -> list[list[float]]:
    """
    Normalize candidate action sequence to list-of-lists [[...], ...].
    Accepts list, tuple, or numpy arrays; H×D or D (H=1) are fine.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # noqa

    if a_seq is None:
        return []
    if isinstance(a_seq, (list, tuple)):
        if len(a_seq) > 0 and isinstance(a_seq[0], (list, tuple, float, int)):
            # H×D or D
            if len(a_seq) > 0 and isinstance(a_seq[0], (float, int)):
                # D → [D]
                return [list(map(float, a_seq))]
            # H×D
            return [list(map(float, x)) for x in a_seq]
        return []
    # numpy
    if np is not None and isinstance(a_seq, np.ndarray):
        if a_seq.ndim == 1:
            return [a_seq.astype("float32").tolist()]
        return a_seq.astype("float32").tolist()
    return []

class EnergyWithGate:
    """
    Hybrid energy wrapper.

    - mode="off": pure base energy (JEPA Eq.5).
    - mode="fepgate": EARLY gate: call SIML first; if rejected & hard_gate=True, skip base predictor
                      entirely (saves compute). If soft (hard_gate=False), add λ*(1-ok).

    This class is side-effect free; online SIML memory updates are handled in the MPC scaffold
    *after* the best action is chosen.
    """

    def __init__(
        self,
        base_predictor: Callable[..., float],
        mode: str = "off",
        lambda_penalty: float = 0.5,
        hard_gate: bool = False,
        siml_ctx: Any = None,
        tau: Optional[float] = None,
    ):
        self.base_predictor = base_predictor
        self.mode = (mode or "off").lower()
        self.lambda_penalty = float(lambda_penalty)
        self.hard_gate = bool(hard_gate)
        self.siml_ctx = siml_ctx
        self.tau = tau

    def _base(self, a_seq, z_k, s_k, z_g) -> float:
        return float(self.base_predictor(a_seq, z_k, s_k, z_g))

    def score(self, a_seq, z_k, s_k, z_g) -> float:
        if self.mode in ("off", "grip"):
            return self._base(a_seq, z_k, s_k, z_g)

        if self.mode == "fepgate":
            # If SIML isn't usable, fall back to base.
            if (self.siml_ctx is None) or (not getattr(self.siml_ctx, "ready", lambda: False)()) or (self.tau is None):
                return self._base(a_seq, z_k, s_k, z_g)

            # call surprise gate FIRST (cheap), to decide whether to compute base (expensive)
            try:
                from siml.siml_gate import fep_ok  # lightweight shim
                ok_bit, _surpr = fep_ok(_to_list2(a_seq), self.siml_ctx, s_k, float(self.tau))
            except Exception:
                # if the gate crashes, never block planning
                ok_bit = 1

            if self.hard_gate and ok_bit == 0:
                # SKIP base predictor entirely → compute savings
                return _BIG

            base = self._base(a_seq, z_k, s_k, z_g)
            # soft penalty if not ok
            return base + self.lambda_penalty * (1 - ok_bit)

        # unknown mode → base
        return self._base(a_seq, z_k, s_k, z_g)
