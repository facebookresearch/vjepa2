# src/siml/surprise/siml_surprise.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from siml.agent_bridge import get_siml_backend

def fep_surprise(a_seq: np.ndarray, z_k: np.ndarray, s_k: np.ndarray, cfg: Dict[str, Any]) -> Tuple[float, bool]:
    tau = float(cfg.get("tau", 0.06))
    siml = get_siml_backend(cfg)
    s = siml.surprise(a_seq=a_seq, z_k=z_k, s_k=s_k)
    return float(s), bool(s < tau)

# keep your old import path working
# src/siml/surprise/pickplace_surprise.py
# from .siml_surprise import fep_surprise as fep_surprise_pickplace
