from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from siml.envs.pickplace_obs import PickPlaceObs, advance_state_once, DEFAULT_L1_RADIUS

def fep_surprise_pickplace(a_seq: List[List[float]], s_k: Dict[str, Any], extras: Optional[dict] = None) -> float:
    """H=1 surprise: if gripped → ||cube_{t+1}-pad||_2 else ||ee_{t+1}-cube||_2."""
    obs = PickPlaceObs.from_s_k(s_k)
    step = a_seq[0] if len(a_seq) else [0,0,0,0]
    l1r = float((extras or {}).get("l1_radius", DEFAULT_L1_RADIUS))
    nxt = advance_state_once(obs, step, l1_radius=l1r)
    if nxt.grip_bool:
        return float(np.linalg.norm(nxt.cube - nxt.pad, ord=2))
    else:
        return float(np.linalg.norm(nxt.ee   - nxt.cube, ord=2))
