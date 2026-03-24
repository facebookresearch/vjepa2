from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

DEFAULT_L1_RADIUS = 0.075

@dataclass
class PickPlaceObs:
    ee:   np.ndarray  # (3,)
    cube: np.ndarray  # (3,)
    pad:  np.ndarray  # (3,)
    grip_bool: bool

    @staticmethod
    def from_s_k(s_k: Dict) -> "PickPlaceObs":
        def _vec3(d, keys):  # robust key lookup
            for k in keys:
                v = d.get(k)
                if v is not None and len(v) >= 3:
                    return np.asarray(v, dtype=np.float32)[:3]
            raise KeyError(f"Missing position for any of {keys}")
        ee   = _vec3(s_k, ("ee","ee_pos","end_effector","pose","xyz"))
        cube = _vec3(s_k, ("cube","cube_pos","obj","object"))
        pad  = _vec3(s_k, ("pad","pad_pos","target","goal"))
        grip = bool(s_k.get("grip", s_k.get("grip_bool", 0)))
        return PickPlaceObs(ee, cube, pad, grip)

def _clip_l1(v: np.ndarray, r: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if np.sum(np.abs(v)) <= r: return v
    u = np.abs(v); s = np.sort(u)[::-1]; cssv = np.cumsum(s)
    rho = np.nonzero(s * (np.arange(1, len(s)+1)) > (cssv - r))[0][-1]
    theta = (cssv[rho] - r) / float(rho + 1)
    return np.sign(v) * np.maximum(u - theta, 0)

def advance_state_once(obs: PickPlaceObs, a_step: List[float], l1_radius: float = DEFAULT_L1_RADIUS) -> PickPlaceObs:
    dx, dy, dz, grip_delta = float(a_step[0]), float(a_step[1]), float(a_step[2]), float(a_step[-1])
    dpos = _clip_l1(np.array([dx,dy,dz], dtype=np.float32), l1_radius)
    ee_next = obs.ee + dpos
    grip_next = bool(int(obs.grip_bool) or (grip_delta > 0.5))
    cube_next = ee_next.copy() if grip_next else obs.cube.copy()
    return PickPlaceObs(ee_next, cube_next, obs.pad.copy(), grip_next)
