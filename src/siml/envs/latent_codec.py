# src/siml/envs/latent_codec.py
import numpy as np

def encode_latent(obs) -> np.ndarray:
    """
    Cheap, deterministic surrogate encoder:
    maps PickPlaceObs -> (16,16,1408) latent with smooth structure so that
    SIML can learn online from actual observations.
    """
    z = np.zeros((16, 16, 1408), dtype=np.float32)

    # channel 0..2 carry ee-pad vector (scaled)
    dv = (obs.ee - obs.pad)
    z[..., 0] = dv[0] * 10.0
    z[..., 1] = dv[1] * 10.0
    z[..., 2] = dv[2] * 10.0

    # channel 3 = grip bit
    z[..., 3] = 1.0 if getattr(obs, "grip_bool", 0) else 0.0

    # low-freq spatial structure (helps SIML cluster)
    xs = np.linspace(-1, 1, 16, dtype=np.float32)
    grid = np.outer(xs, xs).astype(np.float32)
    z[..., 4] = grid * (obs.ee[0] * 20.0)
    z[..., 5] = grid * (obs.ee[1] * 20.0)
    z[..., 6] = grid * (obs.ee[2] * 20.0)

    return z
