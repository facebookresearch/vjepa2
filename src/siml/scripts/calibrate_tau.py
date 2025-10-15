#!/usr/bin/env python3
import argparse, json, os
import numpy as np, yaml
from tqdm import trange
from siml.surprise.cli_bridge import get_siml_backend

def _load_z(path: str, shape=(16,16,1408), fill=0.0):
    if path and os.path.exists(path):
        return np.load(path).astype(np.float32)
    return np.full(shape, fill, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="siml/configs/eval/siml_reaching.yaml")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--percentile", type=float, default=10.0)
    ap.add_argument("--warm_steps", type=int, default=256)
    ap.add_argument("--z_noise_sigma", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    siml = get_siml_backend(cfg)

    lat = (cfg.get("latents") or {})
    z_k = _load_z(lat.get("z_k_path"), fill=0.0)

    # warm-up memory around z_k
    for _ in range(args.warm_steps):
        siml.step_update(z_k, np.zeros((7,), np.float32))

    vals = []
    for _ in trange(args.samples, desc="obs-driven"):
        z_obs = z_k + np.random.normal(0, args.z_noise_sigma, z_k.shape).astype(np.float32)
        vals.append(siml.surprise(np.zeros((1,7), np.float32), z_obs, None))

    vals = np.asarray(vals, np.float32)
    tau = float(np.percentile(vals, args.percentile))
    out = {
        "tau": tau,
        "percentile": args.percentile,
        "N": int(args.samples),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "mode": "obs",
        "warm_steps": int(args.warm_steps),
        "sigma": float(args.z_noise_sigma),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
