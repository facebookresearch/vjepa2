# src/siml/evals/energy_proxies.py
from __future__ import annotations
import argparse, json, os, glob
import numpy as np
import pandas as pd

try:
    import cv2  # optional, for optical flow
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

def _series_ci(x, alpha=0.05, nboot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = [rng.choice(x, size=x.size, replace=True).mean() for _ in range(nboot)]
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(x.mean()), float(lo), float(hi)

def _perm_test(x, y, nperm=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = y.mean() - x.mean()
    z = np.concatenate([x, y])
    n = x.size
    cnt = 0
    for _ in range(nperm):
        rng.shuffle(z)
        d = z[n:].mean() - z[:n].mean()
        if abs(d) >= abs(obs): cnt += 1
    return float(obs), float((cnt+1)/(nperm+1))

def _energy_from_trace_csv(csv_path, rot_weight=0.25, grip_cost=0.02):
    """
    Expect columns: dx,dy,dz, droll,dpitch,dyaw, grip_bool  (and optionally frame paths)
    Returns dict with action-norm energy, path length, jerk energy, grip count.
    """
    df = pd.read_csv(csv_path)
    # paths
    pos = df[["dx","dy","dz"]].to_numpy(float)
    rot = df[["droll","dpitch","dyaw"]].to_numpy(float) if set(["droll","dpitch","dyaw"]).issubset(df.columns) else np.zeros((len(df),3))
    grip = df["grip_bool"].to_numpy(int) if "grip_bool" in df.columns else np.zeros(len(df),int)

    l1_pos = np.abs(pos).sum(axis=1)
    l1_rot = np.abs(rot).sum(axis=1)

    E_act = float(l1_pos.sum() + rot_weight * l1_rot.sum() + grip_cost * int(np.sum(np.abs(np.diff(grip))>0)))
    L_path = float(np.linalg.norm(pos, ord=2, axis=1).sum())
    jerk = float(np.abs(np.diff(pos, axis=0)).sum())  # smoothness proxy
    return {"E_act": E_act, "L_path": L_path, "jerk": jerk}

def _flow_energy_from_frames(frame_paths):
    if not HAS_CV2 or len(frame_paths) < 2: return float("nan")
    flow_sum = 0.0
    prev = None
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=-1).mean()
            flow_sum += float(mag)
        prev = img
    return float(flow_sum)

def _maybe_frame_paths(df_row):
    # if your trace includes a "frame_path" column or similar, adapt here; else return []
    cols = [c for c in df_row.index if "frame" in c.lower() and str(df_row[c]).endswith((".png",".jpg",".jpeg"))]
    return [df_row[c] for c in cols]

def gather_dir(root, split="off", rot_weight=0.25, grip_cost=0.02, use_flow=False):
    """Aggregate per-episode proxies for a results root."""
    traces = sorted(glob.glob(os.path.join(root, split, "traces", "ep*.csv")))
    E_act, L_path, jerk, E_flow = [], [], [], []
    for t in traces:
        m = _energy_from_trace_csv(t, rot_weight, grip_cost)
        E_act.append(m["E_act"]); L_path.append(m["L_path"]); jerk.append(m["jerk"])
        if use_flow:
            # OPTIONAL: if trace rows include frame paths, compute per-ep flow energy (customize if needed)
            E_flow.append(float("nan"))  # placeholder; integrate with your saved frames if available
    out = {"E_act": np.array(E_act), "L_path": np.array(L_path), "jerk": np.array(jerk)}
    if use_flow: out["E_flow"] = np.array(E_flow)
    return out

def summarize_pair(root):
    off = gather_dir(root, "off")
    fep = gather_dir(root, "fepgate")
    res = {}
    for k in off.keys():
        X, Y = off[k], fep[k]
        off_ci = _series_ci(X)
        fep_ci = _series_ci(Y)
        d_ci = _series_ci(Y - X)
        obs, p = _perm_test(X, Y)
        res[k] = {
            "off_mean_CI": list(off_ci),
            "fep_mean_CI": list(fep_ci),
            "delta_mean_CI": list(d_ci),  # FEP - OFF (want negative for E_act, L_path, jerk)
            "perm_test": {"d": obs, "p": p},
            "N": int(X.size)
        }
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="results directories, e.g. results/reaching_big_*")
    ap.add_argument("--out", default="results/energy_stats.json")
    args = ap.parse_args()

    agg = {}
    for r in args.roots:
        if not os.path.exists(r): continue
        agg[r] = summarize_pair(r)

    # Concatenate across roots for a global summary
    keys = ["E_act","L_path","jerk"]
    global_off, global_fep = {k:[] for k in keys}, {k:[] for k in keys}
    for r in args.roots:
        # re-read raw arrays to concat
        o = gather_dir(r, "off"); f = gather_dir(r, "fepgate")
        for k in keys:
            global_off[k].append(o[k]); global_fep[k].append(f[k])
    glob_res = {}
    for k in keys:
        X = np.concatenate(global_off[k]) if global_off[k] else np.array([])
        Y = np.concatenate(global_fep[k]) if global_fep[k] else np.array([])
        off_ci = _series_ci(X) if X.size else (float("nan"),float("nan"),float("nan"))
        fep_ci = _series_ci(Y) if Y.size else (float("nan"),float("nan"),float("nan"))
        d_ci = _series_ci(Y - X) if X.size and Y.size else (float("nan"),float("nan"),float("nan"))
        obs, p = _perm_test(X, Y) if X.size and Y.size else (float("nan"),float("nan"))
        glob_res[k] = {
            "off_mean_CI": list(off_ci), "fep_mean_CI": list(fep_ci),
            "delta_mean_CI": list(d_ci), "perm_test": {"d": obs, "p": p},
            "N": int(X.size)
        }

    out = {"per_root": agg, "global": glob_res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f: json.dump(out, f, indent=2)
    print(f"[energy_proxies] wrote {args.out}")

if __name__ == "__main__":
    main()
