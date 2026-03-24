import argparse, json, os
import numpy as np
import pandas as pd

def load(path):
    df = pd.read_csv(path)
    return df.pivot(index="dx", columns="dy", values="energy").sort_index(), df

def nearest_idx(vals, x):
    arr = np.asarray(vals); i = int(np.argmin(np.abs(arr - x))); return i, arr[i]

def finite_diffs(Z, xs, ys, ix, iy, h=None):
    # central diffs at grid point (ix,iy)
    nx, ny = Z.shape
    ix = np.clip(ix, 1, nx-2); iy = np.clip(iy, 1, ny-2)
    hx = (xs[ix+1]-xs[ix-1])/2.0 if h is None else h
    hy = (ys[iy+1]-ys[iy-1])/2.0 if h is None else h
    dEx = (Z[ix+1,iy]-Z[ix-1,iy])/(2*hx)
    dEy = (Z[ix,iy+1]-Z[ix,iy-1])/(2*hy)
    d2Ex = (Z[ix+1,iy]-2*Z[ix,iy]+Z[ix-1,iy])/(hx*hx)
    d2Ey = (Z[ix,iy+1]-2*Z[ix,iy]+Z[ix,iy-1])/(hy*hy)
    d2Exy = (Z[ix+1,iy+1]-Z[ix+1,iy-1]-Z[ix-1,iy+1]+Z[ix-1,iy-1])/(4*hx*hy)
    H = np.array([[d2Ex, d2Exy],[d2Exy, d2Ey]], dtype=float)
    g = np.array([dEx, dEy], dtype=float)
    return g, H

def metrics(land_csv, gt=(0.0,-0.1), origin=(0.0,0.0)):
    Zp, df = load(land_csv)
    xs = Zp.index.values.astype(float)
    ys = Zp.columns.values.astype(float)
    Z = Zp.to_numpy()  # shape (nx, ny) with dx as rows, dy as cols

    # min location & min-shift to GT action
    imin = np.unravel_index(np.argmin(Z), Z.shape)
    dx_min, dy_min = xs[imin[0]], ys[imin[1]]
    min_shift = float(np.hypot(dx_min-gt[0], dy_min-gt[1]))

    # gradient alignment at origin vs GT direction
    ix0, dx0 = nearest_idx(xs, origin[0])
    iy0, dy0 = nearest_idx(ys, origin[1])
    g0, H0 = finite_diffs(Z, xs, ys, ix0, iy0)
    gt_dir = np.array([gt[0]-origin[0], gt[1]-origin[1]], dtype=float)
    if np.linalg.norm(gt_dir) > 0 and np.linalg.norm(g0) > 0:
        cos = float(np.dot(-g0, gt_dir)/(np.linalg.norm(g0)*np.linalg.norm(gt_dir)))
    else:
        cos = float("nan")

    # conditioning (Hessian eigenvalue ratio) at the basin minimum
    gmin, Hmin = finite_diffs(Z, xs, ys, imin[0], imin[1])
    w = np.linalg.eigvalsh(Hmin)
    cond = float((w.max()/max(w.min(), 1e-9)) if np.all(np.isfinite(w)) else float("nan"))

    return {
        "csv": land_csv,
        "min_dx": float(dx_min), "min_dy": float(dy_min),
        "min_shift_to_gt": min_shift,
        "grad_align_origin_cos": cos,
        "cond_number_at_min": cond
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_csv", default="results/landscape/landscape_off.csv")
    ap.add_argument("--fep_csv", default="results/landscape/landscape_fep.csv")
    ap.add_argument("--gt_dx", type=float, default=0.0)
    ap.add_argument("--gt_dy", type=float, default=-0.1)
    args = ap.parse_args()
    for tag, p in [("off", args.off_csv), ("fep", args.fep_csv)]:
        m = metrics(p, gt=(args.gt_dx, args.gt_dy))
        out = os.path.join(os.path.dirname(p), f"metrics_{tag}.json")
        with open(out, "w") as f: json.dump(m, f, indent=2)
        print(f"[landscape_metrics] wrote {out}")
