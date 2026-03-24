import sys, pandas as pd, numpy as np, json
p = sys.argv[1]
df = pd.read_csv(p)
t = (df['timestamp_ms'].to_numpy() - df['timestamp_ms'].iloc[0]) / 1000.0  # seconds
P = df['power_W'].to_numpy()
E_Wh = float(np.trapz(P, t) / 3600.0)
print(json.dumps({
    "file": p,
    "gpu_energy_Wh": E_Wh,
    "avg_power_W": float(P.mean()),
    "duration_s": float(t[-1])}))
