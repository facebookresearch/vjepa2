#!/usr/bin/env python3
import os, json, csv, numpy as np, matplotlib.pyplot as plt

OFF_SUM = "results/reaching_off/off/summary.json"
OFF_MET = "results/reaching_off/off/metrics.csv"
OFF_PWR = "results/reaching_off/power.json"

FEP_SUM = "results/reaching_fep/fepgate/summary.json"
FEP_MET = "results/reaching_fep/fepgate/metrics.csv"
FEP_PWR = "results/reaching_fep/power.json"

def j(p):
    with open(p) as f: return json.load(f)

def m(summary, key):
    d = summary.get(key, {}); 
    return float(d.get("mean", np.nan)), float(d.get("std", np.nan))

def wh_per_ep(pwr, N):
    return float(pwr["gpu_energy_Wh"]) / float(N) if (pwr and N) else np.nan

def load_errors(metrics_csv_path):
    vals = []
    with open(metrics_csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                vals.append(float(row["final_err_m"]))
            except Exception:
                pass
    return np.array(vals, dtype=float)

# ---- load summaries & power
S_off, S_fep = j(OFF_SUM), j(FEP_SUM)
P_off, P_fep = j(OFF_PWR), j(FEP_PWR)
N_off, N_fep = int(S_off.get("N", 0)), int(S_fep.get("N", 0))

off_err_mu, off_err_sd = m(S_off, "final_err_m")
fep_err_mu, fep_err_sd = m(S_fep, "final_err_m")
off_lat_mu, off_lat_sd = m(S_off, "avg_latency_ms")
fep_lat_mu, fep_lat_sd = m(S_fep, "avg_latency_ms")
off_mon_mu, off_mon_sd = m(S_off, "monotonicity")
fep_mon_mu, fep_mon_sd = m(S_fep, "monotonicity")

e_off = wh_per_ep(P_off, N_off)
e_fep = wh_per_ep(P_fep, N_fep)

# ---- per-episode errors for violin
errs_off = load_errors(OFF_MET)
errs_fep = load_errors(FEP_MET)

os.makedirs("results/figs", exist_ok=True)

def bar(ax, labels, means, errs=None, ylabel="", title=""):
    x = np.arange(len(labels))
    ax.bar(x, means)
    if errs is not None:
        ax.errorbar(x, means, yerr=errs, fmt="none", ecolor="k", capsize=5, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Energy / episode
fig, ax = plt.subplots(figsize=(6,4.5))
bar(ax, ["OFF","FEP"], [e_off, e_fep], None, "Wh / episode", "Energy per episode")
vals = np.array([e_off, e_fep], dtype=float); vals = vals[np.isfinite(vals)]
if vals.size: ax.set_ylim(0, max(0.05, float(vals.max())*1.25))
fig.tight_layout(); fig.savefig("results/figs/energy_per_ep.png", dpi=160)

# Final error (bars)
fig, ax = plt.subplots(figsize=(6,4.5))
bar(ax, ["OFF","FEP"], [off_err_mu, fep_err_mu], [off_err_sd, fep_err_sd], "meters", "Final Error (m)")
fig.tight_layout(); fig.savefig("results/figs/final_err.png", dpi=160)

# Final error distribution (violin)
fig, ax = plt.subplots(figsize=(6,4.5))
parts = ax.violinplot([errs_off, errs_fep], positions=[0,1], showmeans=True, showextrema=True, showmedians=False)
ax.set_xticks([0,1]); ax.set_xticklabels(["OFF","FEP"])
ax.set_ylabel("Final error (m)"); ax.set_title("Final error distribution")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# add mean markers (mean is also shown by violin's showmeans; this emphasises it)
ax.scatter([0,1], [np.mean(errs_off), np.mean(errs_fep)], zorder=3, s=30, color="k")
fig.tight_layout(); fig.savefig("results/figs/final_err_violin.png", dpi=160)

# Latency
fig, ax = plt.subplots(figsize=(6,4.5))
bar(ax, ["OFF","FEP"], [off_lat_mu, fep_lat_mu], [off_lat_sd, fep_lat_sd], "ms", "Avg Latency (ms)")
fig.tight_layout(); fig.savefig("results/figs/latency.png", dpi=160)

# Monotonicity
fig, ax = plt.subplots(figsize=(6,4.5))
bar(ax, ["OFF","FEP"], [off_mon_mu, fep_mon_mu], [off_mon_sd, fep_mon_sd], "fraction ↓ steps", "Monotonicity")
fig.tight_layout(); fig.savefig("results/figs/monotonicity.png", dpi=160)

print("wrote: results/figs/{energy_per_ep,final_err,final_err_violin,latency,monotonicity}.png")
