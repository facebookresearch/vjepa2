# SIML × V‑JEPA 2 — One‑bit Surprise Gate for Latent Planning

**TL;DR.** We add a single binary from a SIML sidecar to the frozen V‑JEPA 2 planning loop:
- **B1 Grip‑bit** (task bit used by sampler/env), and  
- **B2 Surprise‑gate** (dynamics bit computed by a tiny active‑inference sidecar).  

With the same CEM/MPC budget, no retraining:
- Final reaching error drops **0.193 → 0.109 m**
- Monotonicity **0.12 → 0.43** (smoother progress)
- Latency **2.28 s → 1.13 s** per step
- Energy/episode **0.058 → 0.029 Wh**

> All numbers are 100 episodes on our workstation; V‑JEPA 2 weights are **frozen**. The bit only affects action selection.

---

## Quick start

### 0) Environment
```bash
conda create -n vjepa2-312 python=3.12 -y
conda activate vjepa2-312
pip install .    # from the V-JEPA 2 repo root (or -e . for dev)
pip install nvidia-ml-py3 pyyaml
```
### 1) Calibration of Tau
```
python -m siml.scripts.calibrate_tau \
  --config siml/configs/eval/siml_reaching.yaml \
  --samples 512 \
  --percentile 10 \
  --warm_steps 256 \
  --z_noise_sigma 1e-3
```
### 2) Perform Runs
```
# OFF (baseline)
python siml/scripts/power_wrap.py --out siml/results/reaching_off/power.json -- \
  python -m siml.evals.reach_eval \
    --config siml/configs/eval/siml_reaching.yaml \
    --out_dir siml/results/reaching_off \
    --only off

# FEP surprise gate
python siml/scripts/power_wrap.py --out siml/results/reaching_fep/power.json -- \
  python -m siml.evals.reach_eval \
    --config siml/configs/eval/siml_reaching.yaml \
    --out_dir siml/results/reaching_fep \
    --only fepgate \
    --tau <your_tau_here>
```
this writes:
```
results/
  reaching_off/
    power.json, power.csv
    off/summary.json, off/metrics.csv
  reaching_fep/
    power.json, power.csv
    fepgate/summary.json, fepgate/metrics.csv
```
Make the figures:
```
python -m siml.scripts.make_figs \
  --off_root  siml/results/reaching_off \
  --fep_root  siml/results/reaching_fep \
  --out_dir   siml/results/plots

```

---

## A couple of final clarifications

- **Are we using “real” V‑JEPA 2 data?**  
  Yes for encoding: you already exported latents via the V‑JEPA 2 encoder (16×16×D) and fed them to the planner. For forecasting you can use either the released V‑JEPA 2‑AC (when available) or the small surrogate forecaster used during your bring‑up. The surprise‑gate is agnostic to which forecaster you plug in.

- **Why does the gate reduce energy and latency?**  
  Because the elite set is re-fit on plausible candidates, the proposal distribution narrows around feasible moves faster. That means fewer wasted samples in later iterations → less compute per step → lower Wh/episode.

- **Monotonicity (again)** is simply the fraction of step‑to‑step decreases in position error. It’s a nice proxy for “does planning make consistent progress” and maps well to the “Fig. 8 behavior” in Meta’s paper.

