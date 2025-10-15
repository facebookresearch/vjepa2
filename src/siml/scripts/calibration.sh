#!/usr/bin/env bash
set -euo pipefail

# Where your base YAML lives
BASE_CFG="../configs/eval/siml_reaching.yaml"
EPISODES="${EPISODES:-300}"
STEPS="${STEPS:-6}"

# Create a patched copy of the YAML with {tau, lambda_penalty, gate_hard} overrides.
mk_cfg() {
  local in="$1" out="$2" tau="$3" lam="$4" hard="$5"
  python - "$in" "$out" "$tau" "$lam" "$hard" <<'PY'
import sys, yaml
src, dst, tau, lam, hard = sys.argv[1:]
with open(src, "r") as f:
    cfg = yaml.safe_load(f) or {}
if tau.lower() != "none":
    cfg["tau"] = float(tau)
if lam.lower() != "none":
    cfg["lambda_penalty"] = float(lam)
cfg["gate_hard"] = (hard.lower() == "true")
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[mk_cfg] wrote {dst}")
PY
}

# Run the eval for a given config and output dir
run_eval() {
  local cfg="$1" out="$2"
  python -m siml.evals.reach_eval \
    --config "$cfg" \
    --out_dir "$out" \
    --episodes "$EPISODES" \
    --steps "$STEPS"
}

# Summarize acceptance, success, and key metrics for a results root
summarize() {
  local root="$1"
  python - "$root" <<'PY'
import os, glob, json, pandas as pd, sys
root = sys.argv[1]
sum_path = os.path.join(root, "fepgate", "summary.json")
if os.path.exists(sum_path):
    with open(sum_path) as f: summary = json.load(f)
else:
    summary = {"warning": f"missing {sum_path}"}
# gate acceptance for chosen actions
acc = []
for p in glob.glob(os.path.join(root, "fepgate", "traces", "ep*.csv")):
    df = pd.read_csv(p)
    if "ok_bit" in df:
        acc.append(df["ok_bit"].mean())
accept = float(pd.Series(acc).mean()) if acc else float("nan")
# success rate (<= tol within steps)
met = os.path.join(root, "fepgate", "metrics.csv")
succ = float("nan")
if os.path.exists(met):
    dfm = pd.read_csv(met)
    succ = float((dfm["steps_to_tol"] <= dfm["steps"]).mean() * 100.0)
print(json.dumps({
    "root": root,
    "gate_accept_mean": accept,
    "success_rate_pct": succ,
    "summary_fepgate": summary
}, indent=2))
PY
}

echo "== Baseline (use values from $BASE_CFG) =="
BASE_OUT="../results/reaching_baseline"
run_eval "$BASE_CFG" "$BASE_OUT"
summarize "$BASE_OUT"

echo "== Lambda sweep (soft penalty) =="
for L in 0.1 0.5 1.0; do
  OUT="../results/reaching_lambda_${L}"
  TMP="../tmp/reach_cfg_lambda_${L}.yaml"
  mkdir -p "$(dirname "$TMP")"
  # keep tau from base cfg; set lambda_penalty
  mk_cfg "$BASE_CFG" "$TMP" "none" "$L" "false"
  run_eval "$TMP" "$OUT"
  summarize "$OUT"
done

echo "== Hard gate =="
OUT="../results/reaching_hardgate"
TMP="../tmp/reach_cfg_hard.yaml"
mk_cfg "$BASE_CFG" "$TMP" "$(python - <<'PY'
import yaml
with open("../configs/eval/siml_reaching.yaml") as f:
    cfg=yaml.safe_load(f) or {}
print(cfg.get("tau", 0.073))
PY
)" "0.5" "true"
run_eval "$TMP" "$OUT"
summarize "$OUT"

echo "== Tau sweep =="
for T in 0.060 0.073 0.085; do
  OUT="../results/reaching_tau_${T}"
  TMP="../tmp/reach_cfg_tau_${T}.yaml"
  mk_cfg "$BASE_CFG" "$TMP" "$T" "0.5" "false"
  run_eval "$TMP" "$OUT"
  summarize "$OUT"
done

echo "All done."
