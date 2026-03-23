#!/bin/bash
# ST-A² full validation pipeline with resume support
#
# Runs the COMPLETE test plan on a single A100 (40GB):
#   Phase 1: Setup (clone, venv, deps, data)
#   Phase 2: Unit tests (9 tests)
#   Phase 3: Multi-resolution ablation sweep (4 resolutions × 2 configs)
#   Phase 4: Fine-tune vitl.pt → ST-A² (1,000 steps SSL annealing)
#   Phase 5: 3-way downstream eval (baseline, ST-A² no-FT, ST-A² finetuned)
#   Phase 6: Results summary
#
# Resume support: each phase writes a marker file on completion.
# If the script crashes, re-run it and it will skip completed phases.
#
# Usage:
#   bash scripts/setup_and_finetune.sh                 # full pipeline
#   bash scripts/setup_and_finetune.sh --skip-download # skip K400 download
#   bash scripts/setup_and_finetune.sh --reset         # clear markers, start fresh
#
# Recommended: run inside tmux
#   tmux new -s pipeline
#   bash ~/vjepa2/scripts/setup_and_finetune.sh
set -e

SKIP_DOWNLOAD=false
RESET=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --reset) RESET=true ;;
    esac
done

REPO_DIR=~/vjepa2
CKPT_DIR=~/checkpoints
DATA_DIR=~/data
K400_TAR_DIR=$DATA_DIR/k400_targz/val
K400_VID_DIR=$DATA_DIR/k400/val
K400_CSV=$DATA_DIR/k400/k400_val_paths.csv
FINETUNE_OUT=~/finetune/area_attn/vitl.256px.16f
EVAL_BASELINE=~/evals/k400-baseline
EVAL_NOFT=~/evals/k400-area-attn-noft
EVAL_FINETUNED=~/evals/k400-area-attn-finetuned
MARKER_DIR=~/pipeline_markers

# --- Resume support ---
if [ "$RESET" = true ]; then
    echo "Clearing all markers..."
    rm -rf "$MARKER_DIR"
fi
mkdir -p "$MARKER_DIR"

phase_done() { [ -f "$MARKER_DIR/$1.done" ]; }
mark_done() { date > "$MARKER_DIR/$1.done"; echo "  ✓ Phase '$1' complete."; }

echo "============================================"
echo " ST-A² Full Validation Pipeline"
echo " Unit Tests → Ablation → Fine-tune → Eval"
echo "============================================"
echo ""

# Check for completed phases
COMPLETED=0
for p in setup unit_tests ablation finetune eval_baseline eval_noft eval_finetuned; do
    if phase_done "$p"; then
        echo "  ✓ $p (already done)"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  ○ $p (pending)"
    fi
done
echo ""

# ==================================================================
# PHASE 1: SETUP
# ==================================================================
if phase_done "setup"; then
    echo "[Phase 1] Setup — SKIPPING (already done)"
    cd "$REPO_DIR"
    source .venv/bin/activate
else
    echo "[Phase 1] Setup..."

    # 1a. Clone / update repo
    if [ ! -d "$REPO_DIR" ]; then
        echo "  Cloning repo..."
        git clone -b feat/st-a2-area-attention https://github.com/tarassh/vjepa2.git "$REPO_DIR"
    else
        echo "  Repo exists, pulling latest..."
        cd "$REPO_DIR" && git pull
    fi

    # 1b. Python venv + deps
    echo "  Setting up Python environment..."
    cd "$REPO_DIR"
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -q decord pandas pyyaml timm scipy networkx einops psutil opencv-python-headless

    # 1c. Download vitl.pt
    echo "  Downloading vitl.pt checkpoint..."
    mkdir -p "$CKPT_DIR"
    if [ ! -f "$CKPT_DIR/vitl.pt" ]; then
        wget -q --show-progress https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P "$CKPT_DIR/"
    else
        echo "  Already downloaded."
    fi

    # 1d. Download + extract K400 val
    if [ "$SKIP_DOWNLOAD" = false ]; then
        echo "  Downloading K400 validation set..."
        mkdir -p "$K400_TAR_DIR" "$K400_VID_DIR"

        if [ ! -f "$K400_TAR_DIR/k400_val_path.txt" ]; then
            wget -q https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt -P "$K400_TAR_DIR/"
        fi

        EXPECTED=$(wc -l < "$K400_TAR_DIR/k400_val_path.txt")
        EXISTING=$(find "$K400_TAR_DIR" -name "*.tar.gz" 2>/dev/null | wc -l)
        if [ "$EXISTING" -lt "$EXPECTED" ]; then
            echo "  Downloading $EXPECTED tar files ($EXISTING already present)..."
            wget -c -q --show-progress -i "$K400_TAR_DIR/k400_val_path.txt" -P "$K400_TAR_DIR/"
        else
            echo "  All $EXPECTED tar files already downloaded."
        fi

        echo "  Extracting K400 val videos..."
        EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
        if [ "$EXTRACTED" -lt 1000 ]; then
            for f in "$K400_TAR_DIR"/*.tar.gz; do
                tar xzf "$f" -C "$K400_VID_DIR/" 2>/dev/null || true
            done
            EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
        fi
        echo "  $EXTRACTED videos extracted."
    else
        echo "  Skipping K400 download (--skip-download)."
    fi

    # 1e. Generate CSV manifest
    echo "  Generating CSV manifest..."
    if [ ! -f "$DATA_DIR/k400/val.csv" ]; then
        wget -q https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P "$DATA_DIR/k400/"
    fi
    python "$REPO_DIR/scripts/prepare_k400_csv.py" \
        --val_dir "$K400_VID_DIR" \
        --annotations "$DATA_DIR/k400/val.csv" \
        --output "$K400_CSV"
    VIDEO_COUNT=$(wc -l < "$K400_CSV")
    echo "  CSV has $VIDEO_COUNT videos."

    # 1f. Generate all configs via Python
    echo "  Preparing configs..."
    mkdir -p "$FINETUNE_OUT" "$EVAL_BASELINE" "$EVAL_NOFT" "$EVAL_FINETUNED"

    python3 << PYEOF
import yaml

CKPT = "$CKPT_DIR/vitl.pt"
K400 = "$K400_CSV"
FT_OUT = "$FINETUNE_OUT"
EVAL_BASE_DIR = "$EVAL_BASELINE"
EVAL_NOFT_DIR = "$EVAL_NOFT"
EVAL_FT_DIR = "$EVAL_FINETUNED"
REPO = "$REPO_DIR"

EVAL_BATCH_SIZE = 16
EVAL_NUM_EPOCHS = 10
EVAL_NUM_SEGMENTS = 1
EVAL_NUM_VIEWS = 1
EVAL_NUM_HP_SWEEPS = 5

def configure_eval(cfg, folder, checkpoint):
    cfg["folder"] = folder
    cfg["resume_checkpoint"] = False
    cfg["experiment"]["data"]["dataset_train"] = K400
    cfg["experiment"]["data"]["dataset_val"] = K400
    cfg["experiment"]["data"]["num_segments"] = EVAL_NUM_SEGMENTS
    cfg["experiment"]["data"]["num_views_per_segment"] = EVAL_NUM_VIEWS
    cfg["experiment"]["optimization"]["batch_size"] = EVAL_BATCH_SIZE
    cfg["experiment"]["optimization"]["num_epochs"] = EVAL_NUM_EPOCHS
    cfg["experiment"]["optimization"]["multihead_kwargs"] = \
        cfg["experiment"]["optimization"]["multihead_kwargs"][:EVAL_NUM_HP_SWEEPS]
    cfg["model_kwargs"]["checkpoint"] = checkpoint
    return cfg

# 1. Fine-tune config
with open(f"{REPO}/configs/train/vitl16/finetune-256px-16f-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["folder"] = FT_OUT
cfg["data"]["datasets"] = [K400]
cfg["data"]["datasets_weights"] = [1.0]
cfg["data"]["dataset_fpcs"] = [16]
cfg["optimization"]["anneal_ckpt"] = CKPT
with open(f"{REPO}/configs/train/vitl16/finetune-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# 2. Eval: baseline
with open(f"{REPO}/configs/eval/vitl/k400.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_BASE_DIR, CKPT)
with open(f"{REPO}/configs/eval/vitl/k400-baseline-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# 3. Eval: ST-A² no fine-tune
with open(f"{REPO}/configs/eval/vitl/k400-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_NOFT_DIR, CKPT)
with open(f"{REPO}/configs/eval/vitl/k400-area-attn-noft-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# 4. Eval: ST-A² fine-tuned
with open(f"{REPO}/configs/eval/vitl/k400-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_FT_DIR, f"{FT_OUT}/latest.pt")
with open(f"{REPO}/configs/eval/vitl/k400-finetuned-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("  All configs written.")
print(f"  Eval settings: batch={EVAL_BATCH_SIZE}, epochs={EVAL_NUM_EPOCHS}, "
      f"segments={EVAL_NUM_SEGMENTS}, views={EVAL_NUM_VIEWS}")
PYEOF

    mark_done "setup"
fi

# ==================================================================
# PHASE 2: UNIT TESTS
# ==================================================================
if phase_done "unit_tests"; then
    echo ""
    echo "[Phase 2] Unit Tests — SKIPPING (already done)"
else
    echo ""
    echo "============================================"
    echo " [Phase 2] Unit Tests"
    echo "============================================"
    cd "$REPO_DIR"
    python notebooks/test_area_attention.py
    mark_done "unit_tests"
fi

# ==================================================================
# PHASE 3: MULTI-RESOLUTION ABLATION SWEEP
# ==================================================================
if phase_done "ablation"; then
    echo ""
    echo "[Phase 3] Ablation Sweep — SKIPPING (already done)"
else
    echo ""
    echo "============================================"
    echo " [Phase 3] Multi-Resolution Ablation Sweep"
    echo "  4 resolutions × 2 configs = 8 runs"
    echo "  100 steps each + per-layer profiling"
    echo "============================================"
    cd "$REPO_DIR"
    python notebooks/ablation_h100_sweep.py
    mark_done "ablation"
fi

# ==================================================================
# PHASE 4: FINE-TUNE (1,000 steps SSL annealing)
# ==================================================================
if phase_done "finetune"; then
    echo ""
    echo "[Phase 4] Fine-tune — SKIPPING (already done)"
else
    echo ""
    echo "============================================"
    echo " [Phase 4] Fine-tune: vitl.pt → ST-A²"
    echo "  1,000 steps SSL annealing"
    echo "============================================"
    cd "$REPO_DIR"
    python -m app.main \
        --fname configs/train/vitl16/finetune-local.yaml \
        --devices cuda:0
    echo ""
    echo "  Checkpoint: $FINETUNE_OUT/latest.pt"
    mark_done "finetune"
fi

# ==================================================================
# PHASE 5: THREE-WAY DOWNSTREAM EVAL
# ==================================================================
echo ""
echo "============================================"
echo " [Phase 5] 3-Way Downstream Eval"
echo "  Identical settings for fair comparison"
echo "============================================"

cd "$REPO_DIR"

# 5a. Baseline
if phase_done "eval_baseline"; then
    echo ""
    echo "  Eval 1/3: Baseline — SKIPPING (already done)"
else
    echo ""
    echo "  ── Eval 1/3: BASELINE (vitl.pt, full attention) ──"
    python -m evals.main --fname configs/eval/vitl/k400-baseline-local.yaml --devices cuda:0
    mark_done "eval_baseline"
fi

# 5b. ST-A² no fine-tune
if phase_done "eval_noft"; then
    echo ""
    echo "  Eval 2/3: ST-A² no fine-tune — SKIPPING (already done)"
else
    echo ""
    echo "  ── Eval 2/3: ST-A² NO FINE-TUNE (vitl.pt) ──"
    python -m evals.main --fname configs/eval/vitl/k400-area-attn-noft-local.yaml --devices cuda:0
    mark_done "eval_noft"
fi

# 5c. ST-A² fine-tuned
if phase_done "eval_finetuned"; then
    echo ""
    echo "  Eval 3/3: ST-A² fine-tuned — SKIPPING (already done)"
else
    echo ""
    echo "  ── Eval 3/3: ST-A² FINE-TUNED (latest.pt) ──"
    python -m evals.main --fname configs/eval/vitl/k400-finetuned-local.yaml --devices cuda:0
    mark_done "eval_finetuned"
fi

# ==================================================================
# PHASE 6: RESULTS SUMMARY
# ==================================================================
echo ""
echo "============================================"
echo " [Phase 6] Results Summary"
echo "============================================"

# 6a. Ablation sweep results
if [ -f "$REPO_DIR/sweep_summary.csv" ]; then
    echo ""
    echo "── Ablation Sweep ──"
    python3 << PYABLATION
import csv
with open("$REPO_DIR/sweep_summary.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Group by resolution
resolutions = []
seen = set()
for r in rows:
    res = r["resolution"]
    if res not in seen:
        resolutions.append(res)
        seen.add(res)

print(f"  {'Resolution':<12} {'Visible':>7} {'BL Step(ms)':>12} {'ST Step(ms)':>12} {'Δ Time':>8} {'BL Loss':>9} {'ST Loss':>9} {'Δ Loss':>8}")
print(f"  {'-'*80}")
for res in resolutions:
    bl = next((r for r in rows if r["resolution"] == res and r["config"] == "baseline"), None)
    st = next((r for r in rows if r["resolution"] == res and r["config"] == "st_a2"), None)
    if bl and st:
        bl_time = float(bl["avg_step_ms"])
        st_time = float(st["avg_step_ms"])
        bl_loss = float(bl["final_loss"])
        st_loss = float(st["final_loss"])
        dt = (st_time - bl_time) / bl_time * 100
        dl = (st_loss - bl_loss) / bl_loss * 100
        print(f"  {res:<12} {bl['visible_tokens']:>7} {bl_time:>11.1f}ms {st_time:>11.1f}ms {dt:>+7.1f}% {bl_loss:>9.4f} {st_loss:>9.4f} {dl:>+7.1f}%")
print()
PYABLATION
fi

# 6b. Downstream eval results
echo "── Downstream Eval (K400 Frozen Probe) ──"

python3 << PYRESULTS
import glob

results = {}
for name, folder in [
    ("Baseline (vitl.pt)", "$EVAL_BASELINE"),
    ("ST-A² no FT (vitl.pt)", "$EVAL_NOFT"),
    ("ST-A² finetuned", "$EVAL_FINETUNED"),
]:
    logs = glob.glob(f"{folder}/**/log_r0.csv", recursive=True)
    if not logs:
        results[name] = "NO RESULTS"
        continue
    epochs = []
    with open(logs[0]) as f:
        for line in f:
            line = line.strip()
            if line.startswith("epoch,") or not line:
                epochs = []
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                epochs.append((int(parts[0]), float(parts[1]), float(parts[2])))
    results[name] = epochs

print(f"  {'Model':<25} {'Epoch 1':>10} {'Epoch 2':>10} {'Epoch 3':>10}")
print(f"  {'-'*58}")
for name in ["Baseline (vitl.pt)", "ST-A² no FT (vitl.pt)", "ST-A² finetuned"]:
    data = results.get(name)
    if isinstance(data, str):
        print(f"  {name:<25} {data}")
    elif not data:
        print(f"  {name:<25} NO DATA")
    else:
        vals = {e[0]: e[2] for e in data}
        e1 = f"{vals.get(1, 0):.2f}%" if vals.get(1) else "—"
        e2 = f"{vals.get(2, 0):.2f}%" if vals.get(2) else "—"
        e3 = f"{vals.get(3, 0):.2f}%" if vals.get(3) else "—"
        print(f"  {name:<25} {e1:>10} {e2:>10} {e3:>10}")
print()
PYRESULTS

echo "Raw logs:"
echo "  Ablation: $REPO_DIR/sweep_summary.csv"
echo "  Baseline: $EVAL_BASELINE/"
echo "  No FT:    $EVAL_NOFT/"
echo "  Finetuned: $EVAL_FINETUNED/"
echo ""
echo "============================================"
echo " ALL PHASES COMPLETE"
echo "============================================"
