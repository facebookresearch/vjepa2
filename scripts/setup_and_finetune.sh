#!/bin/bash
# ST-A² full pipeline: setup → fine-tune → 3-way eval on Lambda A100
#
# Runs all three evaluations with identical settings for a fair comparison:
#   1. Baseline (vitl.pt, full attention)
#   2. ST-A² no fine-tune (vitl.pt loaded into area attention layers)
#   3. ST-A² fine-tuned (1,000-step SSL annealing with area attention)
#
# Usage:
#   bash scripts/setup_and_finetune.sh                 # full pipeline
#   bash scripts/setup_and_finetune.sh --skip-download # skip K400 download
#   bash scripts/setup_and_finetune.sh --eval-only     # skip fine-tune
#
# Recommended: run inside tmux
#   tmux new -s finetune
#   bash ~/vjepa2/scripts/setup_and_finetune.sh
set -e

SKIP_DOWNLOAD=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --eval-only) EVAL_ONLY=true; SKIP_DOWNLOAD=true ;;
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

echo "============================================"
echo " ST-A² Full Pipeline"
echo " Setup → Fine-tune → 3-way Eval"
echo "============================================"

# ==================================================================
# PHASE 1: SETUP
# ==================================================================

# --- 1. Clone / update repo ---
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/9] Cloning repo..."
    git clone -b feat/st-a2-area-attention https://github.com/tarassh/vjepa2.git "$REPO_DIR"
else
    echo "[1/9] Repo exists, pulling latest..."
    cd "$REPO_DIR" && git pull
fi

# --- 2. Python venv + deps ---
echo "[2/9] Setting up Python environment..."
cd "$REPO_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q decord pandas pyyaml timm scipy networkx einops psutil opencv-python-headless

# --- 3. Download vitl.pt ---
echo "[3/9] Downloading vitl.pt checkpoint..."
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/vitl.pt" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P "$CKPT_DIR/"
else
    echo "  Already downloaded."
fi

# --- 4-5. Download + extract K400 val ---
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[4/9] Downloading K400 validation set..."
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

    echo "[5/9] Extracting K400 val videos..."
    EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$EXTRACTED" -lt 1000 ]; then
        for f in "$K400_TAR_DIR"/*.tar.gz; do
            tar xzf "$f" -C "$K400_VID_DIR/" 2>/dev/null || true
        done
        EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    fi
    echo "  $EXTRACTED videos extracted."
else
    echo "[4/9] Skipping download (--skip-download)."
    echo "[5/9] Skipping extraction."
fi

# --- 6. Generate CSV manifest ---
echo "[6/9] Generating CSV manifest..."
if [ ! -f "$DATA_DIR/k400/val.csv" ]; then
    wget -q https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P "$DATA_DIR/k400/"
fi
python "$REPO_DIR/scripts/prepare_k400_csv.py" \
    --val_dir "$K400_VID_DIR" \
    --annotations "$DATA_DIR/k400/val.csv" \
    --output "$K400_CSV"

VIDEO_COUNT=$(wc -l < "$K400_CSV")
echo "  CSV has $VIDEO_COUNT videos."

# --- 7. Generate all configs via Python ---
echo "[7/9] Preparing configs..."
mkdir -p "$FINETUNE_OUT" "$EVAL_BASELINE" "$EVAL_NOFT" "$EVAL_FINETUNED"

FINETUNE_CFG="$REPO_DIR/configs/train/vitl16/finetune-local.yaml"
EVAL_BASELINE_CFG="$REPO_DIR/configs/eval/vitl/k400-baseline-local.yaml"
EVAL_NOFT_CFG="$REPO_DIR/configs/eval/vitl/k400-area-attn-noft-local.yaml"
EVAL_FINETUNED_CFG="$REPO_DIR/configs/eval/vitl/k400-finetuned-local.yaml"

python3 << PYEOF
import yaml

CKPT = "$CKPT_DIR/vitl.pt"
K400 = "$K400_CSV"
FT_OUT = "$FINETUNE_OUT"
EVAL_BASE_DIR = "$EVAL_BASELINE"
EVAL_NOFT_DIR = "$EVAL_NOFT"
EVAL_FT_DIR = "$EVAL_FINETUNED"
REPO = "$REPO_DIR"

# Shared eval settings for fair comparison
EVAL_BATCH_SIZE = 16
EVAL_NUM_EPOCHS = 3
EVAL_NUM_SEGMENTS = 1
EVAL_NUM_VIEWS = 1

def configure_eval(cfg, folder, checkpoint):
    """Apply identical eval settings to any eval config."""
    cfg["folder"] = folder
    cfg["resume_checkpoint"] = False
    cfg["experiment"]["data"]["dataset_train"] = K400
    cfg["experiment"]["data"]["dataset_val"] = K400
    cfg["experiment"]["data"]["num_segments"] = EVAL_NUM_SEGMENTS
    cfg["experiment"]["data"]["num_views_per_segment"] = EVAL_NUM_VIEWS
    cfg["experiment"]["optimization"]["batch_size"] = EVAL_BATCH_SIZE
    cfg["experiment"]["optimization"]["num_epochs"] = EVAL_NUM_EPOCHS
    cfg["experiment"]["optimization"]["multihead_kwargs"] = \
        cfg["experiment"]["optimization"]["multihead_kwargs"][:3]
    cfg["model_kwargs"]["checkpoint"] = checkpoint
    return cfg

# --- 1. Fine-tune config ---
with open(f"{REPO}/configs/train/vitl16/finetune-256px-16f-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["folder"] = FT_OUT
cfg["data"]["datasets"] = [K400]
cfg["data"]["datasets_weights"] = [1.0]
cfg["data"]["dataset_fpcs"] = [16]
cfg["optimization"]["anneal_ckpt"] = CKPT
with open(f"{REPO}/configs/train/vitl16/finetune-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# --- 2. Eval: baseline (vitl.pt, full attention) ---
with open(f"{REPO}/configs/eval/vitl/k400.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_BASE_DIR, CKPT)
with open(f"{REPO}/configs/eval/vitl/k400-baseline-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# --- 3. Eval: ST-A² no fine-tune (vitl.pt into area attention) ---
with open(f"{REPO}/configs/eval/vitl/k400-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_NOFT_DIR, CKPT)
with open(f"{REPO}/configs/eval/vitl/k400-area-attn-noft-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# --- 4. Eval: ST-A² fine-tuned (latest.pt) ---
with open(f"{REPO}/configs/eval/vitl/k400-area-attn.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = configure_eval(cfg, EVAL_FT_DIR, f"{FT_OUT}/latest.pt")
with open(f"{REPO}/configs/eval/vitl/k400-finetuned-local.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("  All 4 configs written successfully.")
print(f"  Eval settings: batch={EVAL_BATCH_SIZE}, epochs={EVAL_NUM_EPOCHS}, "
      f"segments={EVAL_NUM_SEGMENTS}, views={EVAL_NUM_VIEWS}")
PYEOF

echo "  Configs:"
echo "    Fine-tune:       $FINETUNE_CFG"
echo "    Eval baseline:   $EVAL_BASELINE_CFG"
echo "    Eval ST-A² noFT: $EVAL_NOFT_CFG"
echo "    Eval ST-A² FT:   $EVAL_FINETUNED_CFG"

# ==================================================================
# PHASE 2: FINE-TUNE
# ==================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "============================================"
    echo " [8/9] FINE-TUNING: vitl.pt → ST-A²"
    echo "        (1,000 steps SSL annealing)"
    echo "============================================"
    cd "$REPO_DIR"
    python -m app.main \
        --fname "$FINETUNE_CFG" \
        --devices cuda:0
    echo ""
    echo "Fine-tune complete. Checkpoint: $FINETUNE_OUT/latest.pt"
else
    echo ""
    echo "[8/9] Skipping fine-tune (--eval-only)."
fi

# ==================================================================
# PHASE 3: THREE-WAY EVAL
# ==================================================================
echo ""
echo "[9/9] Running 3-way downstream evaluation..."

echo ""
echo "============================================"
echo " EVAL 1/3: BASELINE (vitl.pt, full attn)"
echo "============================================"
cd "$REPO_DIR"
python -m evals.main --fname "$EVAL_BASELINE_CFG" --devices cuda:0

echo ""
echo "============================================"
echo " EVAL 2/3: ST-A² NO FINE-TUNE (vitl.pt)"
echo "============================================"
python -m evals.main --fname "$EVAL_NOFT_CFG" --devices cuda:0

echo ""
echo "============================================"
echo " EVAL 3/3: ST-A² FINE-TUNED (latest.pt)"
echo "============================================"
python -m evals.main --fname "$EVAL_FINETUNED_CFG" --devices cuda:0

# ==================================================================
# PHASE 4: RESULTS SUMMARY
# ==================================================================
echo ""
echo "============================================"
echo " ALL DONE — Printing results"
echo "============================================"

python3 << PYRESULTS
import csv, glob, os

results = {}
for name, folder in [
    ("Baseline", "$EVAL_BASELINE"),
    ("ST-A² (no FT)", "$EVAL_NOFT"),
    ("ST-A² (finetuned)", "$EVAL_FINETUNED"),
]:
    logs = glob.glob(f"{folder}/**/log_r0.csv", recursive=True)
    if not logs:
        results[name] = "NO RESULTS"
        continue
    # Read the last run's results (file may have multiple header rows)
    epochs = []
    with open(logs[0]) as f:
        for line in f:
            line = line.strip()
            if line.startswith("epoch,") or not line:
                epochs = []  # reset on new header = new HP sweep
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                epochs.append((int(parts[0]), float(parts[1]), float(parts[2])))
    results[name] = epochs

print()
print(f"{'Model':<22} {'Epoch 1':>10} {'Epoch 2':>10} {'Epoch 3':>10}")
print("-" * 55)
for name in ["Baseline", "ST-A² (no FT)", "ST-A² (finetuned)"]:
    data = results.get(name)
    if isinstance(data, str):
        print(f"{name:<22} {data}")
    else:
        vals = {e[0]: e[2] for e in data}  # epoch -> val_acc
        e1 = f"{vals.get(1, 0):.2f}%" if vals.get(1) else "—"
        e2 = f"{vals.get(2, 0):.2f}%" if vals.get(2) else "—"
        e3 = f"{vals.get(3, 0):.2f}%" if vals.get(3) else "—"
        print(f"{name:<22} {e1:>10} {e2:>10} {e3:>10}")
print()
PYRESULTS

echo "Raw logs:"
echo "  $EVAL_BASELINE/"
echo "  $EVAL_NOFT/"
echo "  $EVAL_FINETUNED/"
echo "============================================"
