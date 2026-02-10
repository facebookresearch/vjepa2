#!/bin/bash
# ST-A² fine-tune: one-shot setup + fine-tune + eval on Lambda A10/A100
#
# Downloads K400 val, fine-tunes vitl.pt with area attention for 1,000 steps,
# then runs frozen probe eval comparing baseline vs fine-tuned ST-A².
#
# Usage:
#   bash scripts/setup_and_finetune.sh          # run all steps
#   bash scripts/setup_and_finetune.sh --skip-download   # skip data download
#   bash scripts/setup_and_finetune.sh --eval-only       # skip fine-tune, run eval only
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
EVAL_FINETUNED=~/evals/k400-finetuned

echo "============================================"
echo " ST-A² Fine-tune + Eval Setup"
echo "============================================"

# ------------------------------------------------------------------
# 1. Clone / update repo
# ------------------------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/8] Cloning repo..."
    git clone -b feat/st-a2-area-attention https://github.com/tarassh/vjepa2.git "$REPO_DIR"
else
    echo "[1/8] Repo exists, pulling latest..."
    cd "$REPO_DIR" && git pull
fi

# ------------------------------------------------------------------
# 2. Python venv + deps
# ------------------------------------------------------------------
echo "[2/8] Setting up Python environment..."
cd "$REPO_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q decord pandas pyyaml timm scipy networkx einops psutil opencv-python-headless

# ------------------------------------------------------------------
# 3. Download vitl.pt checkpoint
# ------------------------------------------------------------------
echo "[3/8] Downloading vitl.pt checkpoint..."
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/vitl.pt" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P "$CKPT_DIR/"
else
    echo "  Already downloaded."
fi

# ------------------------------------------------------------------
# 4. Download + extract K400 val
# ------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[4/8] Downloading K400 validation set..."
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

    echo "[5/8] Extracting K400 val videos..."
    EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$EXTRACTED" -lt 1000 ]; then
        for f in "$K400_TAR_DIR"/*.tar.gz; do
            tar xzf "$f" -C "$K400_VID_DIR/" 2>/dev/null || true
        done
        EXTRACTED=$(find "$K400_VID_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    fi
    echo "  $EXTRACTED videos extracted."
else
    echo "[4/8] Skipping download (--skip-download)."
    echo "[5/8] Skipping extraction."
fi

# ------------------------------------------------------------------
# 5. Generate CSV manifest
# ------------------------------------------------------------------
echo "[6/8] Generating CSV manifest..."
if [ ! -f "$DATA_DIR/k400/val.csv" ]; then
    wget -q https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P "$DATA_DIR/k400/"
fi
python "$REPO_DIR/scripts/prepare_k400_csv.py" \
    --val_dir "$K400_VID_DIR" \
    --annotations "$DATA_DIR/k400/val.csv" \
    --output "$K400_CSV"

VIDEO_COUNT=$(wc -l < "$K400_CSV")
echo "  CSV has $VIDEO_COUNT videos."

# ------------------------------------------------------------------
# 6. Create fine-tune config with local paths
# ------------------------------------------------------------------
echo "[7/8] Preparing configs..."
mkdir -p "$FINETUNE_OUT" "$EVAL_BASELINE" "$EVAL_FINETUNED"

# --- Fine-tune config: use K400 val as SSL training data ---
FINETUNE_CFG="$REPO_DIR/configs/train/vitl16/finetune-local.yaml"
cp "$REPO_DIR/configs/train/vitl16/finetune-256px-16f-area-attn.yaml" "$FINETUNE_CFG"
sed -i \
    -e "s|/your_folder/finetune/area_attn/vitl.256px.16f|$FINETUNE_OUT|" \
    -e "s|/your_vjepa2_checkpoints/vitl.pt|$CKPT_DIR/vitl.pt|" \
    -e "/datasets:/,/datasets_weights:/{
        /datasets:/!{/datasets_weights:/!d}
    }" \
    -e "s|datasets:|datasets:\n  - $K400_CSV|" \
    -e "/datasets_weights:/,/batch_size:/{
        /datasets_weights:/!{/batch_size:/!d}
    }" \
    -e "s|datasets_weights:|datasets_weights:\n  - 1.0|" \
    -e "/dataset_fpcs:/,/tubelet_size:/{
        /dataset_fpcs:/!{/tubelet_size:/!d}
    }" \
    -e "s|dataset_fpcs:|dataset_fpcs:\n  - 16|" \
    "$FINETUNE_CFG"

# --- Baseline eval config ---
EVAL_BASELINE_CFG="$REPO_DIR/configs/eval/vitl/k400-baseline-local.yaml"
cp "$REPO_DIR/configs/eval/vitl/k400.yaml" "$EVAL_BASELINE_CFG"
sed -i \
    -e "s|/your_vjepa2_checkpoints/vitl.pt|$CKPT_DIR/vitl.pt|" \
    -e "s|/your_data_path/k400_train_paths.csv|$K400_CSV|" \
    -e "s|/your_data_path/k400_val_paths.csv|$K400_CSV|" \
    -e "s|/your_folder/evals/vitl/k400|$EVAL_BASELINE|" \
    -e "s|num_segments: 8|num_segments: 1|" \
    -e "s|num_views_per_segment: 3|num_views_per_segment: 1|" \
    -e "s|batch_size: 32|batch_size: 16|" \
    "$EVAL_BASELINE_CFG"
# Trim HP sweeps to 3 combos for speed
python -c "
import yaml, sys
cfg_path = '$EVAL_BASELINE_CFG'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
# Keep only 3 HP combos
if 'multihead_kwargs' in cfg.get('optimization', {}):
    cfg['optimization']['multihead_kwargs'] = cfg['optimization']['multihead_kwargs'][:3]
cfg['optimization']['num_epochs'] = 3
cfg['optimization']['resume_checkpoint'] = False
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# --- Finetuned ST-A² eval config ---
EVAL_FINETUNED_CFG="$REPO_DIR/configs/eval/vitl/k400-finetuned-local.yaml"
cp "$REPO_DIR/configs/eval/vitl/k400-area-attn.yaml" "$EVAL_FINETUNED_CFG"
# Point to fine-tuned checkpoint (latest saved during fine-tune)
sed -i \
    -e "s|/your_vjepa2_checkpoints/vitl-area-attn.pt|$FINETUNE_OUT/jepa-latest.pth.tar|" \
    -e "s|/your_data_path/k400_train_paths.csv|$K400_CSV|" \
    -e "s|/your_data_path/k400_val_paths.csv|$K400_CSV|" \
    -e "s|/your_folder/evals/vitl/k400-area-attn|$EVAL_FINETUNED|" \
    -e "s|num_segments: 8|num_segments: 1|" \
    -e "s|num_views_per_segment: 3|num_views_per_segment: 1|" \
    -e "s|batch_size: 32|batch_size: 16|" \
    "$EVAL_FINETUNED_CFG"
python -c "
import yaml, sys
cfg_path = '$EVAL_FINETUNED_CFG'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
if 'multihead_kwargs' in cfg.get('optimization', {}):
    cfg['optimization']['multihead_kwargs'] = cfg['optimization']['multihead_kwargs'][:3]
cfg['optimization']['num_epochs'] = 3
cfg['optimization']['resume_checkpoint'] = False
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

echo "  Configs ready:"
echo "    Fine-tune:  $FINETUNE_CFG"
echo "    Eval base:  $EVAL_BASELINE_CFG"
echo "    Eval tuned: $EVAL_FINETUNED_CFG"

# ------------------------------------------------------------------
# 7. Fine-tune (1,000 steps)
# ------------------------------------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "============================================"
    echo " FINE-TUNING: vitl.pt → ST-A² (1,000 steps)"
    echo "============================================"
    cd "$REPO_DIR"
    python -m app.main \
        --fname "$FINETUNE_CFG" \
        --devices cuda:0
    echo ""
    echo "Fine-tune complete. Checkpoint: $FINETUNE_OUT/jepa-latest.pth.tar"
else
    echo ""
    echo "[7/8] Skipping fine-tune (--eval-only)."
fi

# ------------------------------------------------------------------
# 8. Run downstream evals
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " EVAL 1/2: BASELINE (full attention)"
echo "============================================"
cd "$REPO_DIR"
python -m evals.main --fname "$EVAL_BASELINE_CFG" --devices cuda:0

echo ""
echo "============================================"
echo " EVAL 2/2: ST-A² FINE-TUNED"
echo "============================================"
python -m evals.main --fname "$EVAL_FINETUNED_CFG" --devices cuda:0

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
echo ""
echo "Results:"
echo "  Baseline:      $EVAL_BASELINE/"
echo "  ST-A² tuned:   $EVAL_FINETUNED/"
echo ""
echo "Compare with:"
echo "  cat $EVAL_BASELINE/k400-val-results.csv"
echo "  cat $EVAL_FINETUNED/k400-val-results.csv"
echo "============================================"
