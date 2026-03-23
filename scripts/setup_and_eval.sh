#!/bin/bash
# ST-A² downstream eval: one-shot setup + run on Lambda A10/A100
# Usage: bash ~/vjepa2/scripts/setup_and_eval.sh
set -e

echo "============================================"
echo " ST-A² Downstream Eval Setup"
echo "============================================"

# --- 1. Clone repo ---
if [ ! -d ~/vjepa2 ]; then
    echo "[1/7] Cloning repo..."
    git clone -b feat/st-a2-area-attention https://github.com/tarassh/vjepa2.git ~/vjepa2
else
    echo "[1/7] Repo exists, pulling latest..."
    cd ~/vjepa2 && git pull
fi

# --- 2. Python venv + deps ---
echo "[2/7] Setting up Python environment..."
cd ~/vjepa2
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q decord pandas pyyaml timm scipy networkx

# --- 3. Download checkpoint ---
echo "[3/7] Downloading vitl.pt checkpoint..."
mkdir -p ~/checkpoints
if [ ! -f ~/checkpoints/vitl.pt ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P ~/checkpoints/
else
    echo "  Already downloaded."
fi

# --- 4. Download K400 val set ---
echo "[4/7] Downloading K400 validation set..."
mkdir -p ~/data/k400_targz/val ~/data/k400/val
if [ ! -f ~/data/k400_targz/val/k400_val_path.txt ]; then
    wget -q https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt -P ~/data/k400_targz/val/
fi
# Count expected vs existing tars
EXPECTED=$(wc -l < ~/data/k400_targz/val/k400_val_path.txt)
EXISTING=$(find ~/data/k400_targz/val -name "*.tar.gz" 2>/dev/null | wc -l)
if [ "$EXISTING" -lt "$EXPECTED" ]; then
    echo "  Downloading $EXPECTED tar files ($EXISTING already present)..."
    wget -c -q --show-progress -i ~/data/k400_targz/val/k400_val_path.txt -P ~/data/k400_targz/val/
else
    echo "  All $EXPECTED tar files already downloaded."
fi

# --- 5. Extract ---
echo "[5/7] Extracting K400 val videos..."
EXTRACTED=$(find ~/data/k400/val -name "*.mp4" 2>/dev/null | wc -l)
if [ "$EXTRACTED" -lt 1000 ]; then
    for f in ~/data/k400_targz/val/*.tar.gz; do
        tar xzf "$f" -C ~/data/k400/val/ 2>/dev/null || true
    done
    EXTRACTED=$(find ~/data/k400/val -name "*.mp4" 2>/dev/null | wc -l)
fi
echo "  $EXTRACTED videos extracted."

# --- 6. Generate CSV + update configs ---
echo "[6/7] Generating CSV manifest and configs..."
# Download annotations for flat layout (CVDF tars extract without class dirs)
if [ ! -f ~/data/k400/val.csv ]; then
    wget -q https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P ~/data/k400/
fi
python ~/vjepa2/scripts/prepare_k400_csv.py \
    --val_dir ~/data/k400/val \
    --annotations ~/data/k400/val.csv \
    --output ~/data/k400/k400_val_paths.csv

# Baseline config
cp ~/vjepa2/configs/eval/vitl/k400.yaml ~/vjepa2/configs/eval/vitl/k400-local.yaml
sed -i \
    -e "s|/your_vjepa2_checkpoints/vitl.pt|/home/ubuntu/checkpoints/vitl.pt|" \
    -e "s|/your_data_path/k400_train_paths.csv|/home/ubuntu/data/k400/k400_val_paths.csv|" \
    -e "s|/your_data_path/k400_val_paths.csv|/home/ubuntu/data/k400/k400_val_paths.csv|" \
    -e "s|/your_folder/evals/vitl/k400|/home/ubuntu/evals/k400-baseline|" \
    ~/vjepa2/configs/eval/vitl/k400-local.yaml

# ST-A² config
sed -i \
    -e "s|/your_vjepa2_checkpoints/vitl-area-attn.pt|/home/ubuntu/checkpoints/vitl.pt|" \
    -e "s|/your_data_path/k400_train_paths.csv|/home/ubuntu/data/k400/k400_val_paths.csv|" \
    -e "s|/your_data_path/k400_val_paths.csv|/home/ubuntu/data/k400/k400_val_paths.csv|" \
    -e "s|/your_folder/evals/vitl/k400-area-attn|/home/ubuntu/evals/k400-area-attn|" \
    ~/vjepa2/configs/eval/vitl/k400-area-attn.yaml

mkdir -p ~/evals/k400-baseline ~/evals/k400-area-attn

# --- 7. Run evals ---
echo "[7/7] Running evaluations..."
echo ""
echo "============================================"
echo " BASELINE (full attention)"
echo "============================================"
cd ~/vjepa2
python -m evals.main --fname configs/eval/vitl/k400-local.yaml --devices cuda:0

echo ""
echo "============================================"
echo " ST-A² (area attention)"
echo "============================================"
python -m evals.main --fname configs/eval/vitl/k400-area-attn.yaml --devices cuda:0

echo ""
echo "============================================"
echo " DONE — compare results in:"
echo "   ~/evals/k400-baseline/"
echo "   ~/evals/k400-area-attn/"
echo "============================================"
