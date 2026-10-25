#!/usr/bin/env bash
# Stage 1: tokenizer pretraining (soft RQ-VAE, Codeword Uniformity Regularization,
# Annealed Inference Alignment).
#
# Usage: bash run_stage1_tokenizer.sh [DATASET] [extra args for rqvae/main.py]
#   DATASET     Beauty (default) or Pet_Supplies
#   DATA_DIR    default ./dataset/$DATASET  (must contain item_emb_td.parquet)
#   CKPT_DIR    default ./checkpoints/$DATASET-tokenizer
# Output: $CKPT_DIR/best_loss_model.pth
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-Beauty}"; shift || true
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/dataset/$DATASET}" && pwd)"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/$DATASET-tokenizer}"
mkdir -p "$CKPT_DIR"; CKPT_DIR="$(cd "$CKPT_DIR" && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT/rqvae"
"$PYTHON" -u main.py \
  --data_path "$DATA_DIR/item_emb_td.parquet" \
  --ckpt_dir "$CKPT_DIR" \
  --seed 2025 \
  --epochs 3000 --batch_size 1024 --lr 1e-3 --learner AdamW \
  --lr_scheduler_type linear --warmup_epochs 50 --weight_decay 1e-4 --eval_step 50 \
  --num_emb_list 256 256 256 --e_dim 32 --layers 512 256 128 64 \
  --norm_type none --loss_type mse --kmeans_init True --kmeans_iters 100 \
  --vq_type soft --distance l2 \
  --temperature 0.01 --temperature_final 0.001 --temperature_schedule linear \
  --temperature_anneal_start_epoch 0 --temperature_anneal_end_epoch 3000 \
  --diversity_scale 1e-4 \
  "$@"
echo "Tokenizer checkpoint: $CKPT_DIR/best_loss_model.pth"
