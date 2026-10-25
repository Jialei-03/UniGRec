#!/usr/bin/env bash
# Stage 2 (end-to-end joint training) + stage 3 (fine-tuning) + test evaluation.
#
# Usage: bash run_stage2_3_unigrec.sh [DATASET] [SEED] [extra args for model/train.py]
#   DATASET     Beauty (default) or Pet_Supplies
#   SEED        training seed (default 42; the paper reports 42, 2026 and 2027)
#   DATA_DIR    default ./dataset/$DATASET
#   RQVAE_PATH  default ./checkpoints/$DATASET-tokenizer/best_loss_model.pth
#   SAVE_DIR    default ./checkpoints/$DATASET-UniGRec-s$SEED
# Output: $SAVE_DIR/best_model.pth (joint), best_finetune_model.pth, results.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-Beauty}"; shift || true
SEED="${1:-42}"; shift || true
DATA_DIR="${DATA_DIR:-$ROOT/dataset/$DATASET}"
RQVAE_PATH="${RQVAE_PATH:-$ROOT/checkpoints/$DATASET-tokenizer/best_loss_model.pth}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/$DATASET-UniGRec-s$SEED}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -u "$ROOT/model/train.py" \
  --data_dir "$DATA_DIR" \
  --item_emb_path "$DATA_DIR/item_emb_td.parquet" \
  --rqvae_path "$RQVAE_PATH" \
  --save_dir "$SAVE_DIR" \
  --seed "$SEED" \
  --max_len 20 --max_collision 100 \
  --d_model 128 --num_layers 6 --num_decoder_layers 6 --d_ff 512 --num_heads 4 --d_kv 64 --dropout_rate 0.1 \
  --batch_size 512 --num_epochs 200 --patience 40 --selection_start_epoch 11 \
  --t5_lr 5e-3 --rqvae_lr 2e-7 --anchor_lr 1e-3 --weight_decay 0.05 --warmup_ratio 0.1 \
  --alpha 0.5 --warm_epochs 20 \
  --lambda_seq 0.05 --lambda_hist 1.5 --lambda_tgt 0.05 \
  --anchor_temperature 0.07 --anchor_momentum 0.99 --anchor_hidden_dim 128 \
  --finetune_lr 2e-4 --finetune_epochs 100 --finetune_patience 10 \
  --beam_size 30 --topk 5 10 20 \
  "$@"
