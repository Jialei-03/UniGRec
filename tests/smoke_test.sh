#!/usr/bin/env bash
# End-to-end smoke test on a tiny synthetic dataset (CPU, a few minutes):
# stage 1 -> stage 2 + 3 with beam-search evaluation -> AVR fitting.
# Usage: bash tests/smoke_test.sh [WORK_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHON
echo "== work dir: $WORK"

"$PYTHON" "$ROOT/tests/make_toy_data.py" --out_dir "$WORK/Toy"

echo "== stage 1"
DATA_DIR="$WORK/Toy" CKPT_DIR="$WORK/Toy-tokenizer" bash "$ROOT/run_stage1_tokenizer.sh" Toy \
  --epochs 4 --eval_step 2 --warmup_epochs 1 --kmeans_iters 10 --num_workers 0 --device cpu \
  --temperature_anneal_end_epoch 4
test -f "$WORK/Toy-tokenizer/best_loss_model.pth"

echo "== stage 2 + 3"
DATA_DIR="$WORK/Toy" RQVAE_PATH="$WORK/Toy-tokenizer/best_loss_model.pth" SAVE_DIR="$WORK/Toy-UniGRec-s42" \
  bash "$ROOT/run_stage2_3_unigrec.sh" Toy 42 \
  --d_model 32 --num_layers 2 --num_decoder_layers 2 --d_ff 64 --num_heads 4 --d_kv 8 \
  --anchor_hidden_dim 32 --batch_size 64 --eval_batch_size 64 --num_epochs 2 --warm_epochs 1 \
  --selection_start_epoch 1 --finetune_epochs 2 --beam_size 10 --num_workers 0 --device cpu
test -f "$WORK/Toy-UniGRec-s42/best_finetune_model.pth"
cat "$WORK/Toy-UniGRec-s42/results.json"

echo "== AVR"
DATA_DIR="$WORK/Toy" SAVE_DIR="$WORK/Toy-UniGRec-s42" bash "$ROOT/run_avr.sh" Toy 42 --beam_size 10 --device cpu
test -f "$WORK/Toy-UniGRec-s42/avr/avr_results.json"
echo "== smoke test passed"
