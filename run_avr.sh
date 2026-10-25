#!/usr/bin/env bash
# Anchor-Verified Re-ranking: fit the re-ranking weights on validation, report test.
#
# Usage: bash run_avr.sh [DATASET] [SEED] [extra args for avr/run_avr.py]
#   DATA_DIR    default ./dataset/$DATASET
#   SAVE_DIR    default ./checkpoints/$DATASET-UniGRec-s$SEED (output of run_stage2_3_unigrec.sh)
# Output: $SAVE_DIR/avr/avr_results.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-Beauty}"; shift || true
SEED="${1:-42}"; shift || true
DATA_DIR="${DATA_DIR:-$ROOT/dataset/$DATASET}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/$DATASET-UniGRec-s$SEED}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -u "$ROOT/avr/run_avr.py" \
  --checkpoint "$SAVE_DIR/best_finetune_model.pth" \
  --data_dir "$DATA_DIR" \
  --item_emb_path "$DATA_DIR/item_emb_td.parquet" \
  --output_dir "$SAVE_DIR/avr" \
  --beam_size 30 \
  "$@"
