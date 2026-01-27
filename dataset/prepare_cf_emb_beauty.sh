#!/bin/bash
# Export SASRec CF embedding: cf_emb_sasrec{hidden_dim}.parquet
# To switch dataset: change DATASET (must match RecBole dataset name and the subdirectory under data/)

set -e
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

DATASET="Beauty"

SASREC_CKPT="/NAS/xxx/RecBole-master/saved/SASRec-Dec-21-2025_18-22-00.pth"
RECBOLE="/NAS/xxx/RecBole-master"

echo "[${DATASET}] Export SASRec CF embedding -> ${DATASET}/cf_emb_sasrec256.parquet"

python prepare_cf_emb.py \
  --sasrec_checkpoint "$SASREC_CKPT" \
  --recbole_path "$RECBOLE" \
  --recbole_dataset "$DATASET" \
  --unigrec_item_emb_parquet "${DATASET}/item_emb_td.parquet" \
  --hidden_dim 256 \
  --output_parquet "${DATASET}/cf_emb_sasrec256.parquet"

echo "[${DATASET}] Done: cf_emb_sasrec256.parquet"
