#!/bin/bash
# Finetune-only mode: load from joint-training best_model.pth, then run finetuning only
# Usage: run from the UniGRec root directory, or from anywhere (the script will cd to UniGRec)

set -e

cd "$(dirname "$0")"

#############################################
# GPU
#############################################
GPU_ID=3
#############################################

DATASET="Beauty"

# Data
DATA_DIR="./dataset/${DATASET}"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VALID_DATA="${DATA_DIR}/valid.parquet"
TEST_DATA="${DATA_DIR}/test.parquet"
ITEM_EMB="${DATA_DIR}/item_emb_td.parquet"

# RQVAE checkpoint (used for building model/mapping; finetune_only will load the joint checkpoint afterwards to override weights)
RQVAE_MODEL="./rqvae/ckpt/${DATASET}/soft-32-diversity-td/best_loss_model.pth"

# SASRec teacher (the joint-training checkpoint contains sasrec_* params, so finetune-only must also build teacher-related modules to load_state_dict successfully)
SASREC_EMB_PATH="${DATA_DIR}/cf_emb_sasrec256.parquet"

# Joint-training best checkpoint (make sure best_model.pth exists under this directory)
JOINT_CKPT_DIR="/NAS/lijialei/UniGRec/ckpt/Beauty/Grid-Joint-2-g0.01-d0.1-T5LR-5e-3-RQLR-2e-7-warmconstant-LossTemp-1.0"
JOINT_CKPT_PATH="${JOINT_CKPT_DIR}/best_model.pth"


# T5 architecture (must match joint training, otherwise you may see attention weight size mismatch / unexpected key)
T5_ENCODER_LAYERS=6
T5_DECODER_LAYERS=6
T5_D_MODEL=128
T5_D_FF=512
T5_NUM_HEADS=4
T5_D_KV=64
T5_DROPOUT=0.1

# Fixed parameters (keep consistent with run_train_unigrec.sh)
ALPHA=0.0
BETA=1.0
DIVERSITY_WEIGHT=0.0
BATCH_SIZE=512
TAU=0.001
MAX_LEN=20
BEAM_SIZE=30
WEIGHT_DECAY=0.05

# Finetuning stage only
FINETUNE_LR=4e-4
FINETUNE_EPOCHS=100
FINETUNE_PATIENCE=10
FINETUNE_WARMUP_RATIO=0.05
FINETUNE_SCHEDULER_TYPE="constant"
FINETUNE_EARLY_STOP_METRIC="ndcg@10"

# SwanLab
SWAN_NAME="${DATASET}-FinetuneOnly-from-$(basename "${JOINT_CKPT_DIR}")-lr_${FINETUNE_LR}"

# Output directory
OUTPUT_DIR="./ckpt/${DATASET}/FinetuneOnly-from-$(basename "${JOINT_CKPT_DIR}")"
mkdir -p "${OUTPUT_DIR}"

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ Finetune-Only"
echo "│ DATASET: ${DATASET}"
echo "│ JOINT_CKPT_PATH: ${JOINT_CKPT_PATH}"
echo "│ OUTPUT_DIR: ${OUTPUT_DIR}"
echo "└──────────────────────────────────────────────────────────────┘"

CUDA_VISIBLE_DEVICES=${GPU_ID} python model/train.py \
  --finetune_only \
  --joint_checkpoint_path "${JOINT_CKPT_PATH}" \
  --train_data "${TRAIN_DATA}" \
  --valid_data "${VALID_DATA}" \
  --test_data "${TEST_DATA}" \
  --item_emb_path "${ITEM_EMB}" \
  --rqvae_model_path "${RQVAE_MODEL}" \
  --use_sasrec_teacher \
  --sasrec_emb_path "${SASREC_EMB_PATH}" \
  --save_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma 0.0 \
  --delta 0.0 \
  --diversity_weight "${DIVERSITY_WEIGHT}" \
  --tau "${TAU}" \
  --max_len "${MAX_LEN}" \
  --beam_size "${BEAM_SIZE}" \
  --num_layers "${T5_ENCODER_LAYERS}" \
  --num_decoder_layers "${T5_DECODER_LAYERS}" \
  --d_model "${T5_D_MODEL}" \
  --d_ff "${T5_D_FF}" \
  --num_heads "${T5_NUM_HEADS}" \
  --d_kv "${T5_D_KV}" \
  --dropout_rate "${T5_DROPOUT}" \
  --train_num_workers 8 \
  --eval_num_workers 4 \
  --pin_memory \
  --prefetch_factor 4 \
  --prefetch_to_gpu \
  --exp_name "${SWAN_NAME}" \
  --use_code_offset \
  --use_swanlab \
  --enable_finetune \
  --finetune_lr "${FINETUNE_LR}" \
  --finetune_epochs "${FINETUNE_EPOCHS}" \
  --finetune_patience "${FINETUNE_PATIENCE}" \
  --finetune_early_stop_metric "${FINETUNE_EARLY_STOP_METRIC}" \
  --finetune_warmup_ratio "${FINETUNE_WARMUP_RATIO}" \
  --finetune_scheduler_type "${FINETUNE_SCHEDULER_TYPE}"

echo "Finetune-only finished. Model saved to ${OUTPUT_DIR}"
