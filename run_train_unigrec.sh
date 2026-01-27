#!/bin/bash
# Stage 2: end-to-end joint training (with finetuning)
# Usage: run from the UniGRec root directory, or from anywhere (the script will cd to UniGRec)
# Prerequisite: run run_train_rqvae.sh first, and set RQVAE_MODEL to its best_loss_model.pth

set -e

cd "$(dirname "$0")"

#############################################
# Experiment range and GPU
#############################################
START_IDX=1
END_IDX=1
IDX_LIST=""
GPU_ID=0
#############################################

DATASET="Beauty"

# Data and RQVAE checkpoint (stage 1 writes to rqvae/ckpt/${DATASET}/latest/best_loss_model.pth)
DATA_DIR="./dataset/${DATASET}"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VALID_DATA="${DATA_DIR}/valid.parquet"
TEST_DATA="${DATA_DIR}/test.parquet"
ITEM_EMB="${DATA_DIR}/item_emb_td.parquet"
RQVAE_MODEL="./rqvae/ckpt/${DATASET}/soft-32-diversity-td/best_loss_model.pth"
SASREC_EMB_PATH="${DATA_DIR}/cf_emb_sasrec256.parquet"

# Fixed parameters
ALPHA=0.5
BETA=1.0
DIVERSITY_WEIGHT=0.0
BATCH_SIZE=512
NUM_EPOCHS=200
TAU=0.001
MAX_LEN=20
BEAM_SIZE=30
PATIENCE=15
WEIGHT_DECAY=0.05

# T5 architecture
T5_ENCODER_LAYERS=6
T5_DECODER_LAYERS=6
T5_D_MODEL=128
T5_D_FF=512
T5_NUM_HEADS=4
T5_D_KV=64
T5_DROPOUT=0.1

WARMUP_RATIO=0.1

# Finetuning
FINETUNE_LR=4e-4
FINETUNE_EPOCHS=100
FINETUNE_PATIENCE=10
FINETUNE_WARMUP_RATIO=0.05
FINETUNE_SCHEDULER_TYPE="constant"

# Search space (extend as needed)
T5_LR_LIST=(5e-3)
RQVAE_LR_LIST=(2e-7 2e-8)
GAMMA_LIST=(0.01)
DELTA_LIST=(0.1)
SCHED_LIST=("constant_with_warmup")
SCHED_NAME_LIST=("warmconstant")

EXPS=()

build_experiments() {
  local idx=1
  for i in "${!T5_LR_LIST[@]}"; do
    for j in "${!RQVAE_LR_LIST[@]}"; do
      for k in "${!SCHED_LIST[@]}"; do
        for m in "${!GAMMA_LIST[@]}"; do
          for n in "${!DELTA_LIST[@]}"; do
              local t5lr=${T5_LR_LIST[$i]}
              local rqlr=${RQVAE_LR_LIST[$j]}
              local sched=${SCHED_LIST[$k]}
              local sched_name=${SCHED_NAME_LIST[$k]}
              local gamma=${GAMMA_LIST[$m]}
              local delta=${DELTA_LIST[$n]}
              local exp_name="Grid-Joint-g${gamma}-d${delta}-T5LR-${t5lr}-RQLR-${rqlr}-${sched_name}"
              local swan_name="${DATASET}-Grid-g${gamma}-d${delta}-${sched_name}-t5_${t5lr}-rq_${rqlr}"
              EXPS+=("${idx}|${t5lr}|${rqlr}|${sched}|${sched_name}|${gamma}|${delta}|${exp_name}|${swan_name}")
              idx=$((idx+1))
        done
      done
    done
  done
  done
}

run_experiment() {
  local rec="$1"
  IFS='|' read -r EXP_IDX T5_LR RQVAE_LR SCHED_TYPE SCHED_NAME GAMMA DELTA EXP_NAME SWAN_NAME <<< "${rec}"
  local OUTPUT_DIR="./ckpt/${DATASET}/${EXP_NAME}"
  mkdir -p "${OUTPUT_DIR}"

  echo "┌──────────────────────────────────────────────────────────────┐"
  echo "│ Experiment index: ${EXP_IDX} | EXP: ${EXP_NAME}"
  echo "│ T5_LR=${T5_LR}, RQVAE_LR=${RQVAE_LR}, Scheduler=${SCHED_NAME}"
  echo "│ gamma=${GAMMA}, delta=${DELTA}"
  echo "│ RQVAE: ${RQVAE_MODEL}"
  echo "│ SASRec: ${SASREC_EMB_PATH}"
  echo "└──────────────────────────────────────────────────────────────┘"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python model/train.py \
    --train_data "${TRAIN_DATA}" \
    --valid_data "${VALID_DATA}" \
    --test_data "${TEST_DATA}" \
    --item_emb_path "${ITEM_EMB}" \
    --rqvae_model_path "${RQVAE_MODEL}" \
    --save_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --t5_lr "${T5_LR}" \
    --rqvae_lr "${RQVAE_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --scheduler_type "${SCHED_TYPE}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --delta "${DELTA}" \
    --diversity_weight "${DIVERSITY_WEIGHT}" \
    --tau "${TAU}" \
    --use_sasrec_teacher \
    --sasrec_emb_path "${SASREC_EMB_PATH}" \
    --max_len "${MAX_LEN}" \
    --beam_size "${BEAM_SIZE}" \
    --num_layers "${T5_ENCODER_LAYERS}" \
    --num_decoder_layers "${T5_DECODER_LAYERS}" \
    --d_model "${T5_D_MODEL}" \
    --d_ff "${T5_D_FF}" \
    --num_heads "${T5_NUM_HEADS}" \
    --d_kv "${T5_D_KV}" \
    --dropout_rate "${T5_DROPOUT}" \
    --patience "${PATIENCE}" \
    --early_stop_metric "ndcg@10" \
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
    --finetune_early_stop_metric "ndcg@10" \
    --finetune_warmup_ratio "${FINETUNE_WARMUP_RATIO}" \
    --finetune_scheduler_type "${FINETUNE_SCHEDULER_TYPE}"

  echo " Index ${EXP_IDX} finished. Model saved to ${OUTPUT_DIR}"
  echo "---------------------------------------------------------------"
}

main() {
  build_experiments
  local total=${#EXPS[@]}
  echo "Total joint-training hyperparameter combinations: ${total}."

  local start=${START_IDX:-1}
  local end=${END_IDX:-${total}}
  if [[ -n "${IDX_LIST}" ]]; then
    IFS=',' read -ra LIST <<< "${IDX_LIST}"
    for idx in "${LIST[@]}"; do
      run_experiment "${EXPS[$((idx-1))]}"
    done
  else
    for ((i=start; i<=end; i++)); do
      run_experiment "${EXPS[$((i-1))]}"
    done
  fi
}

main "$@"
