#!/bin/bash
# Stage 1: RQVAE Pre-training


set -e

cd "$(dirname "$0")"

# =============================================================================
# Hyperparameter Configuration
# =============================================================================

GPU_ID="5"
DATASET="Beauty"

RQVAE_DATA_PATH="../dataset/${DATASET}/item_emb_td.parquet"
RQVAE_CKPT_DIR="./ckpt/${DATASET}"

RQVAE_EPOCHS="3000"
RQVAE_BATCH_SIZE="1024"
RQVAE_LR="1e-3"
RQVAE_EVAL_STEP="50"
RQVAE_NUM_EMB_LIST="256 256 256"
RQVAE_E_DIM="32"
RQVAE_LAYERS="512 256 128 64"
RQVAE_SEED="2025"

DROP_LAST=""

# VQ parameters (vq_type: standard | soft)
RQVAE_VQ_TYPE="soft"
RQVAE_DISTANCE="l2"
RQVAE_TEMPERATURE="0.01"
RQVAE_KMEANS_INIT="True"
RQVAE_KMEANS_ITERS="100"
RQVAE_NORM_TYPE="none"

# Only effective for SoftVectorQuantizer (soft)
RQVAE_TEMP_SCHEDULE="linear"
RQVAE_TEMP_FINAL="0.001"
RQVAE_TEMP_START_EPOCH="0"
RQVAE_TEMP_END_EPOCH="3000"
RQVAE_DIVERSITY_SCALE="1e-4"

cd rqvae
python main.py \
    --data_path "$RQVAE_DATA_PATH" \
    --ckpt_dir "$RQVAE_CKPT_DIR" \
    --epochs $RQVAE_EPOCHS \
    --batch_size $RQVAE_BATCH_SIZE \
    --lr $RQVAE_LR \
    --eval_step $RQVAE_EVAL_STEP \
    --num_emb_list $RQVAE_NUM_EMB_LIST \
    --e_dim $RQVAE_E_DIM \
    --layers $RQVAE_LAYERS \
    --device cuda:0 \
    --gpu_ids "$GPU_ID" \
    --seed $RQVAE_SEED \
    --norm_type "$RQVAE_NORM_TYPE" \
    --vq_type "$RQVAE_VQ_TYPE" \
    --distance "$RQVAE_DISTANCE" \
    --temperature $RQVAE_TEMPERATURE \
    --temperature_schedule "$RQVAE_TEMP_SCHEDULE" \
    --temperature_final $RQVAE_TEMP_FINAL \
    --temperature_anneal_start_epoch $RQVAE_TEMP_START_EPOCH \
    --temperature_anneal_end_epoch $RQVAE_TEMP_END_EPOCH \
    --kmeans_init $RQVAE_KMEANS_INIT \
    --kmeans_iters $RQVAE_KMEANS_ITERS \
    --diversity_scale $RQVAE_DIVERSITY_SCALE \
    $([ -n "$DROP_LAST" ] && echo "--drop_last")
cd ..
echo "Pre-trained weights saved at rqvae/ckpt/${DATASET}/latest/best_loss_model.pth"
