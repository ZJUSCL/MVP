#!/bin/bash
_CONDA_ROOT="/home/yunzhu/anaconda3"
\. "$_CONDA_ROOT/etc/profile.d/conda.sh" || return $?
conda activate base


cd /home/yunzhu/MVP
# accelerate launch --num_processes 4 /home/yunzhu/v_know/guidance_parallel.py --json_file /home/yunzhu/v_know/actor.json
# 设置NCCL�~E�~W��~W��~W�为6�~O�~W��~H21600000毫�~R�~I
export NCCL_TIMEOUT=21600000
export NCCL_BLOCKING_WAIT=1

# �~E��~V�~X�~L~V设置
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
# Global configuration variables
export ATTN_LAYER=48
export TARGET_TOKEN_ID=","
export MAX_INFERENCES=2
export BATCH_SIZE=1
export MODEL_PATH="/run/determined/NAS1/Qwen/Qwen3-VL-32B-Instruct"

# Dataset paths
export SCREENSPOT_PRO_BASE_DIR="/run/determined/NAS1/ScreenSpot-Pro"  

# Install dependencies
pip install transformers==4.57.1
echo "Running ScreenSpot-Pro..."
torchrun --nproc_per_node=2 eval_sspro_qwen3vl_official.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file_dir "$SCREENSPOT_PRO_BASE_DIR/annotations" \
    --base_image_dir "$SCREENSPOT_PRO_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --output_path results/mvp_qwen3vl32b.json



export ATTN_LAYER=20
export MODEL_PATH="/run/determined/NAS1/Qwen/Qwen3-VL-8B-Instruct"
torchrun --nproc_per_node=4 eval_sspro_qwen3vl_official.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file_dir "$SCREENSPOT_PRO_BASE_DIR/annotations" \
    --base_image_dir "$SCREENSPOT_PRO_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --output_path results/mvp_qwen3vl8b.json

echo "All experiments completed!"