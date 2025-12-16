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
export UI_VISION_BASE_DIR="/run/determined/NAS1/ServiceNow/ui-vision"
export SCREENSPOT_PRO_BASE_DIR="/run/determined/NAS1/ScreenSpot-Pro"  
export OSWORLD_G_BASE_DIR="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark"

# Install dependencies
pip install transformers==4.57.1

# Run UI-Vision experiments
echo "Running UI-Vision Basic..."
torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_basic.json" \
    --base_image_dir "$UI_VISION_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

echo "Running UI-Vision Functional..."
torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_functional.json" \
    --base_image_dir "$UI_VISION_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

echo "Running UI-Vision Spatial..."
torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_spatial.json" \
    --base_image_dir "$UI_VISION_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

# Run ScreenSpot-Pro experiment
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

Run OSWorld-G experiment
echo "Running OSWorld-G..."
torchrun --nproc_per_node=2 mvp_osworldg_qwen3vl.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file "$OSWORLD_G_BASE_DIR/OSWorld-G_refined_classified.json" \
    --base_image_dir "$OSWORLD_G_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE



echo "All experiments completed!"