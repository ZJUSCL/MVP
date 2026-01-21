#!/bin/bash

# Global configuration variables
export ATTN_LAYER=24
export TARGET_TOKEN_ID=","
export MAX_INFERENCES=2
export BATCH_SIZE=1
export MODEL_PATH="/run/determined/NAS1/Qwen/Qwen3-VL-30B-A3B-Instruct"

# Dataset paths
export UI_VISION_BASE_DIR="/run/determined/NAS1/ServiceNow/ui-vision"
export SCREENSPOT_PRO_BASE_DIR="/run/determined/NAS1/ScreenSpot-Pro"  
export OSWORLD_G_BASE_DIR="/run/determined/NAS1/public/yunzhu/OSWorld-G/benchmark"

# Install dependencies
pip install transformers==4.57.1

# Run UI-Vision experiments
#echo "Running UI-Vision Basic..."
#torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
#    --attn_layer $ATTN_LAYER \
#    --target_token_id "$TARGET_TOKEN_ID" \
#    --max_inferences $MAX_INFERENCES \
#    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_basic.json" \
#    --base_image_dir "$UI_VISION_BASE_DIR/images" \
#    --model_path "$MODEL_PATH" \
#    --batch_size $BATCH_SIZE
#
#echo "Running UI-Vision Functional..."
#torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
#    --attn_layer $ATTN_LAYER \
#    --target_token_id "$TARGET_TOKEN_ID" \
#    --max_inferences $MAX_INFERENCES \
#    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_functional.json" \
#    --base_image_dir "$UI_VISION_BASE_DIR/images" \
#    --model_path "$MODEL_PATH" \
#    --batch_size $BATCH_SIZE
#
#echo "Running UI-Vision Spatial..."
#torchrun --nproc_per_node=2 mvp_uivision_qwen3vl.py \
#    --attn_layer $ATTN_LAYER \
#    --target_token_id "$TARGET_TOKEN_ID" \
#    --max_inferences $MAX_INFERENCES \
#    --json_file "$UI_VISION_BASE_DIR/annotations/element_grounding/element_grounding_spatial.json" \
#    --base_image_dir "$UI_VISION_BASE_DIR/images" \
#    --model_path "$MODEL_PATH" \
#    --batch_size $BATCH_SIZE

# Run ScreenSpot-Pro experiment
echo "Running ScreenSpot-Pro..."
torchrun --nproc_per_node=1 mvp_sspro_qwen3vl_moe.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file_dir "$SCREENSPOT_PRO_BASE_DIR/annotations" \
    --base_image_dir "$SCREENSPOT_PRO_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

# Run OSWorld-G experiment
#echo "Running OSWorld-G..."
#torchrun --nproc_per_node=2 mvp_osworldg_qwen3vl.py \
#    --attn_layer $ATTN_LAYER \
#    --target_token_id "$TARGET_TOKEN_ID" \
#    --max_inferences $MAX_INFERENCES \
#    --json_file "$OSWORLD_G_BASE_DIR/OSWorld-G_refined_classified.json" \
#    --base_image_dir "$OSWORLD_G_BASE_DIR/images" \
#    --model_path "$MODEL_PATH" \
#    --batch_size $BATCH_SIZE

echo "All experiments completed!"
