#!/bin/bash

# Base parameters
BASE_MODEL="base_reasoning_23456_3chain/model_epoch_40.pt"
COMMON_PARAMS="--num_epochs 40 --batch_size 16 --learning_rate 3e-5 --train_samples 15000 --val_samples 16 --eval_samples 100 --log_interval 100 --save_interval 5"

# Hop configuration
HOP_PARAMS="--reasoning True --reasoning_chains 2,3,4,5,6 --vary_hash=True --num_chains=4"

# LoRA configurations to test
# Format: "rank alpha"
# Using alpha ≈ rank/2 for better scaling
CONFIGS=(
    "8 4"       # Starting point - minimal adaptation
    "16 8"      # Classic LoRA paper value
    "32 16"     # Medium adaptation
    "64 32"     # Higher adaptation
    "128 64"    # Very high adaptation
    "256 128"   # Maximum adaptation
)

# Run training for each configuration
for config in "${CONFIGS[@]}"; do
    read -r rank alpha <<< "$config"
    
    OUTPUT_DIR="./output_lora_r${rank}_a${alpha}_hops1-6"
    
    echo "Starting training with rank=${rank}, alpha=${alpha}"
    echo "Output directory: ${OUTPUT_DIR}"
    
    python gpt2_lora_train.py $COMMON_PARAMS \
        --output_dir "$OUTPUT_DIR" \
        --pre_trained_model "$BASE_MODEL" \
        --lora_layers 12 \
        --lora_rank "$rank" \
        --lora_alpha "$alpha" \
        $HOP_PARAMS
    
    echo "Completed training for rank=${rank}, alpha=${alpha}"
    echo "----------------------------------------"
done

# Print completion message
echo "All LoRA rank sweep training runs completed!"