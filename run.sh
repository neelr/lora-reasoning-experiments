#!/bin/bash

# Path to your Python script
SCRIPT_PATH="./gpt2_lora_train.py"

# Common parameters
COMMON_PARAMS="--num_epochs 40 --batch_size 16 --learning_rate 3e-5 --train_samples 15000 --val_samples 50 --eval_samples 100 --log_interval 100 --save_interval 2 --lora_rank 128 --lora_alpha 64"

# Run no LoRA base model
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_base_no_lora_15 --lora_layers 12 --max_hops 15 --vary_hops True --hash_length 4 --chain_length 20 --pre_trained_model output_base_no_lora_5/model_epoch_40.pt --reasoning False

# Run no LoRA base model vary 4 reasoning
python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./base_reasoning_23456_3chain --vary_hops True --hash_length 4 --pre_trained_model base_reasoning_2345_4chain_vary/model_epoch_40.pt --reasoning True --reasoning_chains 2,3,4,5,6 --vary_hash=True --num_chains=3
 
# Run 12 layer LoRA (1-4 hops)
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_12layer_lora_1-4hops --lora_layers 12 --max_hops 4 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt

# Run 6 layer LoRA (1-4 hops)e
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_6layer_lora_1-4hops --lora_layers 6 --max_hops 4 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt

# Run 6 layer LoRA (1-4 hops) with different rank/alpha
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_6layer_lora_1-4hops_rank_alpha_variation --lora_layers 6 --max_hops 4 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt --lora_rank 256 --lora_alpha 128

# Run 6 layer LoRA (1-4 hops) with another rank/alpha variation
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_6layer_lora_1-4hops_rank_alpha_variation_2 --lora_layers 6 --max_hops 4 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt --lora_rank 64 --lora_alpha 32

# Run 6 layer LoRA (1-2 hops)
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_6layer_lora_1-2hops_alpha_variation --lora_layers 6 --max_hops 2 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt

# Run 6 layer LoRA (1-2 hops)
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_6layer_lora_1-2hops_alpha_variation_2 --lora_layers 6 --max_hops 2 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt --lora_rank 64 --lora_alpha 32

# Run 12 layer LoRA (1-2 hops, no pre-trained model)
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_12layer_loraffn_1-2hops --lora_layers 12 --max_hops 2 --vary_hops True --hash_length 4 --chain_length 10 --pre_trained_model hh_2v/model_epoch_40.pt

# New run: Base model, no LoRA, reasoning [3,3,2], hash length 4
# python $SCRIPT_PATH $COMMON_PARAMS --output_dir ./output_base_no_lora_reasoning --hash_length 4 --reasoning True --reasoning_hash_length 4 --reasoning_chains 3,3,2 --pre_trained_model none