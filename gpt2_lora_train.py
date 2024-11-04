import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import random
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import torch.nn.functional as F

# Global configuration
CONFIG = {
    'output_dir': './hh_2v_lora_12layer',
    'num_epochs': 160,
    'batch_size': 32,
    'learning_rate': 3e-5,
    'train_samples': 15000,
    'val_samples': 1000,
    'eval_samples': 100,
    'log_interval': 100,
    'save_interval': 5,
    'hash': {
        'max_hops': 2,
        'vary_hops': True,
        'hash_length': 4,
        'chain_length': 10
    },
    'lora': {
        'use_lora': False,
        'rank': 128,
        'alpha': 64,
        'n_layers_to_modify': 1
    },
    'pre_trained_model': 'hh_2v/model_epoch_40.pt',
    'reasoning': {
        'enabled': False,
        'hash_length': 5,
        'chains': [3, 4, 5, 6],
        'num_chains': 4,
        'vary_hash':False
    }
}


class HashDataset(Dataset):
    def __init__(self, tokenizer, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.data = []
        for _ in range(num_samples):
            hash_map = self.generate_hash_map(chain_length=CONFIG['hash']['chain_length'], hash_length=CONFIG['hash']['hash_length'])
            start = list(hash_map.keys())[0]  # Start from the first hash in the chain
            if CONFIG['hash']['vary_hops']:
                current_hops = random.randint(1, CONFIG['hash']['max_hops'])
            else:
                current_hops = CONFIG['hash']['max_hops']
            target = self.perform_hash_hop(hash_map, start, current_hops)
            prompt = f"Map:\n"
            # Randomize the order of hash mappings in the prompt
            hash_items = list(hash_map.items())
            random.shuffle(hash_items)
            for key, value in hash_items:
                prompt += f"{key}=>{value}\n"
            prompt += f"Start: {start}\nHops: {current_hops}\nTarget:"
            full_text = f"{prompt} {target}"
            self.data.append((full_text, hash_map, start, current_hops, target, prompt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text, _, _, _, _, _ = self.data[idx]
        return self.tokenizer.encode(full_text, truncation=True, max_length=512)

    def get_eval_item(self, idx):
        return self.data[idx]

    @staticmethod
    def generate_hash_map(chain_length: int = CONFIG['hash']['chain_length'], hash_length: int = CONFIG['hash']['hash_length']) -> Dict[str, str]:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        hashes = [''.join(random.choice(chars) for _ in range(hash_length)) for _ in range(chain_length)]
        hash_map = {}
        for i in range(chain_length - 1):
            hash_map[hashes[i]] = hashes[i + 1]
        return hash_map

    @staticmethod
    def perform_hash_hop(hash_map: Dict[str, str], start: str, hops: int) -> str:
        current = start
        for _ in range(hops):
            if current in hash_map:
                current = hash_map[current]
            else:
                break  # Stop if we reach the end of the chain
        return current

import random
from typing import List, Tuple
from torch.utils.data import Dataset

class ReasoningHashDataset(Dataset):
    def __init__(self, tokenizer, num_samples: int = 1000, hash_length: int = 5, 
                 chains: List[int] = [2, 3, 4, 5, 6], vary_hash: bool = True, num_chains: int = 5):
        self.tokenizer = tokenizer
        self.data = []
        for _ in range(num_samples):
            if vary_hash:
                # Step 1: Sample 1 chain between the smallest and second biggest
                smallest_chain = min(chains)
                second_biggest_chain = sorted(chains)[-2]
                first_chain = random.choice([c for c in chains if smallest_chain <= c <= second_biggest_chain])
                
                # Step 2: Add random chains bigger than the selected chain
                larger_chains = [c for c in chains if c > first_chain]
                if num_chains - 1 <= len(larger_chains):
                    selected_chains = [first_chain] + random.sample(larger_chains, num_chains - 1)
                else:
                    selected_chains = [first_chain] + random.choices(larger_chains, k=num_chains - 1)
            else:
                selected_chains = chains[:num_chains]  # Take the first num_chains elements

            selected_chains.sort()  # Sort to ensure the shortest is first

            hash_list, start, shortest_target = self.generate_hash_list(
                hash_length=hash_length,
                chains=selected_chains
            )
            prompt = f"Map:\n"
            random.shuffle(hash_list)
            for key, value in hash_list:
                prompt += f"{key}=>{value}\n"
            prompt += f"Start: {start}\nTask: shortest path\nTarget:"
            full_text = f"{prompt} {shortest_target}"
            self.data.append((full_text, hash_list, start, shortest_target, prompt))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text, _, _, _, _ = self.data[idx]
        return self.tokenizer.encode(full_text, truncation=True, max_length=512)

    def get_eval_item(self, idx):
        return self.data[idx]

    @staticmethod
    def generate_hash_list(hash_length: int, chains: List[int]) -> Tuple[List[Tuple[str, str]], str, str]:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        hash_list = []
        
        # Generate start hash
        start = ''.join(random.choice(chars) for _ in range(hash_length))
        
        shortest_length = min(chains)
        shortest_target = None
        
        # Generate chains
        for chain_length in chains:
            current = start
            for _ in range(chain_length - 1):  # -1 because start is already counted
                next_hash = ''.join(random.choice(chars) for _ in range(hash_length))
                hash_list.append((current, next_hash))
                current = next_hash
            
            if chain_length == shortest_length:
                shortest_target = current
        
        return hash_list, start, shortest_target

    @staticmethod
    def find_shortest_path(hash_list: List[Tuple[str, str]], start: str) -> int:
        graph = {}
        for src, dest in hash_list:
            if src not in graph:
                graph[src] = []
            graph[src].append(dest)

        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, length = queue.pop(0)
            if current not in graph:
                return length  # Return the length of the path
            
            if current not in visited:
                visited.add(current)
                for next_hash in graph[current]:
                    queue.append((next_hash, length + 1))
        
        return -1  # This should never happen if the hash list is correctly generated

def collate_fn(batch):
    max_len = max(len(item) for item in batch)
    padded_batch = []
    attention_masks = []
    for item in batch:
        padded_item = item + [0] * (max_len - len(item))
        attention_mask = [1] * len(item) + [0] * (max_len - len(item))
        padded_batch.append(padded_item)
        attention_masks.append(attention_mask)
    return torch.tensor(padded_batch), torch.tensor(attention_masks)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=CONFIG['lora']['rank'], alpha=CONFIG['lora']['alpha'], dtype=torch.float16):
        super().__init__()
        rank = CONFIG['lora']['rank']
        alpha = CONFIG['lora']['alpha']
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=dtype) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype) / rank)
        self.scaling = alpha / rank

    def forward(self, x):
        if x.dtype != self.lora_A.dtype:
            x = x.to(self.lora_A.dtype)
        return (x @ self.lora_A @ self.lora_B) * self.scaling

def add_lora_to_linear_and_replace(layer):
    in_features, out_features = layer.nx, layer.nf
    dtype = layer.weight.dtype
    lora_layer = LoRALayer(in_features, out_features, dtype=dtype)

    class CombinedLayer(nn.Module):
        def __init__(self, base_layer, lora_layer):
            super().__init__()
            self.base_layer = base_layer
            self.lora_layer = lora_layer

        def forward(self, x):
            return self.base_layer(x) + self.lora_layer(x)

    combined_layer = CombinedLayer(layer, lora_layer)
    return combined_layer.to(layer.weight.device)

def replace_with_lora_layers_last_n(model, n_layers_to_modify=CONFIG['lora']['n_layers_to_modify']):
    n_layers_to_modify = CONFIG['lora']['n_layers_to_modify']
    count = 0
    for name, module in reversed(list(model.named_modules())):
        if any(x in name for x in ["mlp.c_proj"]): # ['attn.c_attn', 'attn.c_proj']
            parent_name = name.rsplit(".", 1)[0]
            parent_module = model.get_submodule(parent_name) if parent_name else model
            child_name = name.split(".")[-1]

            layer_type = "Attention" if "attn" in name else "FFN"
            print(f"Replacing {child_name} ({layer_type}) in {parent_name} with LoRA-enhanced layer")
            setattr(parent_module, child_name, add_lora_to_linear_and_replace(module))

            count += 1
            if count >= n_layers_to_modify:
                break

    return model

def set_lora_parameters_trainable(model):
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            print(f"LoRA parameter: {name}, size: {param.shape}")
        else:
            param.requires_grad = False

def load_pre_trained_model(model, pre_trained_path):
    if os.path.exists(pre_trained_path):
        print(f"Loading pre-trained model from {pre_trained_path}")
        state_dict = torch.load(pre_trained_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        print(f"Pre-trained model not found at {pre_trained_path}. Using default initialization.")
    return model

def get_dataset_class():
    """Factory function to return the appropriate dataset class based on configuration."""
    return ReasoningHashDataset if CONFIG['reasoning']['enabled'] else HashDataset

def create_dataset(tokenizer, num_samples):
    """Create a dataset instance based on the current configuration."""
    DatasetClass = get_dataset_class()
    if CONFIG['reasoning']['enabled']:
        return DatasetClass(tokenizer, num_samples=num_samples,
                            hash_length=CONFIG['reasoning']['hash_length'],
                            chains=CONFIG['reasoning']['chains'],
                            num_chains=CONFIG['reasoning']['num_chains'],
                            vary_hash=CONFIG['reasoning']['vary_hash'])
    else:
        return DatasetClass(tokenizer, num_samples=num_samples)

import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, tokenizer, logging=False):
    def log(message):
        if logging:
            logger.info(message)

    log(f"Input shapes: logits {logits.shape}, labels {labels.shape}")

    # Find the index of the "Target" token
    target_token_id = tokenizer.encode("Target", add_special_tokens=False)[0]
    log(f"'Target' token ID: {target_token_id}")

    # Find the positions of "Target" tokens in the entire batch
    target_positions = (labels == target_token_id).nonzero(as_tuple=True)
    batch_indices, target_indices = target_positions

    if len(batch_indices) == 0:
        log("'Target' token not found in any batch")
        return None, [], []

    # Create a mask for valid positions (after "Target" token)
    batch_size, seq_length = labels.shape
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for b, t in zip(batch_indices, target_indices):
        mask[b, t+1:] = True

    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # Apply the mask to shifted logits and labels
    valid_logits = shift_logits[shift_mask].view(-1, shift_logits.size(-1))
    valid_labels = shift_labels[shift_mask].view(-1)

    # Compute loss
    loss = F.cross_entropy(valid_logits, valid_labels)
    log(f"Computed loss: {loss.item()}")

    # Get predictions
    predictions = torch.argmax(valid_logits, dim=-1)

    # Reconstruct full sequences for logging
    full_predictions = torch.zeros_like(shift_labels)
    full_predictions[shift_mask] = predictions
    
    predicted_texts = []
    actual_texts = []

    for i in range(batch_size):
        if i in batch_indices:
            start_idx = target_indices[batch_indices == i].item() + 1
            pred_text = tokenizer.decode(full_predictions[i, start_idx-1:])
            actual_text = tokenizer.decode(shift_labels[i, start_idx-1:])
            predicted_texts.append(pred_text)
            actual_texts.append(actual_text)
            if logging:
                log(f"Batch {i}:")
                log(f"Predicted: {pred_text}")
                log(f"Actual: {actual_text}")
        else:
            predicted_texts.append("")
            actual_texts.append("")
            if logging:
                log(f"Batch {i}: No 'Target' token found")

    return loss

def train():
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Load pre-trained model
    model = load_pre_trained_model(model, CONFIG['pre_trained_model'])
    
    if CONFIG['reasoning']['enabled']:
        print("Reasoning dataset enabled!")

    if CONFIG['lora']['use_lora']:
        model = replace_with_lora_layers_last_n(model)
        set_lora_parameters_trainable(model)
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = create_dataset(tokenizer, CONFIG['train_samples'])
    val_dataset = create_dataset(tokenizer, CONFIG['val_samples'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(trainable_params, lr=CONFIG['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * CONFIG['num_epochs'])
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(os.path.join(CONFIG['output_dir'], "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(CONFIG['output_dir'], "checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        model.train()
        total_train_loss = 0
        
        # Regenerate train dataset every three epochs
        if epoch % 1 == 0:
            train_dataset = create_dataset(tokenizer, CONFIG['train_samples'])
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
        
        for batch_idx, (batch, attention_masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")):
            batch, attention_masks = batch.to(device), attention_masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch, attention_mask=attention_masks)
            logits = outputs.logits
            
            loss = compute_loss(logits, batch, tokenizer)

            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (batch_idx + 1) % CONFIG['log_interval'] == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Batch {batch_idx+1}/{len(train_loader)}")
                print(f"Train Loss: {total_train_loss/(batch_idx+1):.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch, attention_masks in val_loader:
                batch, attention_masks = batch.to(device), attention_masks.to(device)
                outputs = model(batch, attention_mask=attention_masks)
                logits = outputs.logits
                
                loss = compute_loss(logits, batch, tokenizer, logging=True)
                total_val_loss += loss.item()
            
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} Summary")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, os.path.join(CONFIG['output_dir'], "checkpoint.pt"))
        
        # Save model
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], f"model_epoch_{epoch+1}.pt"))
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'loss_plot.png'))
    
    # Final evaluation
    evaluate(model, tokenizer)

def evaluate(model, tokenizer):
    model.eval()
    
    def generate_hash(prompt, max_length=20):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)
        pad_token_id = tokenizer.eos_token_id
        
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=pad_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Target:")[-1].strip()[:CONFIG['hash']['hash_length']]
    
    eval_dataset = HashDataset(tokenizer, num_samples=CONFIG['eval_samples'])
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(eval_dataset)):
        full_text, hash_map, start, hops, actual_target, prompt = eval_dataset.get_eval_item(i)
        predicted_target = generate_hash(prompt)
        
        total_predictions += 1
        if predicted_target == actual_target:
            correct_predictions += 1
        
        print(f"Start: {start}, Hops: {hops}")
        print(f"Predicted: {predicted_target}")
        print(f"Actual: {actual_target}")
        print(f"Correct: {predicted_target == actual_target}\n")
    
    accuracy = correct_predictions / total_predictions
    print(f"\nOverall Accuracy: {accuracy:.2f}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    
    # Save evaluation results
    with open(os.path.join(CONFIG['output_dir'], 'evaluation_results.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.2f}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Total Predictions: {total_predictions}\n")

def save_config(config, filename='config.json'):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename='config.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return CONFIG

def update_config(args):
    config = load_config()
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.train_samples:
        config['train_samples'] = args.train_samples
    if args.val_samples:
        config['val_samples'] = args.val_samples
    if args.eval_samples:
        config['eval_samples'] = args.eval_samples
    if args.log_interval:
        config['log_interval'] = args.log_interval
    if args.save_interval:
        config['save_interval'] = args.save_interval
    if args.max_hops:
        config['hash']['max_hops'] = args.max_hops
    if args.vary_hops is not None:
        config['hash']['vary_hops'] = args.vary_hops
    if args.hash_length:
        config['hash']['hash_length'] = args.hash_length
    if args.chain_length:
        config['hash']['chain_length'] = args.chain_length
    config['lora']['use_lora'] = True
    if args.lora_rank:
        config['lora']['rank'] = args.lora_rank
    if args.lora_alpha:
        config['lora']['alpha'] = args.lora_alpha
    if args.lora_layers:
        config['lora']['n_layers_to_modify'] = args.lora_layers
    if args.pre_trained_model:
        config['pre_trained_model'] = args.pre_trained_model
    config['reasoning']['enabled'] = True
    if args.reasoning_hash_length:
        config['reasoning']['hash_length'] = args.reasoning_hash_length
    if args.reasoning_chains:
        config['reasoning']['chains'] = [int(x) for x in args.reasoning_chains.split(',')]
    if args.num_chains:
        config['reasoning']['num_chains'] = args.num_chains
    config['reasoning']['vary_hash'] = True
    
    save_config(config)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hash chain model with customizable configuration.")
    parser.add_argument("--output_dir", type=str, help="Output directory for model checkpoints and results")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--train_samples", type=int, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, help="Number of validation samples")
    parser.add_argument("--eval_samples", type=int, help="Number of evaluation samples")
    parser.add_argument("--log_interval", type=int, help="Logging interval during training")
    parser.add_argument("--save_interval", type=int, help="Model saving interval (in epochs)")
    parser.add_argument("--max_hops", type=int, help="Maximum number of hops in hash chain")
    parser.add_argument("--vary_hops", type=bool, help="Whether to vary the number of hops")
    parser.add_argument("--hash_length", type=int, help="Length of each hash in the chain")
    parser.add_argument("--chain_length", type=int, help="Length of the hash chain")
    parser.add_argument("--use_lora", type=bool, help="Whether to use LoRA")
    parser.add_argument("--lora_rank", type=int, help="Rank for LoRA layers")
    parser.add_argument("--lora_alpha", type=int, help="Alpha for LoRA layers")
    parser.add_argument("--lora_layers", type=int, help="Number of layers to modify with LoRA")
    parser.add_argument("--pre_trained_model", type=str, help="Path to pre-trained model")
    parser.add_argument("--reasoning", type=bool, help="Use reasoning hash dataset")
    parser.add_argument("--reasoning_hash_length", type=int, help="Hash length for reasoning dataset")
    parser.add_argument("--reasoning_chains", type=str, help="Comma-separated list of chain lengths for reasoning dataset")
    parser.add_argument("--num_chains", type=int, help="Number of chains for reasoning dataset")
    parser.add_argument("--vary_hash", type=bool, help="Whether to vary the hash for reasoning dataset")
    
    args = parser.parse_args()
    
    # Update configuration based on command-line arguments
    CONFIG = update_config(args)
    
    # Start training
    train()