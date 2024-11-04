import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import random
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global configuration
CONFIG = {
    'output_dir': './hh_2vf',
    'num_epochs': 40,
    'batch_size': 32,
    'learning_rate': 3e-5,
    'train_samples': 15000,
    'val_samples': 1000,
    'eval_samples': 100,
    'log_interval': 100,
    'save_interval': 1,
    'hash': {
        'max_hops': 2,
        'vary_hops': True,
        'hash_length': 4,
        'chain_length': 10
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

def train():
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = HashDataset(tokenizer, num_samples=CONFIG['train_samples'])
    val_dataset = HashDataset(tokenizer, num_samples=CONFIG['val_samples'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
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
        # if epoch % 3 == 0:
        #     train_dataset = HashDataset(tokenizer, num_samples=CONFIG['train_samples'])
        #     train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
        
        for batch_idx, (batch, attention_masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")):
            batch, attention_masks = batch.to(device), attention_masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch, attention_mask=attention_masks, labels=batch)
            loss = outputs.loss
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
                outputs = model(batch, attention_mask=attention_masks, labels=batch)
                total_val_loss += outputs.loss.item()
        
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

if __name__ == "__main__":
    train()