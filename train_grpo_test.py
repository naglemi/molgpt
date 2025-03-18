#!/usr/bin/env python
"""
Simplified GRPO test script that initializes a new model without loading pretrained weights.
This is just to verify the implementation runs without crashing.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
import time
import random
from torch.nn import functional as F

# Import MolGPT modules
from train.dataset import SmileDataset
from train.model import GPT, GPTConfig
from train.utils import set_seed
from train.grpo_utils import PPOLoss
from train.rewards import SimilarityReward

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def create_reference_model(model):
    """Create a frozen copy of the model to serve as reference"""
    ref_model = GPT(model.config)
    ref_model.load_state_dict(model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model

def main():
    # Set random seed
    set_seed(42)
    
    # Load data
    data_path = '/home/ubuntu/molgpt/datasets/moses2.csv'
    print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    # Display column information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Split data into train and validation sets
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'test']
    
    if len(val_data) == 0:
        # If no test split is found, create a random split
        train_size = int(0.8 * len(df))
        train_data = df[:train_size]
        val_data = df[train_size:train_size+100]  # Just use 100 samples for validation
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    
    # Extract SMILES, scaffolds, and properties
    train_smiles = train_data['smiles'].tolist()
    train_scaffolds = train_data['scaffold_smiles'].tolist()
    train_props = train_data['qed'].values
    
    val_smiles = val_data['smiles'].tolist()
    val_scaffolds = val_data['scaffold_smiles'].tolist()
    val_props = val_data['qed'].values
    
    # Load vocabulary
    vocab_path = '/home/ubuntu/molgpt/moses2_stoi.json'
    with open(vocab_path, 'r') as f:
        stoi = json.load(f)
    
    # Create reverse mapping for tokenization
    itos = {v: k for k, v in stoi.items()}
    
    # Dataset parameters
    block_size = 54
    scaffold_max_len = 48
    
    # Create a fake args object with debug attribute
    class Args:
        def __init__(self):
            self.debug = False
            
    args = Args()
    
    # Combine all smiles for content parameter
    all_smiles = train_smiles + val_smiles
    all_scaffolds = train_scaffolds + val_scaffolds
    content = ' '.join(all_smiles + all_scaffolds)
    
    # Create datasets
    train_dataset = SmileDataset(args, train_smiles, content, block_size, aug_prob=0.1, 
                              prop=train_props, scaffold=train_scaffolds, scaffold_maxlen=scaffold_max_len)
    
    val_dataset = SmileDataset(args, val_smiles, content, block_size, aug_prob=0.0, 
                            prop=val_props, scaffold=val_scaffolds, scaffold_maxlen=scaffold_max_len)
    
    print(f"data has {len(train_dataset)} smiles, {len(stoi)} unique characters.")
    
    # Model configuration
    vocab_size = len(stoi)
    n_layer = 8
    n_head = 8
    n_embd = 256
    
    # Initialize a new model with random weights
    mconf = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        num_props=1,
        scaffold=True,
        scaffold_maxlen=scaffold_max_len,
        lstm=False,
        lstm_layers=0
    )
    
    model = GPT(mconf)
    model.to(device)
    
    # Create a reference model for GRPO
    ref_model = GPT(mconf)
    ref_model.load_state_dict(model.state_dict())
    ref_model.to(device)
    
    # Setup GRPO loss and reward
    reward_fn = SimilarityReward()
    grpo_loss = PPOLoss(
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        kl_coef=0.2
    )
    
    # Setup dataloader and optimizer
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Function to sample molecules
    def sample_molecules(model, x, pad_token_id, max_length=100):
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            for _ in range(max_length):
                logits, _ = model(x)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                x = torch.cat([x, next_token], dim=1)
                if (next_token.item() == pad_token_id).all():
                    break
            return x
    
    # Training loop
    print("\nStarting training loop...")
    max_steps = 5  # Only run for 5 steps to test
    
    for step, batch in enumerate(train_loader):
        if step >= max_steps:
            break
        
        # Unpack batch based on what SmileDataset returns
        if len(batch) == 3:
            x, y, p = batch
            scaffold = None
        elif len(batch) == 4:
            x, y, p, scaffold = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
            
        model.train()
        x = x.to(device)
        y = y.to(device) if y is not None else None
        p = p.to(device) if p is not None else None
        scaffold = scaffold.to(device) if scaffold is not None else None
        
        # Forward pass
        logits, _ = model(x)
        
        # Sample trajectories for GRPO
        sampled_smiles = sample_molecules(model, x[:, :10], stoi.get('<pad>', 0))
        
        # For this test, we'll just use dummy rewards
        # In a real implementation, we would convert sampled_smiles to actual SMILES strings
        # and calculate rewards using reward_fn
        dummy_smiles = ["C=CC(=O)CC"] * sampled_smiles.size(0)  # Placeholder
        rewards = reward_fn(dummy_smiles)
        
        # Calculate reference model logits
        with torch.no_grad():
            ref_logits, _ = ref_model(x)
        
        # Calculate GRPO loss
        policy_loss, value_loss, entropy_loss, kl_loss = grpo_loss(
            logits, 
            ref_logits,
            x, 
            rewards
        )
        
        loss = policy_loss + value_loss - entropy_loss + kl_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print metrics
        print(f"Step {step}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, "
              f"KL Loss: {kl_loss.item():.4f}, Mean Reward: {rewards.mean().item():.4f}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 