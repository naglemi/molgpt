#!/usr/bin/env python
"""
MolGPT GRPO Fine-tuning Script

This script demonstrates how to fine-tune a pre-trained MolGPT model using
Group Relative Policy Optimization (GRPO) to optimize towards specific
molecular properties.
"""

import os
import sys
import argparse
import json
import re
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen

# Import MolGPT modules
from train.model import GPT, GPTConfig
from train.dataset import SmileDataset
from train.utils import set_seed
from train.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from train.grpo_loss import MolecularRewardFunction
from generate.utils import sample, canonic_smiles
from moses.utils import get_mol

def main():
    parser = argparse.ArgumentParser(description='Fine-tune MolGPT with GRPO')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='moses2.csv', help='Path to dataset CSV')
    parser.add_argument('--vocab_path', type=str, default='moses2_stoi.json', help='Path to vocabulary JSON')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model checkpoint')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./grpo_models', help='Output directory')
    
    # GRPO arguments
    parser.add_argument('--epsilon', type=float, default=0.2, help='GRPO clipping parameter')
    parser.add_argument('--beta', type=float, default=0.01, help='KL penalty coefficient')
    parser.add_argument('--reward_type', type=str, default='qed', choices=['qed', 'logp', 'combined'], 
                        help='Reward function type')
    parser.add_argument('--group_size', type=int, default=8, help='Number of samples per input')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up wandb if requested
    wandb = None
    if args.use_wandb:
        import wandb
        wandb.init(
            project="molgpt-grpo",
            config=vars(args),
            name=f"grpo-{args.reward_type}-{args.beta}"
        )
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    
    # Split data
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'test']
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    
    # Extract SMILES, scaffolds, and properties
    train_smiles = train_data['smiles'].values
    val_smiles = val_data['smiles'].values
    
    train_scaffolds = train_data['scaffold_smiles'].values
    val_scaffolds = val_data['scaffold_smiles'].values
    
    # Extract optimization target property
    if args.reward_type == 'qed':
        train_props = train_data['qed'].values
        val_props = val_data['qed'].values
    elif args.reward_type == 'logp':
        train_props = train_data['logp'].values
        val_props = val_data['logp'].values
    else:
        # Use QED as default
        train_props = train_data['qed'].values
        val_props = val_data['qed'].values
    
    # Load vocabulary
    with open(args.vocab_path, 'r') as f:
        stoi = json.load(f)
    
    # Create reverse mapping
    itos = {int(i): ch for ch, i in stoi.items()}
    
    # Combine all SMILES for tokenization
    all_smiles = np.concatenate([train_smiles, val_smiles])
    all_scaffolds = np.concatenate([train_scaffolds, val_scaffolds])
    content = ' '.join(all_smiles.tolist() + all_scaffolds.tolist())
    
    # Create datasets
    block_size = 54  # Maximum SMILES length
    scaffold_max_len = 48  # Max scaffold length
    
    # Create datasets
    train_dataset = SmileDataset(args, train_smiles, content, block_size, aug_prob=0.1, 
                               prop=train_props, scaffold=train_scaffolds, scaffold_maxlen=scaffold_max_len)
    val_dataset = SmileDataset(args, val_smiles, content, block_size, aug_prob=0.0, 
                             prop=val_props, scaffold=val_scaffolds, scaffold_maxlen=scaffold_max_len)
    
    # Define model
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")
    
    mconf = GPTConfig(
        vocab_size, 
        block_size, 
        num_props=1,  # Using 1 property for conditioning
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        scaffold=True, 
        scaffold_maxlen=scaffold_max_len,
        lstm=False, 
        lstm_layers=0
    )
    
    # Create the model
    model = GPT(mconf)
    
    # Load pre-trained weights
    print(f"Loading pre-trained model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure GRPO training
    grpo_config = GRPOTrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=2*len(train_smiles)*block_size,
        ckpt_path=os.path.join(args.output_dir, f'molgpt_grpo_{args.reward_type}.pt'),
        num_workers=4,
        # GRPO-specific parameters
        epsilon=args.epsilon,
        beta=args.beta,
        reward_type=args.reward_type,
        group_size=args.group_size,
        temperature=1.0,
        top_k=40,
        ref_update_freq=0  # Never update reference model
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model, 
        train_dataset, 
        val_dataset, 
        grpo_config, 
        stoi, 
        itos
    )
    
    # Train with GRPO
    print("Starting GRPO fine-tuning")
    metrics = trainer.train_grpo(wandb=wandb)
    
    # Print final metrics
    print("\nTraining complete!")
    print("Final metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Generate sample molecules
    print("\nGenerating sample molecules...")
    model.eval()
    
    # Define property values to test
    prop_values = [0.5, 0.75, 0.9]  # QED values
    samples_per_condition = 10
    
    # Define regexe pattern for tokenization
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|\#|\-|\+|\\\\|\/|\:|\~|\@|\?|\>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    # Starting context
    context = "C"  # Start with carbon atom
    
    for prop_value in prop_values:
        print(f"\nGenerating molecules with {args.reward_type} = {prop_value}")
        
        # Tokenize context
        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(samples_per_condition, 1).to(device)
        
        # Set property conditioning
        p = torch.tensor([[prop_value]]).repeat(samples_per_condition, 1).to(device)
        
        # Generate molecules
        with torch.no_grad():
            y = sample(model, x, block_size, temperature=1.0, sample=True, top_k=40, prop=p, scaffold=None)
        
        # Convert to SMILES
        valid_mols = []
        valid_smiles = []
        
        for gen_mol in y:
            completion = ''.join([itos[int(i)] for i in gen_mol])
            completion = completion.replace('<', '')  # Remove padding
            
            mol = get_mol(completion)
            if mol:
                valid_mols.append(mol)
                valid_smiles.append(Chem.MolToSmiles(mol))
        
        # Print results
        print(f"Generated {len(valid_smiles)}/{samples_per_condition} valid molecules")
        for i, smiles in enumerate(valid_smiles):
            print(f"{i+1}. {smiles}")
        
        # Save molecule image if we have valid molecules
        if valid_mols:
            img_path = os.path.join(args.output_dir, f"molecules_{args.reward_type}_{prop_value}.png")
            img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(200, 200), 
                                    legends=[f"{args.reward_type}={prop_value}"] * len(valid_mols))
            img.save(img_path)
            print(f"Saved molecule visualization to {img_path}")
    
    print(f"\nGRPO fine-tuned model saved to {grpo_config.ckpt_path}")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 