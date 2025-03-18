#!/usr/bin/env python
"""
Production MolGPT GRPO Fine-tuning Script

High-performance implementation for multi-GPU training with GRPO to optimize
molecular properties. Optimized for memory efficiency and scaling.
"""

import os
import sys
import argparse
import json
import re
import copy
import time
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen

# Import MolGPT modules
from train.model import GPT, GPTConfig
from train.dataset import SmileDataset
from train.utils import set_seed
from train.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from train.grpo_loss import MolecularRewardFunction, GRPOLoss
from generate.utils import sample, canonic_smiles
from moses.utils import get_mol

def setup_distributed(rank, world_size, port='12355'):
    """Set up distributed process group for multi-GPU training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed process group"""
    dist.destroy_process_group()

def create_reference_model(model):
    """Create a frozen copy of model to serve as reference
    
    Args:
        model: MolGPT model
        
    Returns:
        ref_model: Frozen copy of model
    """
    # Use standard deep copy approach for safety
    ref_model = copy.deepcopy(model)
    
    # Freeze parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Set to evaluation mode
    ref_model.eval()
    
    return ref_model

def main_worker(gpu, ngpus_per_node, args):
    """Main worker function for distributed training
    
    Args:
        gpu: GPU id for this process
        ngpus_per_node: Number of GPUs per node
        args: Command line arguments
    """
    # Set up distributed
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        setup_distributed(args.rank, args.world_size)

    # Local rank and global rank
    args.local_rank = gpu
    
    # Adjust batch size based on number of GPUs
    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    
    # Set device
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility (with offset per GPU)
    set_seed(args.seed + gpu)
    
    # Only main process handles logging
    is_main_process = (not args.distributed) or (args.rank == 0)
    
    if is_main_process:
        print(f"Using device: {device}")
        print(f"Batch size per GPU: {args.batch_size}")
    
    # Set up wandb if requested (only on main process)
    wandb = None
    if args.use_wandb and is_main_process:
        import wandb
        wandb.init(
            project="molgpt-grpo",
            config=vars(args),
            name=f"grpo-{args.reward_type}-{args.beta}-{ngpus_per_node}gpu"
        )
    
    # Load data
    if is_main_process:
        print(f"Loading data from {args.data_path}")
    
    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    
    # Split data
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'test']
    
    if is_main_process:
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
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    
    # Define model
    vocab_size = len(stoi)
    if is_main_process:
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
    if is_main_process:
        print(f"Loading pre-trained model from {args.model_path}")
    
    # Load state dictionary
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        model._set_gradient_checkpointing(True)
    
    # Reference model - create BEFORE wrapping in DDP
    reference_model = create_reference_model(model)
    reference_model.to(device)
    
    # Wrap model with DDP for distributed training
    if args.distributed:
        # Use sync batch norm if batch norm is used
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[gpu], output_device=gpu,
                   find_unused_parameters=False)
    
    # Create output directory (only in main process)
    if is_main_process:
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
        num_workers=args.num_workers,
        # GRPO-specific parameters
        epsilon=args.epsilon,
        beta=args.beta,
        reward_type=args.reward_type,
        group_size=args.group_size,
        temperature=1.0,
        top_k=40,
        ref_update_freq=0  # Never update reference model
    )
    
    # Create custom GRPO Trainer with optimizations
    class OptimizedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add reward caching for efficiency
            self.reward_cache = {}
            # For distributed training
            self.distributed = args.distributed
            self.rank = args.rank if args.distributed else 0
            # Gradient scaler for mixed precision
            self.scaler = GradScaler() if args.fp16 else None
            
        def _compute_rewards(self, molecules):
            """Compute rewards with caching for efficiency"""
            rewards = []
            
            for smiles in molecules:
                if smiles in self.reward_cache:
                    rewards.append(self.reward_cache[smiles])
                else:
                    reward = self.reward_fn([smiles])[0].item()
                    self.reward_cache[smiles] = reward
                    rewards.append(reward)
            
            return torch.tensor(rewards, device=self.device)
    
    # Create trainer
    trainer = OptimizedGRPOTrainer(
        model, 
        train_dataset, 
        val_dataset, 
        grpo_config, 
        stoi, 
        itos,
        reference_model=reference_model  # Pass reference model directly
    )
    
    # Add distributed training attributes
    trainer.train_sampler = train_sampler
    trainer.val_sampler = val_sampler
    
    # Train with GRPO
    if is_main_process:
        print("Starting GRPO fine-tuning")
    
    # Record start time
    start_time = time.time()
    
    # Train
    metrics = trainer.train_grpo(wandb=wandb)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if is_main_process:
        # Print final metrics
        print("\nTraining complete!")
        print(f"Total training time: {elapsed_time:.2f} seconds")
        print("Final metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Generate sample molecules
        print("\nGenerating sample molecules...")
        model.eval()
        
        # Define property values to test
        prop_values = [0.5, 0.75, 0.9]  # QED values
        samples_per_condition = 10
        
        # Define regex pattern for tokenization
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
            
            # Unwrap model if using DDP
            generation_model = model.module if hasattr(model, "module") else model
            
            # Generate molecules
            with torch.no_grad():
                y = sample(generation_model, x, block_size, temperature=1.0, sample=True, top_k=40, prop=p, scaffold=None)
            
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
    
    # Clean up distributed training resources
    if args.distributed:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Fine-tune MolGPT with GRPO (Production Version)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='moses2.csv', help='Path to dataset CSV')
    parser.add_argument('--vocab_path', type=str, default='moses2_stoi.json', help='Path to vocabulary JSON')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model checkpoint')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./grpo_models', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # GRPO arguments
    parser.add_argument('--epsilon', type=float, default=0.2, help='GRPO clipping parameter')
    parser.add_argument('--beta', type=float, default=0.01, help='KL penalty coefficient')
    parser.add_argument('--reward_type', type=str, default='qed', choices=['qed', 'logp', 'combined'], 
                        help='Reward function type')
    parser.add_argument('--group_size', type=int, default=8, help='Number of samples per input')
    
    # Performance optimization arguments
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--sync_bn', action='store_true', help='Use synchronized batch normalization')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--rank', type=int, default=0, help='Rank of this process')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--dist_port', type=str, default='12355', help='Port for distributed training')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for performance analysis')
    
    args = parser.parse_args()
    
    # Automatic detection of distributed environment
    if not args.distributed and torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs, enabling distributed training")
        args.distributed = True
        args.world_size = torch.cuda.device_count()
    
    # Launch processes for multi-GPU training
    if args.distributed:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        # Single GPU training
        main_worker(0, 1, args)

if __name__ == "__main__":
    main() 