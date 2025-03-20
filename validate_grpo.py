#!/usr/bin/env python
"""
GRPO Validation Script - Small-scale test of all components
"""

import torch
import wandb
import kagglehub
from trl import GRPOTrainer, GRPOConfig
from train.model import GPT, GPTConfig
from rdkit import Chem
from rdkit.Chem import QED, sascorer
from copy import deepcopy
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_molecular_rewards(molecules):
    """Compute rewards based on QED and synthetic accessibility"""
    rewards = []
    valid_count = 0
    total_sa = 0
    total_qed = 0
    
    for smiles in molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rewards.append(0.0)
                continue
                
            # Calculate scores
            qed = QED.qed(mol)
            sa_score = -sascorer.calculateScore(mol)  # Negative because lower SA is better
            reward = 0.7 * qed + 0.3 * sa_score
            
            # Track metrics
            valid_count += 1
            total_sa += -sa_score  # Convert back to positive for logging
            total_qed += qed
            
            rewards.append(float(reward))
        except:
            rewards.append(0.0)
            
    # Log batch statistics
    if len(molecules) > 0:
        validity_rate = valid_count / len(molecules)
        mean_sa = total_sa / valid_count if valid_count > 0 else 0
        mean_qed = total_qed / valid_count if valid_count > 0 else 0
        
        wandb.log({
            "validity_rate": validity_rate,
            "mean_sa_score": mean_sa,
            "mean_qed_score": mean_qed
        })
    
    return torch.tensor(rewards)

def validate_grpo():
    """Run small-scale GRPO validation"""
    
    # Initialize wandb
    wandb.init(
        project="molgpt-grpo-validation",
        config={
            "validation_steps": 100,
            "batch_size": 2,
            "group_size": 4,
            "learning_rate": 5e-6,
            "epsilon": 0.1,
            "beta": 0.02
        }
    )
    
    # Create output directory
    output_dir = Path("outputs/grpo_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading pretrained checkpoint...")
    checkpoint_path = kagglehub.dataset_download("virajbagal/ligflow-final-weights")
    
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model with same config as pretrained
    config = GPTConfig(
        vocab_size=checkpoint["config"]["vocab_size"],
        block_size=54,  # Standard SMILES length
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    
    logger.info("Initializing models...")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    ref_model = deepcopy(model)
    
    # Conservative validation config
    grpo_config = GRPOConfig(
        learning_rate=5e-6,          # Conservative learning rate
        per_device_train_batch_size=2,
        group_size=4,                # Minimum viable group size
        max_steps=100,               # Short validation run
        output_dir=str(output_dir),
        logging_steps=1,             # Log every step
        save_steps=25,               # Save 4 checkpoints
        eval_steps=10,               # Evaluate frequently
        epsilon=0.1,                 # Conservative clipping
        beta=0.02,                   # Stronger KL penalty
    )
    
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=grpo_config,
        compute_rewards=compute_molecular_rewards,
        use_reward_scaling=True,
        use_advantage_normalization=True
    )
    
    # Add custom logging callbacks
    def log_step_metrics(args, state, metrics, **kwargs):
        """Log detailed metrics at each step"""
        wandb.log({
            "step": state.global_step,
            "loss": metrics["loss"],
            "kl_div": metrics.get("kl_div", 0),
            "policy_loss": metrics.get("policy_loss", 0),
            "reward_mean": metrics.get("reward_mean", 0),
            "reward_std": metrics.get("reward_std", 0)
        })
    
    trainer.add_callback(log_step_metrics)
    
    logger.info("Starting validation training...")
    try:
        trainer.train()
        logger.info("Validation training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Save final validation metrics
        wandb.log({
            "validation_completed": True,
            "final_step": trainer.state.global_step
        })
        wandb.finish()

if __name__ == "__main__":
    validate_grpo() 