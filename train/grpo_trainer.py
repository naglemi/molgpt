"""
GRPO Trainer for MolGPT

Implements Group Relative Policy Optimization training for MolGPT models,
extending the standard trainer with RL capabilities.
"""

import math
import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
import re
from rdkit import Chem

# Import our modules
from train.trainer import Trainer, TrainerConfig
from train.model import GPT, GPTConfig
from train.grpo_loss import GRPOLoss, create_reference_model, MolecularRewardFunction
from generate.utils import sample, canonic_smiles
from moses.utils import get_mol

logger = logging.getLogger(__name__)

class GRPOTrainerConfig(TrainerConfig):
    """Configuration for GRPO Trainer
    
    Extends the standard TrainerConfig with GRPO-specific parameters.
    """
    
    # GRPO parameters
    epsilon = 0.2  # Clipping parameter
    beta = 0.01  # KL penalty coefficient
    reward_type = 'qed'  # Type of molecular reward ('qed', 'logp', 'combined')
    
    # Sampling parameters
    group_size = 8  # Number of samples per input
    temperature = 1.0  # Temperature for sampling
    top_k = None  # Top-k sampling parameter
    
    # Reference model update frequency (in epochs)
    ref_update_freq = 0  # 0 means never update, fixed reference model
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k,v in kwargs.items():
            setattr(self, k, v)


class GRPOTrainer(Trainer):
    """GRPO Trainer for MolGPT
    
    Implements training with Group Relative Policy Optimization.
    """
    
    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos, reward_function=None):
        """Initialize GRPO Trainer
        
        Args:
            model: MolGPT model to train
            train_dataset: Training dataset
            test_dataset: Validation dataset
            config: GRPOTrainerConfig
            stoi: String to index vocabulary mapping
            itos: Index to string vocabulary mapping
            reward_function: Optional custom reward function (defaults to QED)
        """
        super().__init__(model, train_dataset, test_dataset, config, stoi, itos)
        
        # Create reference model (frozen copy of initial policy)
        self.ref_model = create_reference_model(model)
        
        # Initialize GRPO loss function
        self.grpo_loss = GRPOLoss(epsilon=config.epsilon, beta=config.beta)
        
        # Initialize reward function
        if reward_function is None:
            self.reward_fn = MolecularRewardFunction(reward_type=config.reward_type)
        else:
            self.reward_fn = reward_function
            
        # Regex pattern for tokenizing SMILES
        self.pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|\#|\-|\+|\\\\|\/|\:|\~|\@|\?|\>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
    
    def train_grpo(self, wandb=None):
        """Train the model using GRPO
        
        Args:
            wandb: Weights & Biases logger (optional)
            
        Returns:
            training metrics dictionary
        """
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            
            losses = []
            policy_losses = []
            kl_losses = []
            rewards = []
            
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"GRPO {split} epoch {epoch+1}")
            for it, (x, y, p, scaffold) in pbar:
                # Forward pass to get logits (no loss computation yet)
                x = x.to(self.device)
                y = y.to(self.device)
                if p is not None:
                    p = p.to(self.device)
                if scaffold is not None:
                    scaffold = scaffold.to(self.device)
                
                # For each batch, generate multiple molecule completions
                with torch.no_grad():
                    sampled_molecules = self._sample_molecules(x, p, scaffold, config.group_size)
                
                # Calculate rewards for sampled molecules
                batch_rewards = self._compute_rewards(sampled_molecules)
                rewards.append(torch.mean(batch_rewards).item())
                
                # Compute GRPO loss
                if is_train:
                    # Zero gradients
                    model.zero_grad()
                    
                    # Compute GRPO loss
                    loss, policy_loss, kl_loss = self.grpo_loss(
                        model, self.ref_model, x, y, p, scaffold, batch_rewards
                    )
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    
                    # Learning rate decay
                    if config.lr_decay:
                        self._adjust_learning_rate(optimizer, it + len(loader) * epoch, config)
                
                # Update metrics
                losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                kl_losses.append(kl_loss.item())
                
                # Update progress bar
                pbar.set_description(f"GRPO {split} epoch {epoch+1} | loss: {loss.item():.4f}")
                
                # Report to wandb
                if is_train and wandb is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/policy_loss": policy_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/mean_reward": torch.mean(batch_rewards).item(),
                        "train/lr": optimizer.param_groups[0]['lr']
                    })
            
            # Calculate epoch-level metrics
            mean_loss = np.mean(losses)
            mean_policy_loss = np.mean(policy_losses)
            mean_kl_loss = np.mean(kl_losses)
            mean_reward = np.mean(rewards)
            
            # Report epoch metrics
            if wandb is not None:
                wandb.log({
                    f"{split}/loss": mean_loss,
                    f"{split}/policy_loss": mean_policy_loss,
                    f"{split}/kl_loss": mean_kl_loss,
                    f"{split}/mean_reward": mean_reward
                })
            
            return {
                "loss": mean_loss,
                "policy_loss": mean_policy_loss,
                "kl_loss": mean_kl_loss,
                "reward": mean_reward
            }
        
        # Training loop
        best_loss = float('inf')
        metrics = {}
        
        for epoch in range(config.max_epochs):
            # Train
            train_metrics = run_epoch('train')
            metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            
            # Validate
            with torch.no_grad():
                test_metrics = run_epoch('test')
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Save checkpoint
            if test_metrics["loss"] < best_loss:
                best_loss = test_metrics["loss"]
                self.save_checkpoint()
                
            # Update reference model if needed
            if config.ref_update_freq > 0 and (epoch + 1) % config.ref_update_freq == 0:
                self.ref_model = create_reference_model(model)
                logger.info(f"Updated reference model at epoch {epoch+1}")
            
            # Generate samples
            if epoch % 5 == 0 or epoch == config.max_epochs - 1:
                self._generate_samples(wandb, prefix=f"epoch_{epoch+1}")
        
        return metrics
    
    def _sample_molecules(self, x, props, scaffolds, num_samples=8):
        """Generate multiple molecule completions per input
        
        Args:
            x: Input token indices (B, T)
            props: Conditioning properties (B, num_props)
            scaffolds: Conditioning scaffolds (B, scaffold_maxlen)
            num_samples: Number of samples per input
            
        Returns:
            sampled_molecules: List of SMILES strings for generated molecules
        """
        sampled_molecules = []
        batch_size = x.size(0)
        
        # For each input in the batch
        for i in range(batch_size):
            # Extract single input
            x_i = x[i:i+1].repeat(num_samples, 1)
            props_i = props[i:i+1].repeat(num_samples, 1) if props is not None else None
            scaffolds_i = scaffolds[i:i+1].repeat(num_samples, 1) if scaffolds is not None else None
            
            # Generate completions
            with torch.no_grad():
                completions = sample(
                    self.model, 
                    x_i, 
                    steps=48, 
                    temperature=self.config.temperature,
                    sample=True, 
                    top_k=self.config.top_k,
                    prop=props_i,
                    scaffold=scaffolds_i
                )
                
            # Convert to SMILES
            for gen_mol in completions:
                # Convert tokens to SMILES
                smiles = ''.join([self.itos[int(i)] for i in gen_mol])
                smiles = smiles.replace('<', '')  # Remove padding tokens
                
                # Canonicalize
                try:
                    mol = get_mol(smiles)
                    if mol:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        sampled_molecules.append(canonical_smiles)
                    else:
                        # Invalid molecule
                        sampled_molecules.append("")
                except:
                    # Error in processing
                    sampled_molecules.append("")
        
        return sampled_molecules
    
    def _compute_rewards(self, molecules):
        """Compute rewards for a list of molecules
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            rewards: Tensor of reward values (B*num_samples,)
        """
        # Use the reward function to score molecules
        rewards = self.reward_fn(molecules)
        return rewards
    
    def _adjust_learning_rate(self, optimizer, step, config):
        """Implement learning rate schedule
        
        Args:
            optimizer: PyTorch optimizer
            step: Current step
            config: Training configuration
        """
        if step < config.warmup_tokens:
            # Linear warmup
            lr_mult = float(step) / float(max(1, config.warmup_tokens))
        else:
            # Cosine learning rate decay
            progress = float(step - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Apply multiplier
        lr = config.learning_rate * lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def _generate_samples(self, wandb=None, prefix=""):
        """Generate and log sample molecules
        
        Args:
            wandb: Weights & Biases logger (optional)
            prefix: String prefix for sample names
        """
        model = self.model
        model.eval()
        
        # Define conditions (can be customized)
        conditions = [
            {"name": "default", "prop": None, "scaffold": None},
            {"name": "high_qed", "prop": torch.tensor([[0.9]]), "scaffold": None},
            {"name": "medium_qed", "prop": torch.tensor([[0.5]]), "scaffold": None}
        ]
        
        # Generate samples for each condition
        for cond in conditions:
            # Create starting context (just "C" for carbon)
            context = "C"
            x = torch.tensor(
                [self.stoi[s] for s in self.regex.findall(context)], 
                dtype=torch.long
            )[None,...].to(self.device)
            
            # Set property conditioning
            prop = cond["prop"].to(self.device) if cond["prop"] is not None else None
            
            # Set scaffold conditioning
            scaffold = cond["scaffold"] if cond["scaffold"] is not None else None
            
            # Generate
            with torch.no_grad():
                y = sample(
                    model, 
                    x, 
                    steps=48, 
                    temperature=1.0, 
                    sample=True, 
                    top_k=40,
                    prop=prop,
                    scaffold=scaffold
                )
            
            # Convert to SMILES and validate
            valid_mols = []
            valid_smiles = []
            
            for gen_mol in y:
                completion = ''.join([self.itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                
                # Check if molecule is valid
                mol = get_mol(completion)
                if mol:
                    valid_mols.append(mol)
                    valid_smiles.append(Chem.MolToSmiles(mol))
            
            # Log results
            condition_name = f"{prefix}_{cond['name']}" if prefix else cond['name']
            logger.info(f"Generated {len(valid_smiles)}/{len(y)} valid molecules for condition: {condition_name}")
            
            # Log to wandb if available
            if wandb is not None and valid_smiles:
                try:
                    # Create molecular image grid
                    from rdkit.Chem import Draw
                    img = Draw.MolsToGridImage(
                        valid_mols[:min(16, len(valid_mols))], 
                        molsPerRow=4,
                        subImgSize=(150, 150)
                    )
                    wandb.log({f"molecules/{condition_name}": wandb.Image(img)})
                except Exception as e:
                    logger.warning(f"Failed to create molecule grid: {e}")
                
                # Log SMILES strings
                wandb.log({f"smiles/{condition_name}": wandb.Table(
                    columns=["SMILES"], 
                    data=[[s] for s in valid_smiles[:10]]
                )}) 