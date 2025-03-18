"""
GRPO Loss implementation for MolGPT

This module implements Group Relative Policy Optimization (GRPO) for MolGPT,
allowing fine-tuning of the model towards specific molecular properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen
import math

class GRPOLoss(nn.Module):
    """GRPO Loss for MolGPT
    
    Implements the Group Relative Policy Optimization (GRPO) loss function.
    This enables training a policy model based on molecule generation rewards
    while maintaining proximity to a reference model.
    """
    
    def __init__(self, epsilon=0.2, beta=0.01):
        """Initialize GRPO loss
        
        Args:
            epsilon: Clipping parameter for the surrogate objective
            beta: KL penalty coefficient for the reference model
        """
        super().__init__()
        self.epsilon = epsilon  # Clipping parameter
        self.beta = beta  # KL penalty coefficient
    
    def forward(self, policy_model, ref_model, inputs, targets, props=None, scaffolds=None, rewards=None):
        """Compute GRPO loss
        
        Args:
            policy_model: Current MolGPT policy model
            ref_model: Reference MolGPT model (frozen copy of pre-trained model)
            inputs: Token inputs (B, T)
            targets: Target tokens (B, T)
            props: Conditioning properties (B, num_props)
            scaffolds: Conditioning scaffolds (B, scaffold_maxlen)
            rewards: Pre-computed rewards for each sequence (B,)
        
        Returns:
            loss: Total GRPO loss
            policy_loss: Policy gradient loss component
            kl_loss: KL divergence loss component
        """
        # Get log probs from policy model
        _, _, _, _, policy_token_log_probs = policy_model(
            inputs, targets, props, scaffolds, return_log_probs=True
        )
        
        # Get log probs from reference model (no gradients)
        with torch.no_grad():
            _, _, _, _, ref_token_log_probs = ref_model(
                inputs, targets, props, scaffolds, return_log_probs=True
            )
        
        # Check if we need to normalize rewards within batch
        if rewards is not None:
            advantages = self._normalize_rewards(rewards)
        else:
            # If no rewards provided, assume uniform advantage of 1.0
            advantages = torch.ones_like(policy_token_log_probs[:, 0])
        
        # Reshape advantages to match token dimensions if needed
        advantages = advantages.view(-1, 1).expand_as(policy_token_log_probs)
        
        # Calculate importance ratio
        ratio = torch.exp(policy_token_log_probs - ref_token_log_probs)
        
        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # KL divergence penalty
        kl_div = (ref_token_log_probs.exp() * (ref_token_log_probs - policy_token_log_probs)).mean()
        
        # Total loss
        loss = policy_loss + self.beta * kl_div
        
        return loss, policy_loss, kl_div
    
    def _normalize_rewards(self, rewards, eps=1e-8):
        """Normalize rewards within the batch for relative advantages
        
        Args:
            rewards: Reward values (B,)
            eps: Small epsilon for numerical stability
            
        Returns:
            normalized_rewards: Batch-normalized rewards
        """
        mean = rewards.mean()
        std = rewards.std() + eps
        return (rewards - mean) / std


class MolecularRewardFunction:
    """Reward function for molecular generation
    
    Calculates rewards for generated molecules based on various properties.
    """
    
    def __init__(self, reward_type='qed', weight=1.0):
        """Initialize reward function
        
        Args:
            reward_type: Type of reward ('qed', 'logp', 'combined', etc.)
            weight: Weight for this reward component
        """
        self.reward_type = reward_type
        self.weight = weight
    
    def __call__(self, smiles_list):
        """Calculate rewards for a list of SMILES strings
        
        Args:
            smiles_list: List of generated SMILES strings
            
        Returns:
            rewards: Tensor of reward values (batch_size,)
        """
        rewards = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Invalid molecule
                rewards.append(0.0)
                continue
            
            # Calculate reward based on specified property
            if self.reward_type == 'qed':
                # Drug-likeness score (0-1, higher is better)
                reward = QED.qed(mol)
            elif self.reward_type == 'logp':
                # Water-octanol partition coefficient (targeting 2.0-3.0 range)
                logp = Crippen.MolLogP(mol)
                # Penalize values outside desired range
                reward = 1.0 - min(abs(logp - 2.5), 2.0) / 2.0
            elif self.reward_type == 'combined':
                # Combined reward considering multiple properties
                qed_score = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                logp_score = 1.0 - min(abs(logp - 2.5), 2.0) / 2.0
                # Weighted combination
                reward = 0.7 * qed_score + 0.3 * logp_score
            else:
                # Default to QED
                reward = QED.qed(mol)
            
            rewards.append(float(reward))
        
        # Convert to tensor
        return torch.tensor(rewards, dtype=torch.float32) * self.weight


def create_reference_model(model):
    """Create a frozen copy of the current model to serve as reference
    
    Args:
        model: Current MolGPT model instance
        
    Returns:
        ref_model: Frozen copy of model
    """
    # Create deep copy
    ref_model = torch.nn.modules.module._load_from_state_dict(
        model.state_dict(), 
        model.__class__, 
        model.config
    )
    
    # Freeze parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Set to evaluation mode
    ref_model.eval()
    
    return ref_model 