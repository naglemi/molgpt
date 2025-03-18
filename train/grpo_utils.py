"""
GRPO Utilities for MolGPT

This module implements utility functions and classes for Group Relative Policy 
Optimization (GRPO), including PPO-style loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOLoss(nn.Module):
    """PPO-style loss function for GRPO
    
    Implements a PPO-style loss function for GRPO with clipped surrogate objective,
    value function loss, entropy bonus, and KL divergence penalty.
    """
    
    def __init__(self, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01, kl_coef=0.2):
        """Initialize PPO loss
        
        Args:
            clip_param: Clipping parameter for the surrogate objective
            value_loss_coef: Coefficient for the value function loss
            entropy_coef: Coefficient for the entropy bonus 
            kl_coef: Coefficient for the KL divergence penalty
        """
        super().__init__()
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
    
    def forward(self, policy_logits, ref_logits, tokens, rewards):
        """Compute PPO loss
        
        Args:
            policy_logits: Logits from policy model (B, T, vocab_size)
            ref_logits: Logits from reference model (B, T, vocab_size)
            tokens: Token inputs (B, T)
            rewards: Pre-computed rewards for each sequence (B, 1)
            
        Returns:
            policy_loss: Policy gradient loss component
            value_loss: Value function loss component
            entropy_loss: Entropy bonus component
            kl_loss: KL divergence loss component
        """
        # Ensure proper dimensions
        batch_size, seq_len, vocab_size = policy_logits.size()
        
        # Compute token log probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Get log probs for actual tokens
        token_policy_log_probs = torch.gather(
            policy_log_probs, 
            dim=-1, 
            index=tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        token_ref_log_probs = torch.gather(
            ref_log_probs, 
            dim=-1, 
            index=tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate importance ratio
        ratio = torch.exp(token_policy_log_probs - token_ref_log_probs)
        
        # Use rewards as advantages directly (simplified)
        # In a more complex implementation, we'd compute advantages using value predictions
        advantages = rewards.expand_as(ratio)
        
        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value function loss (simplified, using zero as baseline)
        # In a complete implementation, this would use predictions from a value head
        value_loss = torch.zeros_like(policy_loss)
        
        # Entropy bonus (encourage exploration)
        policy_probs = torch.exp(policy_log_probs)
        entropy = -(policy_probs * policy_log_probs).sum(dim=-1).mean()
        entropy_loss = entropy
        
        # KL divergence penalty
        kl_div = (ref_log_probs.exp() * (ref_log_probs - policy_log_probs)).sum(dim=-1).mean()
        kl_loss = kl_div
        
        return policy_loss, value_loss, entropy_loss, kl_loss


def create_reference_model(model):
    """Create a copy of the current model to serve as reference
    
    Args:
        model: Current model instance
        
    Returns:
        ref_model: Copy of model with frozen parameters
    """
    # Create a new instance with the same parameters
    ref_model = type(model)(**model.config)
    
    # Copy parameters from original model
    ref_model.load_state_dict(model.state_dict())
    
    # Freeze parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Set to evaluation mode
    ref_model.eval()
    
    return ref_model


def compute_normalized_rewards(rewards, eps=1e-8):
    """Normalize rewards within the batch
    
    Args:
        rewards: Reward values (B,)
        eps: Small epsilon for numerical stability
        
    Returns:
        normalized_rewards: Batch-normalized rewards
    """
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(-1)
        
    mean = rewards.mean()
    std = rewards.std() + eps
    return (rewards - mean) / std 