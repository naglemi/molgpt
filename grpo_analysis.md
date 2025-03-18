# 🥷 Ninja Scroll: GRPO Analysis 🥷

## GRPO (Group Relative Policy Optimization) Analysis

GRPO is an advanced reinforcement learning technique developed by DeepSeek for optimizing language models. This scroll analyzes the GRPO implementation found in the finetune_midi codebase and DeepSeek's approach.

### Theoretical Foundation

From the DeepSeek paper, GRPO operates on the following core principles:

1. **Advantage Calculation via Groups**: GRPO avoids learning a separate value model by computing advantages based on a group of similar outputs for each input.

2. **Relative Advantage Computation**: For each group of outputs, rewards are normalized by the group's mean and standard deviation to calculate relative advantages.

3. **Surrogate Objective**: GRPO optimizes a PPO-like clipped surrogate objective that promotes beneficial policy updates while limiting extreme changes:

   ```
   J_GRPO(θ) = E[min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage) - β * KL(π_θ||π_ref)]
   ```
   
   Where:
   - `ratio` is the ratio of current policy probability to old policy probability
   - `advantage` is the computed relative advantage within a group
   - `β` is the KL penalty coefficient
   - `KL(π_θ||π_ref)` is the KL divergence from the reference policy to the current policy

### Implementation in finetune_midi

The finetune_midi codebase implements GRPO in `loss_modules/grpo.py` with the following key components:

1. **Core Function**: `compute_grpo_loss` calculates the GRPO loss by:
   - Sampling noisy data at different timesteps
   - Running both policy and reference models
   - Computing the loss differences via `compute_v_Lt`
   - Calculating advantages from external scores
   - Applying the GRPO clipping algorithm

2. **Loss Difference Calculation**: `compute_v_Lt` determines the difference in loss between the policy and reference models:
   ```python
   # Simplified from grpo.py
   kl_per_node = categorical_kl(ref_log_posterior, policy_log_posterior)
   loss_difference = kl_per_node.sum(dim=1)
   ```

3. **Advantage Computation**: GRPO normalizes scores within a batch to compute relative advantages:
   ```python
   # Simplified from grpo.py
   def normalize_batch_rewards(scores, eps: float = 1e-8) -> torch.Tensor:
       mean = scores.mean()
       std = scores.std() + eps
       return (scores - mean) / std
   ```

4. **Clipped Surrogate Loss**:
   ```python
   # Simplified from grpo.py
   ratio = torch.exp(policy_log_probs - ref_log_probs)
   surrogate1 = ratio * advantage_term 
   surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage_term
   surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
   ```

### Key Differences from Standard PPO

1. **No Value Network**: Unlike PPO, GRPO eliminates the need for a separate value function to estimate advantages, reducing computational overhead.

2. **Group-Based Advantages**: GRPO normalizes rewards within batches/groups of outputs for the same input, leveraging the comparative nature of rewards.

3. **Fixed Reference Model**: GRPO maintains a fixed reference model throughout training iterations, serving as an anchor to prevent policy drift.

### TRL Implementation Insights

The Hugging Face TRL library implements GRPO for language models with the following notable features:

1. **Training Flow**:
   - Sample completions from the current policy
   - Score completions using reward models
   - Normalize rewards within each group
   - Update policy to maximize the GRPO objective

2. **Multi-Reward Support**:
   ```python
   # From TRL GRPOTrainer
   reward_funcs = Union[RewardFunc, list[RewardFunc]]
   ```
   
   This allows combining multiple reward functions for more nuanced optimization.

3. **Chat Template Integration**: Supports conversational models through proper chat template handling.

## Applicability to Diffusion Models

The GRPO implementation in finetune_midi is already adapted for:

1. **Categorical Distributions**: Using categorical KL divergence for discrete variables common in diffusion models.

2. **Timestep Handling**: Supporting diffusion model timestep mechanics with appropriate noise levels.

3. **Multi-Property Tensors**: Handling tensor shapes needed for molecular structures, including:
   - Node features [B, N, C]
   - Edge features [B, N, N, C]
  
This demonstrates GRPO's adaptability beyond language models to structured generative models like those in molecular diffusion. 