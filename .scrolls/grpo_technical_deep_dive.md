# 🥷 Ninja Scroll: GRPO Technical Deep Dive 🥷

## Introduction

This ninja scroll provides a technical deep dive into the Group Relative Policy Optimization (GRPO) implementation in MolGPT. It explains the mathematical foundations, algorithmic details, and implementation specifics of GRPO for molecular generation.

## Mathematical Foundations of GRPO

### Policy Optimization Objective

GRPO is based on the policy gradient framework, where the objective is to maximize the expected reward:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[R]$$

where $\pi_\theta$ is the policy parameterized by $\theta$, and $R$ is the reward.

### Clipped Surrogate Objective

GRPO uses a clipped surrogate objective similar to Proximal Policy Optimization (PPO):

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the current policy and the reference policy
- $A_t$ is the advantage estimate at timestep $t$
- $\epsilon$ is the clipping parameter (typically 0.2)

### KL Divergence Penalty

GRPO adds a KL divergence penalty to ensure the policy doesn't deviate too far from the reference policy:

$$L^{KL}(\theta) = \beta \cdot D_{KL}(\pi_{\theta_{old}} || \pi_\theta)$$

where $\beta$ is the KL penalty coefficient.

### Group Advantage

GRPO extends standard policy optimization by computing advantages within groups of samples:

$$A_i^G = R_i - \frac{1}{|G|} \sum_{j \in G} R_j$$

where $G$ is a group of samples generated from the same context, and $R_i$ is the reward for sample $i$.

## GRPO Algorithm for MolGPT

### Algorithm Overview

1. **Initialize**: Start with a pre-trained MolGPT model as the policy model $\pi_\theta$
2. **Create Reference Model**: Create a frozen copy of the policy model as the reference model $\pi_{\theta_{old}}$
3. **For each training iteration**:
   a. Sample groups of molecules from the policy model
   b. Compute rewards for each molecule
   c. Compute advantages within each group
   d. Compute the GRPO loss (clipped surrogate objective + KL penalty)
   e. Update the policy model parameters
   f. Optionally update the reference model periodically

### Pseudocode

```
Initialize policy model π_θ from pre-trained MolGPT
Create reference model π_θ_old = π_θ
Freeze parameters of π_θ_old

For each epoch:
    For each batch:
        # Sample groups of molecules
        For each input context x_i:
            Generate k molecules {y_i1, y_i2, ..., y_ik} from π_θ(y|x_i)
        
        # Compute rewards
        For each molecule y_ij:
            Compute reward R_ij = reward_fn(y_ij)
        
        # Compute advantages
        For each group i:
            Compute mean reward R̄_i = (1/k) * Σ_j R_ij
            For each molecule j in group i:
                Compute advantage A_ij = R_ij - R̄_i
        
        # Compute probability ratios
        Compute log probs log π_θ(y|x) and log π_θ_old(y|x)
        Compute ratio r = exp(log π_θ(y|x) - log π_θ_old(y|x))
        
        # Compute GRPO loss
        L_clip = min(r * A, clip(r, 1-ε, 1+ε) * A)
        L_kl = KL(π_θ_old || π_θ)
        L = -L_clip + β * L_kl
        
        # Update policy
        Compute gradients ∇_θ L
        Update θ using optimizer
    
    # Optionally update reference model
    If epoch % update_freq == 0:
        π_θ_old = π_θ
        Freeze parameters of π_θ_old
```

## Implementation Details in MolGPT

### Key Components

1. **GRPOLoss Class** (`train/grpo_loss.py`):
   - Implements the clipped surrogate objective and KL divergence penalty
   - Handles token-level probability calculations

2. **GRPOTrainer Class** (`train/grpo_trainer.py`):
   - Manages the GRPO training loop
   - Handles molecule sampling and reward calculation
   - Implements reference model updates

3. **Reward Functions** (`train/rewards.py`):
   - Implements various molecular property reward functions
   - Handles batch processing of molecules

### Token-Level Probability Calculation

In the GPT model's forward method, token-level log probabilities are calculated as follows:

```python
# Calculate log probabilities
log_probs = F.log_softmax(logits, dim=-1)

# Gather log probs for actual tokens
token_log_probs = torch.gather(
    log_probs, 
    dim=-1, 
    index=targets.unsqueeze(-1)
).squeeze(-1)
```

### Importance Ratio Calculation

The probability ratio between the policy and reference models is calculated as:

```python
# Calculate importance ratio
ratio = torch.exp(policy_log_probs - ref_log_probs)
```

### Clipped Surrogate Objective

The clipped surrogate objective is implemented as:

```python
# Clipped surrogate objective
surrogate1 = ratio * advantages
surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
policy_loss = -torch.min(surrogate1, surrogate2).mean()
```

### KL Divergence Penalty

The KL divergence penalty is implemented as:

```python
# KL divergence penalty
kl_div = (ref_log_probs.exp() * (ref_log_probs - policy_log_probs)).sum(dim=-1).mean()
```

## Advanced Implementation Aspects

### Reference Model Management

The reference model is created by copying the policy model and freezing its parameters:

```python
def create_reference_model(model):
    """Create a frozen copy of the current model to serve as reference"""
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model
```

### Group-Based Sampling

The `_sample_molecules` method in `GRPOTrainer` handles group-based sampling:

```python
def _sample_molecules(self, x, props, scaffolds, num_samples=8):
    """Generate multiple molecule completions per input"""
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
```

### Reward Normalization

Rewards are normalized within each batch to improve training stability:

```python
def _normalize_rewards(self, rewards, eps=1e-8):
    """Normalize rewards within the batch for relative advantages"""
    mean = rewards.mean()
    std = rewards.std() + eps
    return (rewards - mean) / std
```

## Performance Considerations

### Computational Efficiency

GRPO is more computationally intensive than standard supervised learning due to:

1. **Multiple Sampling**: Generating multiple molecules per input
2. **Reward Calculation**: Computing rewards for each molecule
3. **Dual Model Evaluation**: Forward passes through both policy and reference models

To improve efficiency:

- Use smaller batch sizes
- Reduce the number of samples per input
- Use simpler reward functions when possible

### Memory Optimization

To reduce memory usage:

- Use gradient accumulation for larger effective batch sizes
- Implement reward calculation in batches
- Use mixed precision training

## Hyperparameter Tuning

### Critical Hyperparameters

1. **Epsilon (ε)**: Controls the clipping range for the surrogate objective
   - Smaller values (e.g., 0.1) lead to more conservative updates
   - Larger values (e.g., 0.3) allow more aggressive policy changes

2. **Beta (β)**: Controls the weight of the KL divergence penalty
   - Smaller values prioritize reward optimization
   - Larger values prioritize staying close to the reference model

3. **Group Size**: Number of samples per input for group advantage calculation
   - Larger groups provide better advantage estimates but require more computation
   - Typical values range from 4 to 16

4. **Learning Rate**: Should be smaller than for supervised learning
   - Typical values range from 1e-5 to 5e-5

### Recommended Hyperparameter Ranges

| Hyperparameter | Range | Default |
|----------------|-------|---------|
| Epsilon (ε) | 0.1 - 0.3 | 0.2 |
| Beta (β) | 0.001 - 0.1 | 0.01 |
| Group Size | 4 - 16 | 8 |
| Learning Rate | 1e-6 - 1e-4 | 1e-5 |
| Reference Update Frequency | 0 - 5 epochs | 0 (never) |

## Conclusion

GRPO provides a powerful framework for fine-tuning MolGPT models towards specific molecular properties. The implementation in MolGPT follows the mathematical foundations of policy optimization while adding group-based advantages for more effective molecular generation.

The key to successful GRPO fine-tuning is balancing between optimizing for rewards and staying close to the reference model. This balance is controlled by the epsilon and beta hyperparameters, which should be tuned based on the specific application and desired properties.