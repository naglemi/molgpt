# MolGPT with GRPO (Group Relative Policy Optimization)

This extension to MolGPT implements Group Relative Policy Optimization (GRPO), an advanced reinforcement learning technique for fine-tuning generative models toward specific molecular properties while maintaining distribution quality.

## Overview

The GRPO implementation allows for:

1. **Property-Targeted Generation**: Fine-tune MolGPT to generate molecules with enhanced specific properties (e.g., drug-likeness, synthetic accessibility)
2. **Controlled Optimization**: Balance between optimization and preserving the learned distribution
3. **Token-Level Probability Tracking**: Enable precise policy gradient optimization at the token level
4. **Multi-Property Targeting**: Support for various molecular reward functions

## Components

The implementation consists of three main components:

1. **Token-Level Probability Tracking**: Modified the `GPT` model to track token-level probabilities required for GRPO.
2. **GRPO Loss Function**: Implemented the core GRPO algorithm with clipping, advantages, and KL-divergence.
3. **GRPO Trainer**: Created a specialized trainer that extends the standard MolGPT trainer with RL capabilities.

## Usage

### Basic Usage

```bash
python train_grpo.py --model_path /path/to/pretrained/model.pt --reward_type qed --batch_size 32 --max_epochs 10
```

### Advanced Usage

```bash
python train_grpo.py \
    --model_path /path/to/pretrained/model.pt \
    --data_path moses2.csv \
    --vocab_path moses2_stoi.json \
    --reward_type combined \
    --beta 0.02 \
    --epsilon 0.15 \
    --group_size 16 \
    --max_epochs 20 \
    --batch_size 32 \
    --lr 2e-5 \
    --output_dir ./grpo_results \
    --use_wandb
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to pre-trained model checkpoint | (Required) |
| `--data_path` | Path to dataset CSV | `moses2.csv` |
| `--vocab_path` | Path to vocabulary JSON | `moses2_stoi.json` |
| `--reward_type` | Type of molecular reward (`qed`, `logp`, `combined`) | `qed` |
| `--epsilon` | GRPO clipping parameter | `0.2` |
| `--beta` | KL penalty coefficient | `0.01` |
| `--group_size` | Number of samples per input | `8` |
| `--batch_size` | Batch size | `32` |
| `--max_epochs` | Maximum training epochs | `10` |
| `--lr` | Learning rate | `1e-5` |
| `--output_dir` | Output directory | `./grpo_models` |
| `--use_wandb` | Enable Weights & Biases logging | (Flag) |

## Reward Functions

The implementation supports several reward functions:

- **QED (Quantitative Estimate of Drug-likeness)**: Scores the drug-likeness of molecules
- **LogP**: Targets optimal water-octanol partition coefficient
- **Combined**: Weighted combination of multiple properties

You can extend with custom reward functions by modifying `MolecularRewardFunction` in `train/grpo_loss.py`.

## Implementation Details

### Token-Level Probability Tracking

The GPT model was modified to track token-level probabilities:

```python
# Extract token-level log probabilities
log_probs = F.log_softmax(logits, dim=-1)
token_log_probs = torch.gather(
    log_probs, 
    dim=-1, 
    index=targets.unsqueeze(-1)
).squeeze(-1)
```

### GRPO Loss Calculation

The core GRPO algorithm computes importance-weighted advantages with clipping:

```python
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
```

## Theoretical Background

GRPO operates by:

1. Computing relative advantages within groups of samples
2. Using a PPO-like clipped surrogate objective
3. Adding a KL divergence term to prevent large policy distribution shifts

This approach allows for effective optimization without requiring a separate value network, making it efficient for fine-tuning generative models.

## References

- DeepSeek-Coder: "Let Code Work Like Human Programmers" (introducing GRPO)
- TRL (Transformer Reinforcement Learning) library implementation of GRPO
- PPO (Proximal Policy Optimization) algorithm 