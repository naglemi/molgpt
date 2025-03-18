# 🥷 Ninja Scroll: GRPO Usage Guide for MolGPT 🥷

## Introduction

This ninja scroll provides a comprehensive guide on how to use Group Relative Policy Optimization (GRPO) with MolGPT for fine-tuning molecular generation models towards specific properties. GRPO is a reinforcement learning technique that allows the model to optimize for desired molecular properties while maintaining proximity to a reference model.

## Prerequisites

Before using GRPO with MolGPT, ensure you have:

1. A pre-trained MolGPT model (trained using the standard training script)
2. A dataset of molecules for fine-tuning
3. A clear objective for what molecular properties you want to optimize

## GRPO Workflow

The GRPO workflow consists of the following steps:

1. **Pre-train a base MolGPT model** using standard supervised learning
2. **Create a reference model** by copying the pre-trained model
3. **Define reward functions** based on desired molecular properties
4. **Fine-tune the model using GRPO** to optimize for the desired properties
5. **Generate molecules** with the fine-tuned model

## Step-by-Step Guide

### 1. Pre-train a Base MolGPT Model

First, train a base MolGPT model using the standard training script:

```bash
python train/train.py \
    --run_name base_model \
    --data_name moses2 \
    --batch_size 384 \
    --max_epochs 10 \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 256
```

This will create a model checkpoint at `weights/base_model.pt`.

### 2. Fine-tune with GRPO

Use the `train_grpo.py` script to fine-tune the pre-trained model using GRPO:

```bash
python train_grpo.py \
    --model_path weights/base_model.pt \
    --reward_type qed \
    --max_epochs 5 \
    --batch_size 32 \
    --lr 1e-5 \
    --epsilon 0.2 \
    --beta 0.01 \
    --output_dir ./grpo_models
```

#### Key GRPO Parameters:

- `--reward_type`: Type of molecular reward function to use (`qed`, `logp`, `combined`)
- `--epsilon`: Clipping parameter for the surrogate objective (default: 0.2)
- `--beta`: KL penalty coefficient to keep the policy close to the reference model (default: 0.01)
- `--group_size`: Number of samples per input for group advantage calculation (default: 8)

### 3. Advanced GRPO Configuration

For more advanced use cases, you can customize the GRPO training process:

#### Custom Reward Functions

You can implement custom reward functions by extending the `MolecularRewardFunction` class in `train/grpo_loss.py` or the reward classes in `train/rewards.py`:

```python
from train.rewards import PropertyReward

class CustomReward(PropertyReward):
    def __init__(self, weight=1.0):
        super().__init__(property_type='custom', weight=weight)
    
    def __call__(self, smiles_list):
        rewards = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                rewards.append(0.0)
                continue
            
            # Calculate custom reward
            # Example: combine QED and synthetic accessibility
            qed_score = QED.qed(mol)
            sa_score = -sascorer.calculateScore(mol)  # Lower is better, so negate
            reward = 0.7 * qed_score + 0.3 * sa_score
            
            rewards.append(float(reward))
        
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) * self.weight
```

#### Multi-Objective Optimization

To optimize for multiple properties simultaneously, use the `combined` reward type or create a custom reward function that combines multiple properties:

```bash
python train_grpo.py \
    --model_path weights/base_model.pt \
    --reward_type combined \
    --max_epochs 5
```

#### Reference Model Update Frequency

By default, the reference model is fixed throughout training. To periodically update the reference model, use the `--ref_update_freq` parameter:

```bash
python train_grpo.py \
    --model_path weights/base_model.pt \
    --reward_type qed \
    --ref_update_freq 1  # Update reference model every epoch
```

### 4. Generating Molecules with Fine-tuned Models

After fine-tuning, you can generate molecules with the fine-tuned model:

```bash
python train_grpo.py \
    --model_path grpo_models/molgpt_grpo_qed.pt \
    --generate_only \
    --num_samples 100 \
    --output_file generated_molecules.csv
```

## Best Practices

1. **Start with a well-trained base model**: The quality of the base model significantly impacts the performance of GRPO fine-tuning.

2. **Use appropriate reward functions**: Choose reward functions that align with your desired molecular properties.

3. **Balance exploration and exploitation**: Adjust the `epsilon` and `beta` parameters to balance between optimizing for rewards and staying close to the reference model.

4. **Monitor training metrics**: Keep an eye on the policy loss, KL divergence, and mean rewards during training.

5. **Validate generated molecules**: Always validate the generated molecules for chemical validity and desired properties.

## Troubleshooting

### Common Issues and Solutions

1. **High KL divergence**: If the KL divergence is too high, the model is deviating too much from the reference model. Increase the `beta` parameter.

2. **Low rewards**: If the rewards are too low, the model might not be learning to optimize for the desired properties. Try decreasing the `beta` parameter or increasing the `epsilon` parameter.

3. **Invalid molecules**: If the model generates many invalid molecules, it might be overfitting to the reward function. Decrease the learning rate or increase the `beta` parameter.

4. **Memory issues**: If you encounter memory issues, reduce the batch size or group size.

## Conclusion

GRPO is a powerful technique for fine-tuning MolGPT models towards specific molecular properties. By following this guide, you can effectively use GRPO to generate molecules with desired properties while maintaining the quality of the base model.

Remember that the key to successful GRPO fine-tuning is finding the right balance between optimizing for rewards and staying close to the reference model. Experiment with different reward functions and hyperparameters to find the optimal configuration for your specific use case.