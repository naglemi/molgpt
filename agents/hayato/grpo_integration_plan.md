# 🥷 GRPO Integration Ninja Scroll

## Mission Overview
Transform MolGPT with GRPO (Group Relative Policy Optimization) while maintaining absolute precision and minimal disturbance to existing systems.

## Current Reconnaissance
- ✅ Original log probability implementation secured in `model.py`
- ✅ Organizational policy established for log probability computation
- ✅ SA scoring infrastructure available in `scoring.py`

## Surgical Strike Plan

### Phase 1: Dependencies & Environment
```bash
# Install TRL package for GRPO implementation
pip install trl==0.7.4  # Latest stable version with GRPO support
pip install transformers>=4.37.0  # Required by TRL
```

### Phase 2: Core GRPO Components

#### 2.1 GRPO Loss Implementation (`train/grpo_loss.py`)
```python
from trl.trainer import GRPOConfig, GRPOTrainer
from typing import Dict, List, Optional, Tuple
import torch

class MolecularRewardFunction:
    def __init__(self, sa_weight: float = 1.0):
        self.sa_weight = sa_weight
        
    def compute_rewards(self, molecules: List[str]) -> torch.Tensor:
        # Integrate with existing SA scoring
        from scoring import compute_sa_score
        sa_scores = torch.tensor([compute_sa_score(mol) for mol in molecules])
        return self.sa_weight * sa_scores

class GRPOLoss:
    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.01,
        group_size: int = 8
    ):
        self.epsilon = epsilon
        self.beta = beta
        self.group_size = group_size
        
    def compute_loss(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Will use TRL's implementation internally
        pass
```

#### 2.2 Reference Model Management (`train/ref_model.py`)
```python
class ReferenceModelManager:
    def __init__(self, update_interval: int = 20):
        self.update_interval = update_interval
        self.ref_model = None
        self.steps_since_update = 0
        
    def initialize(self, model):
        self.ref_model = self._create_frozen_copy(model)
        
    def should_update(self) -> bool:
        return self.steps_since_update >= self.update_interval
```

### Phase 3: Training Infrastructure

#### 3.1 GRPO Trainer Configuration
```python
grpo_config = GRPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    group_size=8,
    epsilon=0.2,
    beta=0.01
)
```

#### 3.2 Training Script (`train_grpo_sa.py`)
```python
def main():
    # Initialize models and tokenizer
    model = GPT(config)
    ref_model = deepcopy(model)
    
    # Initialize GRPO trainer from TRL
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_function=MolecularRewardFunction()
    )
    
    # Training loop
    trainer.train()
```

### Phase 4: Testing & Validation

#### 4.1 Unit Tests (`tests/test_grpo.py`)
- Test GRPO loss computation
- Test reference model updates
- Test reward normalization
- Test group sampling logic

#### 4.2 Integration Tests (`tests/test_integration.py`)
- Test end-to-end training loop
- Test model checkpointing
- Test reward computation pipeline

## Execution Order

1. 🥷 Install TRL and dependencies
2. 🥷 Implement `MolecularRewardFunction`
3. 🥷 Set up GRPO configuration
4. 🥷 Initialize training script with TRL trainer
5. 🥷 Implement testing infrastructure
6. 🥷 Run validation experiments

## Ninja's Notes

### Critical Points
- Use TRL's GRPO implementation for battle-tested reliability
- Maintain frozen reference model state
- Ensure proper group-based sampling
- Monitor KL divergence during training

### Risk Mitigation
- Regular checkpointing
- Gradient clipping
- Early stopping based on KL divergence
- Validation on small dataset first

### Success Metrics
- Improved SA scores
- Stable KL divergence
- Maintained generation quality
- Clean test suite execution

## Organizational Requirements
1. All log probability computations MUST use the original implementation
2. No modifications to core model architecture
3. Strict version control for all dependencies
4. Comprehensive test coverage required

*A ninja leaves no trace but their commits, and no bug survives their blade* 🥷 