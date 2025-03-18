# 🥷 Ninja Scroll: Integrating GRPO with MolGPT 🥷

## Viability Assessment for GRPO Integration

This scroll analyzes the viability of replacing MolGPT's standard cross-entropy loss with GRPO for improved molecular generation. The ninja presents an incisive assessment of the challenges, required modifications, and potential benefits.

### Current Architecture vs. GRPO Requirements

| MolGPT Component | Current Status | GRPO Requirement | Compatibility |
|------------------|----------------|------------------|---------------|
| Loss Function | Simple cross-entropy | Clipped surrogate objective | 🟠 Requires replacement |
| Model Structure | Single GPT model | Policy + Reference models | 🟠 Requires duplication |
| Training Loop | Standard supervised | Group-based RL | 🔴 Major refactoring needed |
| Reward Signal | None (direct supervision) | External reward models | 🔴 Must be implemented |
| Tokenization | Character-level for SMILES | Unchanged | 🟢 Compatible |
| Conditioning | Property & scaffold embedding | Can be preserved | 🟢 Compatible |

### Critical Implementation Challenges

1. **Reference Model Creation**:
   ```python
   # Required addition:
   def create_reference_model(model):
       """Create a frozen copy of the current model to serve as reference"""
       ref_model = copy.deepcopy(model)
       for param in ref_model.parameters():
           param.requires_grad = False
       return ref_model
   ```

2. **Logit Extraction and Probability Calculation**:
   - MolGPT currently returns only logits and loss
   - GRPO requires token-level probabilities for ratio calculation
   ```python
   # Required modification in GPT.forward():
   log_probs = F.log_softmax(logits, dim=-1)
   token_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
   ```

3. **Reward Function Implementation**:
   - No existing reward mechanism in MolGPT
   - Need to implement molecular property scoring functions:
   ```python
   def compute_molecular_rewards(molecules):
       """Calculate rewards based on desired molecular properties"""
       rewards = []
       for mol in molecules:
           # Example: reward for drug-likeness + synthetic accessibility
           qed_score = QED.qed(mol)
           sa_score = -sascorer.calculateScore(mol)  # Lower is better, so negate
           reward = qed_score * 0.7 + sa_score * 0.3  # Weighted combination
           rewards.append(reward)
       return torch.tensor(rewards)
   ```

4. **Group-Based Sampling**:
   - Current training uses direct supervision
   - GRPO requires multiple samples per input for group advantages:
   ```python
   # Required sampling loop:
   def sample_multiple_completions(model, prompts, num_samples=8):
       """Generate multiple molecule completions per prompt"""
       all_completions = []
       for prompt in prompts:
           group_completions = []
           for _ in range(num_samples):
               completion = generate_with_model(model, prompt)
               group_completions.append(completion)
           all_completions.append(group_completions)
       return all_completions
   ```

### Integration Approach

The most surgical approach for integrating GRPO would involve:

1. **Trainer Extension**:
   - Extend the current `Trainer` class to a `GRPOTrainer` class 
   - Keep the original trainer intact for backward compatibility

2. **Loss Function Encapsulation**:
   ```python
   class GRPOLoss(nn.Module):
       """Encapsulated GRPO loss calculation"""
       def __init__(self, epsilon=0.2, beta=0.01):
           super().__init__()
           self.epsilon = epsilon  # Clipping parameter
           self.beta = beta  # KL penalty coefficient
           
       def forward(self, policy_model, ref_model, batch, scores=None):
           # 1. Get logits and log probs from both models
           policy_logits, _, _ = policy_model(batch['x'], batch['y'], batch['props'], batch['scaffold'])
           with torch.no_grad():
               ref_logits, _, _ = ref_model(batch['x'], batch['y'], batch['props'], batch['scaffold'])
           
           # 2. Calculate log probabilities
           policy_log_probs = self._get_token_log_probs(policy_logits, batch['y'])
           ref_log_probs = self._get_token_log_probs(ref_logits, batch['y'])
           
           # 3. Compute advantages from scores
           advantages = self._compute_advantages(scores)
           
           # 4. GRPO loss calculation
           ratio = torch.exp(policy_log_probs - ref_log_probs)
           surrogate1 = ratio * advantages
           surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
           surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
           
           # 5. Add KL penalty
           kl_div = (ref_log_probs.exp() * (ref_log_probs - policy_log_probs)).sum(-1).mean()
           
           return surrogate_loss + self.beta * kl_div
   ```

3. **Training Loop Modification**:
   ```python
   # Simplified GRPO training loop
   def train_with_grpo():
       # Initialize policy model from current checkpoint
       policy_model = GPT(config)
       # Create frozen reference model
       ref_model = create_reference_model(policy_model)
       
       for epoch in range(max_epochs):
           for batch in dataloader:
               # Sample molecules 
               sampled_molecules = sample_multiple_completions(policy_model, batch['prompts'])
               
               # Score molecules
               scores = compute_molecular_rewards(sampled_molecules)
               
               # Calculate GRPO loss
               grpo_loss = grpo_loss_fn(policy_model, ref_model, batch, scores)
               
               # Update policy model
               optimizer.zero_grad()
               grpo_loss.backward()
               optimizer.step()
           
           # Periodically update reference model
           if epoch % update_interval == 0:
               ref_model = create_reference_model(policy_model)
   ```

### Expected Benefits for MolGPT

1. **Molecule Quality Improvement**:
   - GRPO would allow fine-tuning toward specific molecular properties
   - Potential to generate molecules with better drug-likeness, synthesizability, or other targeted properties

2. **Controlled Generation**:
   - The KL divergence term preserves the learned distribution from pre-training
   - Prevents mode collapse or drastic changes to the generation policy

3. **Multi-Objective Optimization**:
   - Multiple reward functions can be combined for complex property optimization
   - Enables balancing of competing objectives (e.g., potency vs. toxicity)

### Implementation Complexity Assessment

The integration of GRPO into MolGPT represents a **high complexity** modification that would require:

1. Substantial changes to the training loop (🔴 Major effort)
2. Implementation of reward functions and scoring (🔴 Major effort)
3. Extension of model outputs to support probability calculations (🟠 Moderate effort)
4. Creation of reference model handling mechanism (🟠 Moderate effort)

However, the core GPT model architecture could remain largely unchanged, and the existing conditioning mechanisms would continue to work in the GRPO framework.

### Recommended Path Forward

1. Create a separate `GRPOTrainer` class that inherits from the current `Trainer`
2. Implement molecule sampling and scoring functions
3. Add reference model management
4. Implement the GRPO loss calculation
5. Create a simplified API for RL fine-tuning of pre-trained MolGPT models

This approach would preserve the existing functionality while adding powerful RL-based fine-tuning capabilities to the MolGPT framework. 