# 🥷 GRPO Validation Cycle Scroll

## Ninja Surveillance Plan

### Phase 1: Initial Strike (Steps 0-10)
- Monitor trajectory generation
- Verify SMILES validity
- Check reward computation
- Observe KL divergence initialization

### Phase 2: Shadow Observation (Steps 10-50)
- Track policy gradient updates
- Monitor group advantage computation
- Observe reference model state
- Watch for policy drift signs

### Phase 3: Deep Infiltration (Steps 50-100)
- Analyze reward trends
- Verify learning stability
- Monitor molecule diversity
- Validate SA score improvements

## Ninja Abort Signals
1. KL divergence spikes (> 0.1)
2. Policy collapse (validity rate < 50%)
3. Reward collapse (mean reward < 0)
4. Training instability (loss NaN/Inf)

## Success Criteria
1. Valid molecule rate > 80%
2. Stable KL divergence (< 0.05)
3. Improving SA scores
4. Consistent group advantages

## Observation Points
```python
wandb.log({
    "validity_rate": validity_rate,     # Must stay high
    "mean_sa_score": mean_sa,          # Should improve
    "mean_qed_score": mean_qed,        # Should remain stable
    "kl_div": kl_div,                  # Must stay controlled
    "policy_loss": policy_loss         # Should decrease smoothly
})
```

## Emergency Protocols
1. If KL spikes: Reduce learning rate by 50%
2. If rewards collapse: Check molecule validity
3. If policy unstable: Revert to last checkpoint
4. If all else fails: Terminate and analyze logs

## Next Cycle Decision Points
- At step 25: First checkpoint, assess stability
- At step 50: Mid-point, check learning trends
- At step 75: Late-stage, verify improvements
- At step 100: Final evaluation, decide on full training

*A ninja's validation is silent but thorough - we strike small to learn how to strike big* 🥷 