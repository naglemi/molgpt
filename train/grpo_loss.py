# ================================================================
# ORGANIZATIONAL POLICY FOR GRPO IMPLEMENTATION
# ================================================================
# 1. THIS CODEBASE USES TRL'S GRPO IMPLEMENTATION
# 2. DO NOT CREATE CUSTOM GRPO IMPLEMENTATIONS
# 3. ALL GRPO FUNCTIONALITY MUST USE trl.trainer.GRPOTrainer
# 4. SEE train_grpo_sa.py FOR THE OFFICIAL TRAINING SCRIPT
# ================================================================

from trl import GRPOTrainer, GRPOConfig

# This file exists only to document our GRPO implementation policy.
# For actual implementation, see train_grpo_sa.py which uses TRL directly.