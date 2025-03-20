#!/bin/bash

# Ensure output directory exists
mkdir -p outputs/grpo_validation

# Install/update requirements
pip install -r requirements.txt

# Run validation
python validate_grpo.py 