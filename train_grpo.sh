#!/bin/bash

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Run GRPO training
python train_grpo_sa.py \
    --pretrained_model checkpoints/pretrained_molgpt.pt \
    --data_path data/molecules.txt \
    --vocab_path data/vocab.json \
    --output_dir outputs/grpo_sa \
    --learning_rate 1e-5 \
    --group_size 8 \
    --batch_size 4 \
    --max_steps 10000 