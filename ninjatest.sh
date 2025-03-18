#!/bin/bash

# Simple test script to verify the import fix
# Only runs the first command from train_moses.sh to test the fix
# Output is redirected to training.log to reduce token consumption

# Run the Python script with nohup and redirect output to training.log
PYTHONPATH=/home/ubuntu/molgpt nohup python train/train.py --run_name test_fix --data_name moses2 --batch_size 384 --max_epochs 1 --num_props 0 > training.log 2>&1 &

# Wait a moment for the process to start
sleep 2

# Display the last 500 lines of the log file
echo "Displaying last 500 lines of training.log. Full log available in training.log"
tail -n 500 training.log

# Inform about how to check the log
echo ""
echo "To check the latest log entries, use: tail -n 500 training.log"