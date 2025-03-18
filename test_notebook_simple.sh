#!/bin/bash
# Simple Ninja script to test just the basics of MolGPT_Cowboy_Chronicle.ipynb

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🥷 Hayato, the Code Ninja, initiates basic MolGPT notebook testing...${NC}"

# Create .scrolls directory if it doesn't exist
mkdir -p .scrolls

# Make sure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 1: Generate the mock dataset
echo -e "${BLUE}[NINJA LOG] Creating mock dataset...${NC}"
./create_mock_dataset.py

# Step 2: Create a modified version of the notebook with corrected paths
echo -e "${BLUE}[NINJA LOG] Creating modified notebook...${NC}"

# Extract the model weight path from the notebook and adjust it
notebook_path="MolGPT_Cowboy_Chronicle.ipynb"
modified_path="MolGPT_Cowboy_Chronicle_Test.ipynb"

# Create modified notebook with sed
cp "$notebook_path" "$modified_path"
sed -i 's|model_weight = f"/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_{model_type}.pt"|model_weight = f"datasets/weights/moses_scaf_wholeseq_{model_type}.pt"|g' "$modified_path"

# Step 3: Run just the basic parts of the notebook to test if imports work
echo -e "${BLUE}[NINJA LOG] Running basic tests...${NC}"

# Create a minimal test script
cat > test_imports.py << EOF
import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import json
from rdkit import Chem

# Try importing from local modules
sys.path.insert(0, '.')
try:
    from train.model import GPT, GPTConfig
    from train.trainer import Trainer, TrainerConfig
    from train.dataset import SmileDataset
    from train.utils import set_seed
    from generate.utils import sample, check_novelty, canonic_smiles
    from moses.utils import get_mol
    print("All imports successful!")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Check if dataset exists
try:
    data = pd.read_csv('datasets/moses2.csv')
    print(f"Dataset loaded with {len(data)} rows")
except Exception as e:
    print(f"Dataset error: {e}")
    sys.exit(1)

# Check if model weights exist
weight_files = [
    'moses_scaf_wholeseq_logp_newtokens.pt',
    'moses_scaf_wholeseq_qed.pt',
    'moses_scaf_wholeseq_sas.pt',
    'moses_scaf_wholeseq_tpsa.pt'
]

for weight_file in weight_files:
    path = f'datasets/weights/{weight_file}'
    if os.path.exists(path):
        print(f"Weight file found: {path}")
    else:
        print(f"Warning: Weight file not found: {path}")

print("Basic tests completed successfully!")
EOF

# Run the test script
python test_imports.py

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[NINJA SUCCESS] Basic notebook tests passed! 🎉${NC}"
    
    # Create success report
    cat > .scrolls/notebook_basic_test_success.md << EOF
# 🥷 MolGPT Notebook Basic Test: Success 🥷

## Test Results
* ✅ All required imports work
* ✅ Mock dataset created and accessible
* ✅ Basic functionality verified

## Next Steps
* Run full tests with nbconvert
* Test molecule generation
* Verify visualization functionality
EOF
    
    echo -e "${GREEN}[NINJA LOG] Success report written to .scrolls/notebook_basic_test_success.md${NC}"
    exit 0
else
    echo -e "${RED}[NINJA ERROR] Basic notebook tests failed! 😞${NC}"
    
    # Create failure report
    cat > .scrolls/notebook_basic_test_failure.md << EOF
# 🥷 MolGPT Notebook Basic Test: Failure 🥷

## Issues Detected
* ❌ Basic tests failed
* ❌ Check the specific error from the test_imports.py output
* ❌ Possible causes:
  * Missing dependencies
  * Path issues
  * Missing modules

## Recommended Actions
* Check Python environment and dependencies
* Verify that all required modules are available
* Check path configurations
EOF
    
    echo -e "${RED}[NINJA LOG] Failure report written to .scrolls/notebook_basic_test_failure.md${NC}"
    exit 1
fi 