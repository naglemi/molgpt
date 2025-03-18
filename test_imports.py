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
