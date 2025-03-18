#!/usr/bin/env python
"""
Test only the essential module imports for MolGPT
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from rdkit import Chem

print("Basic imports successful!")

# Add the current directory to the path to make imports work
sys.path.insert(0, os.path.abspath('.'))

# Check train module
try:
    from train.model import GPT, GPTConfig
    from train.trainer import Trainer, TrainerConfig
    from train.dataset import SmileDataset
    print("✅ Train module imports successful")
except Exception as e:
    print(f"❌ Train module import error: {e}")

# Try set_seed from train.utils
try:
    import train.utils
    set_seed_fn = getattr(train.utils, 'set_seed', None)
    if set_seed_fn is not None:
        print("✅ set_seed function found in train.utils")
    else:
        print("❌ set_seed function not found in train.utils")
except Exception as e:
    print(f"❌ Error importing train.utils: {e}")

# Check generate module and functions
try:
    import generate.utils
    sample_fn = getattr(generate.utils, 'sample', None)
    canonic_smiles_fn = getattr(generate.utils, 'canonic_smiles', None)
    
    if sample_fn is not None:
        print("✅ sample function found in generate.utils")
    else:
        print("❌ sample function not found in generate.utils")
        
    if canonic_smiles_fn is not None:
        print("✅ canonic_smiles function found in generate.utils")
    else:
        print("❌ canonic_smiles function not found in generate.utils")
        
except Exception as e:
    print(f"❌ Error importing generate.utils: {e}")

# Check moses module
try:
    import moses.utils
    get_mol_fn = getattr(moses.utils, 'get_mol', None)
    if get_mol_fn is not None:
        print("✅ get_mol function found in moses.utils")
    else:
        print("❌ get_mol function not found in moses.utils")
except Exception as e:
    print(f"❌ Error importing moses.utils: {e}")

print("\nVerification complete!") 