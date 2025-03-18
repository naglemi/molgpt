#!/usr/bin/env python
"""
Create a mock dataset for testing the MolGPT notebook
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# SMILES examples from the existing dataset
example_smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "CC1=C(C=CC=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)F",
    "CC1=CC=C(C=C1)C(=O)NCCCCN2CCN(CC2)C3=CC=CC=C3OC",
    "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2C(=O)C3=CC=CC=C3C2=O",
    # Add more examples to get a reasonable dataset size
    "COc1cc(ccc1OC)C(=O)NCCN(C)C",
    "CC(C)OC(=O)C(C)NP(=O)(OC1CCCCC1)OC2CCCCC2",
    "CC1=CC=C(C=C1)C(=O)NC2=CC(=CC=C2)C(=O)O",
    "COC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=CC=C3OC",
    "CC1=CC=CC=C1NC(=O)C2=CC=C(C=C2)S(=O)(=O)N",
    "CC1=CC=C(C=C1)C(=O)NC2=CC(=CC=C2)C#N",
    "COC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)S(=O)(=O)N",
    "CC1=CC(=CC=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N",
    "CC1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N",
    "CC1=CC(=CC=C1)C(=O)NC2=CC=C(C=C2)S(=O)(=O)N",
]

# Generate a larger dataset from these examples
def generate_mock_dataset(size=100):
    """Generate a mock dataset for testing"""
    smiles_list = []
    scaffolds = []
    
    # Extend the list to reach desired size
    for _ in range(size // len(example_smiles) + 1):
        smiles_list.extend(example_smiles)
    
    # Limit to desired size
    smiles_list = smiles_list[:size]
    
    # Generate scaffolds
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Generate scaffold
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            scaffolds.append(scaffold)
        else:
            # If invalid molecule, use a simple scaffold
            scaffolds.append("c1ccccc1")
    
    # Generate random properties
    np.random.seed(42)
    logp = np.random.uniform(low=0.5, high=5.0, size=size)
    sas = np.random.uniform(low=1.5, high=6.0, size=size)
    qed = np.random.uniform(low=0.1, high=0.9, size=size)
    tpsa = np.random.uniform(low=20.0, high=140.0, size=size)
    
    # Assign splits (80% train, 10% test, 10% test_scaffolds)
    splits = ['train'] * int(0.8 * size)
    splits.extend(['test'] * int(0.1 * size))
    splits.extend(['test_scaffolds'] * (size - len(splits)))
    np.random.shuffle(splits)
    
    # Create DataFrame
    df = pd.DataFrame({
        'smiles': smiles_list,
        'scaffold_smiles': scaffolds,
        'split': splits,
        'logp': logp,
        'sas': sas,
        'qed': qed,
        'tpsa': tpsa
    })
    
    return df

if __name__ == "__main__":
    # Make sure the datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    # Generate the mock dataset
    df = generate_mock_dataset(size=1000)
    
    # Save to CSV
    output_path = "datasets/moses2.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Mock dataset created at {output_path} with {len(df)} entries")
    
    # Display first few entries
    print("\nFirst 5 entries:")
    print(df.head()) 