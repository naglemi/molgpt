"""
Reward Functions for MolGPT

This module implements various reward functions for molecule generation,
including similarity-based rewards, property-based rewards, and more.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, AllChem
from rdkit import DataStructs
import math

class SimilarityReward:
    """Reward function based on molecular similarity
    
    Calculates rewards for generated molecules based on similarity to a reference molecule
    or based on structural diversity within a batch.
    """
    
    def __init__(self, reference_smiles=None, mode='diversity', weight=1.0):
        """Initialize similarity reward function
        
        Args:
            reference_smiles: Optional reference SMILES to compare against (for similarity mode)
            mode: Reward mode ('similarity', 'diversity', 'novelty')
            weight: Weight for this reward component
        """
        self.mode = mode
        self.weight = weight
        self.reference_mol = None
        
        if reference_smiles and mode == 'similarity':
            self.reference_mol = Chem.MolFromSmiles(reference_smiles)
            if self.reference_mol:
                self.reference_fp = AllChem.GetMorganFingerprintAsBitVect(
                    self.reference_mol, 2, nBits=2048
                )
    
    def __call__(self, smiles_list):
        """Calculate rewards for a list of SMILES strings
        
        Args:
            smiles_list: List of generated SMILES strings
            
        Returns:
            rewards: Tensor of reward values (batch_size,)
        """
        if isinstance(smiles_list, torch.Tensor):
            # If tensor of indices was passed, handle appropriately
            # (This would require tokenizer to convert back to SMILES)
            pass
            
        # Convert SMILES to molecules and compute fingerprints
        mols = []
        fps = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                mols.append(mol)
                fps.append(fp)
            else:
                # Add placeholder for invalid molecules
                mols.append(None)
                fps.append(None)
        
        # Calculate rewards based on mode
        rewards = []
        
        if self.mode == 'similarity' and self.reference_mol:
            # Reward based on similarity to reference molecule
            for mol, fp in zip(mols, fps):
                if mol and fp:
                    similarity = DataStructs.TanimotoSimilarity(fp, self.reference_fp)
                    rewards.append(float(similarity))
                else:
                    rewards.append(0.0)
                    
        elif self.mode == 'diversity':
            # Reward based on diversity within batch
            # Higher average distance to other molecules = higher diversity = higher reward
            for i, (mol_i, fp_i) in enumerate(zip(mols, fps)):
                if mol_i and fp_i:
                    # Calculate average distance to other valid molecules
                    similarities = []
                    for j, (mol_j, fp_j) in enumerate(zip(mols, fps)):
                        if i != j and mol_j and fp_j:
                            sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                            similarities.append(sim)
                    
                    if similarities:
                        # Reward is inversely proportional to average similarity
                        avg_sim = sum(similarities) / len(similarities)
                        # Transform to reward: lower similarity = higher diversity = higher reward
                        diversity = 1.0 - avg_sim
                        rewards.append(float(diversity))
                    else:
                        # No other valid molecules to compare with
                        rewards.append(0.5)  # Neutral reward
                else:
                    rewards.append(0.0)  # Invalid molecule
                    
        else:
            # Default: reward valid molecules with a property score
            for mol in mols:
                if mol:
                    # Use QED (drug-likeness) as default reward
                    try:
                        reward = QED.qed(mol)
                    except:
                        reward = 0.0
                    rewards.append(float(reward))
                else:
                    rewards.append(0.0)
        
        # Convert to tensor
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) * self.weight


class PropertyReward:
    """Reward function based on molecular properties
    
    Calculates rewards for generated molecules based on various properties
    like QED, LogP, synthetic accessibility, etc.
    """
    
    def __init__(self, property_type='qed', target_value=None, weight=1.0):
        """Initialize property reward function
        
        Args:
            property_type: Type of property ('qed', 'logp', 'sa', etc.)
            target_value: Optional target value for the property
            weight: Weight for this reward component
        """
        self.property_type = property_type
        self.target_value = target_value
        self.weight = weight
    
    def __call__(self, smiles_list):
        """Calculate rewards for a list of SMILES strings
        
        Args:
            smiles_list: List of generated SMILES strings
            
        Returns:
            rewards: Tensor of reward values (batch_size,)
        """
        rewards = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                # Invalid molecule
                rewards.append(0.0)
                continue
                
            # Calculate reward based on specified property
            try:
                if self.property_type == 'qed':
                    # Drug-likeness score (0-1, higher is better)
                    prop_value = QED.qed(mol)
                    reward = prop_value  # Already normalized
                    
                elif self.property_type == 'logp':
                    # Water-octanol partition coefficient
                    prop_value = Crippen.MolLogP(mol)
                    
                    if self.target_value is not None:
                        # Penalize deviation from target
                        reward = 1.0 - min(abs(prop_value - self.target_value), 4.0) / 4.0
                    else:
                        # Default target range: 1.0-3.0 (drug-like)
                        reward = 1.0 - min(abs(prop_value - 2.0), 4.0) / 4.0
                
                elif self.property_type == 'combined':
                    # Combined reward considering multiple properties
                    qed_score = QED.qed(mol)
                    logp = Crippen.MolLogP(mol)
                    logp_score = 1.0 - min(abs(logp - 2.0), 4.0) / 4.0
                    
                    # Combined score with weights
                    reward = 0.6 * qed_score + 0.4 * logp_score
                
                else:
                    # Default to QED
                    reward = QED.qed(mol)
                    
            except:
                # Error computing property
                reward = 0.0
                
            rewards.append(float(reward))
        
        # Convert to tensor
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) * self.weight 