"""
Dataset class for MolGPT compatible with TRL's GRPOTrainer
"""

import torch
from torch.utils.data import Dataset
import json

def load_vocab(vocab_path):
    """Load vocabulary from JSON file"""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab

class MoleculeDataset(Dataset):
    """Dataset class for molecular SMILES strings compatible with TRL"""
    
    def __init__(self, data_path, vocab, block_size=54):
        """Initialize dataset
        
        Args:
            data_path: Path to text file with one SMILES string per line
            vocab: Dictionary mapping tokens to indices
            block_size: Maximum sequence length
        """
        self.block_size = block_size
        self.vocab = vocab
        
        # Load SMILES strings
        with open(data_path) as f:
            self.molecules = [line.strip() for line in f]
            
    def __len__(self):
        return len(self.molecules)
        
    def __getitem__(self, idx):
        """Get a single example
        
        Returns dict with:
            input_ids: Tensor of token indices
            attention_mask: Tensor indicating valid positions
        """
        # Get SMILES string
        smiles = self.molecules[idx]
        
        # Convert to tokens
        tokens = list(smiles)
        token_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        
        # Pad/truncate to block_size
        if len(token_ids) > self.block_size:
            token_ids = token_ids[:self.block_size]
        else:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.block_size - len(token_ids))
            
        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * (self.block_size - len(tokens))
        
        return {
            'input_ids': torch.tensor(token_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
