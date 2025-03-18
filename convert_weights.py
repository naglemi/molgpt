import torch
import os
import argparse

def convert_weights(input_path, output_path):
    """
    Convert weights from the old model architecture to the new model architecture.
    
    Args:
        input_path: Path to the input weights file
        output_path: Path to save the converted weights file
    """
    print(f"Loading weights from {input_path}...")
    old_state_dict = torch.load(input_path, map_location='cpu')
    
    # Create a new state dict with the expected keys
    new_state_dict = {}
    
    # Map the old keys to the new keys
    key_mapping = {
        'embedding.weight': 'tok_emb.weight',
        'transformer.layers.0.self_attn.in_proj_weight': ['blocks.0.attn.query.weight', 'blocks.0.attn.key.weight', 'blocks.0.attn.value.weight'],
        'transformer.layers.0.self_attn.in_proj_bias': ['blocks.0.attn.query.bias', 'blocks.0.attn.key.bias', 'blocks.0.attn.value.bias'],
        'transformer.layers.0.self_attn.out_proj.weight': 'blocks.0.attn.proj.weight',
        'transformer.layers.0.self_attn.out_proj.bias': 'blocks.0.attn.proj.bias',
        'transformer.layers.0.linear1.weight': 'blocks.0.mlp.0.weight',
        'transformer.layers.0.linear1.bias': 'blocks.0.mlp.0.bias',
        'transformer.layers.0.linear2.weight': 'blocks.0.mlp.2.weight',
        'transformer.layers.0.linear2.bias': 'blocks.0.mlp.2.bias',
        'transformer.layers.0.norm1.weight': 'blocks.0.ln1.weight',
        'transformer.layers.0.norm1.bias': 'blocks.0.ln1.bias',
        'transformer.layers.0.norm2.weight': 'blocks.0.ln2.weight',
        'transformer.layers.0.norm2.bias': 'blocks.0.ln2.bias',
        # Add mappings for layers 1-3
        'fc.weight': 'head.weight',
        'fc.bias': 'head.bias'
    }
    
    # Add mappings for layers 1-3
    for i in range(1, 4):
        key_mapping[f'transformer.layers.{i}.self_attn.in_proj_weight'] = [f'blocks.{i}.attn.query.weight', f'blocks.{i}.attn.key.weight', f'blocks.{i}.attn.value.weight']
        key_mapping[f'transformer.layers.{i}.self_attn.in_proj_bias'] = [f'blocks.{i}.attn.query.bias', f'blocks.{i}.attn.key.bias', f'blocks.{i}.attn.value.bias']
        key_mapping[f'transformer.layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.proj.weight'
        key_mapping[f'transformer.layers.{i}.self_attn.out_proj.bias'] = f'blocks.{i}.attn.proj.bias'
        key_mapping[f'transformer.layers.{i}.linear1.weight'] = f'blocks.{i}.mlp.0.weight'
        key_mapping[f'transformer.layers.{i}.linear1.bias'] = f'blocks.{i}.mlp.0.bias'
        key_mapping[f'transformer.layers.{i}.linear2.weight'] = f'blocks.{i}.mlp.2.weight'
        key_mapping[f'transformer.layers.{i}.linear2.bias'] = f'blocks.{i}.mlp.2.bias'
        key_mapping[f'transformer.layers.{i}.norm1.weight'] = f'blocks.{i}.ln1.weight'
        key_mapping[f'transformer.layers.{i}.norm1.bias'] = f'blocks.{i}.ln1.bias'
        key_mapping[f'transformer.layers.{i}.norm2.weight'] = f'blocks.{i}.ln2.weight'
        key_mapping[f'transformer.layers.{i}.norm2.bias'] = f'blocks.{i}.ln2.bias'
    
    # Initialize missing tensors
    vocab_size = old_state_dict['embedding.weight'].shape[0]
    embed_dim = old_state_dict['embedding.weight'].shape[1]
    block_size = 54  # Default block size for Moses dataset
    
    # Create positional embeddings
    new_state_dict['pos_emb'] = torch.zeros(1, block_size, embed_dim)
    
    # Create type embeddings
    new_state_dict['type_emb.weight'] = torch.zeros(2, embed_dim)
    
    # Create property network weights
    new_state_dict['prop_nn.weight'] = torch.zeros(1, embed_dim)
    new_state_dict['prop_nn.bias'] = torch.zeros(embed_dim)
    
    # Create attention masks
    for i in range(8):  # Assuming 8 layers
        new_state_dict[f'blocks.{i}.attn.mask'] = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
    
    # Copy weights from old state dict to new state dict
    for old_key, new_key in key_mapping.items():
        if old_key in old_state_dict:
            if isinstance(new_key, list):
                # Split the weights for query, key, value
                if 'in_proj_weight' in old_key:
                    weight = old_state_dict[old_key]
                    chunk_size = weight.size(0) // 3
                    query_weight, key_weight, value_weight = weight.chunk(3)
                    new_state_dict[new_key[0]] = query_weight
                    new_state_dict[new_key[1]] = key_weight
                    new_state_dict[new_key[2]] = value_weight
                # Split the biases for query, key, value
                elif 'in_proj_bias' in old_key:
                    bias = old_state_dict[old_key]
                    chunk_size = bias.size(0) // 3
                    query_bias, key_bias, value_bias = bias.chunk(3)
                    new_state_dict[new_key[0]] = query_bias
                    new_state_dict[new_key[1]] = key_bias
                    new_state_dict[new_key[2]] = value_bias
            else:
                new_state_dict[new_key] = old_state_dict[old_key]
    
    # Save the converted weights
    print(f"Saving converted weights to {output_path}...")
    torch.save(new_state_dict, output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model weights from old to new architecture")
    parser.add_argument("--input", type=str, required=True, help="Path to input weights file")
    parser.add_argument("--output", type=str, required=True, help="Path to output weights file")
    args = parser.parse_args()
    
    convert_weights(args.input, args.output)