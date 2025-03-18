# 🤠 MolGPT Setup Guide - Cowboy Chronicle 🤠

This guide documents the steps to set up the MolGPT environment and get the notebooks running successfully.

## 1. Create and Activate Conda Environment

First, create a new conda environment for MolGPT:

```bash
conda create -n molgpt python=3.8
conda activate molgpt
```

## 2. Install Core Dependencies

Install PyTorch and RDKit:

```bash
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y -c conda-forge rdkit
```

## 3. Install Additional Dependencies

Install other required packages:

```bash
pip install moses pandas numpy matplotlib seaborn jupyter wandb tqdm fsspec
```

## 4. Fix Import Issues

The project had some import issues that needed to be fixed:

1. Created a custom `moses` module with utility functions:
   - Created `/home/ubuntu/molgpt/moses/__init__.py`
   - Created `/home/ubuntu/molgpt/moses/utils.py` with the `get_mol` function and other utilities

2. Fixed import paths in the source files:
   - Updated `train/trainer.py` to import from `generate.utils` instead of `utils`
   - Updated `train/dataset.py` to import from `train.utils` instead of `utils`

3. Updated notebook import paths:
   - Modified `MolGPT_Training.ipynb` and `MolGPT_Generation.ipynb` to use `sys.path.insert(0, '.')` for direct imports

4. Fixed model weights path in `MolGPT_Generation.ipynb`:
   - Updated to use the full path: `/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_{model_type}.pt`

## 5. Running the Notebooks

The notebooks can now be executed using:

```bash
jupyter nbconvert --to html --execute MolGPT_Training.ipynb
jupyter nbconvert --to html --execute MolGPT_Generation.ipynb
```

## 6. Model Architecture Note

The model architecture in the code doesn't match the architecture of the saved weights. The model in the code expects weights with names like "pos_emb", "tok_emb.weight", etc., but the saved weights have names like "embedding.weight", "transformer.layers.0.self_attn.in_proj_weight", etc.

For the Generation notebook to work properly, you would need to either:
1. Modify the model architecture to match the saved weights
2. Train a new model with the current architecture
3. Convert the saved weights to match the expected format

## 7. Available Pre-trained Models

The following pre-trained models are available:

- `/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_logp_newtokens.pt`
- `/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_qed.pt`
- `/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_sas.pt`
- `/home/ubuntu/molgpt/datasets/weights/moses_scaf_wholeseq_tpsa.pt`

These models are trained on the Moses dataset with different property conditions.