#!/usr/bin/env python
"""
MolGPT Notebook Testing Script
This script tests the MolGPT_Cowboy_Chronicle.ipynb notebook using nbconvert
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def log_info(message):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")

def log_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")

def log_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {message}")

def log_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {message}")

def modify_notebook(input_path, output_path):
    """
    Creates a modified version of the notebook with corrected paths and reduced training time.
    """
    log_info(f"Modifying notebook: {input_path} -> {output_path}")
    
    with open(input_path, 'r') as f:
        notebook = json.load(f)
    
    # Make a copy of the notebook
    modified_notebook = notebook.copy()
    
    # Modify paths and training parameters
    for cell in modified_notebook['cells']:
        if cell['cell_type'] == 'code':
            # Fix model weight path
            if 'model_weight = ' in ''.join(cell['source']):
                for i, line in enumerate(cell['source']):
                    if 'model_weight = ' in line:
                        # Update to relative path
                        cell['source'][i] = 'model_weight = "datasets/weights/moses_scaf_wholeseq_qed.pt"\n'
            
            # Reduce training epochs and batch size
            if 'train_config = TrainerConfig(' in ''.join(cell['source']):
                for i, line in enumerate(cell['source']):
                    if 'max_epochs=' in line:
                        cell['source'][i] = '    max_epochs=1,  # Reduced for testing\n'
                    if 'batch_size=' in line:
                        cell['source'][i] = '    batch_size=32,  # Reduced for testing\n'
            
            # Add explicit validation for generated molecules
            if 'all_scaffold_molecules.extend(molecules)' in ''.join(cell['source']):
                cell['source'].append('\n    # Validation check\n')
                cell['source'].append('    print(f"Validation: {len(valid_smiles)} of {len(generated_smiles)} molecules are valid.")\n')
                cell['source'].append('    for i, smiles in enumerate(valid_smiles[:1]):\n')
                cell['source'].append('        mol = get_mol(smiles)\n')
                cell['source'].append('        if mol:\n')
                cell['source'].append('            print(f"First valid molecule SMILES: {smiles}")\n')
    
    # Write the modified notebook
    with open(output_path, 'w') as f:
        json.dump(modified_notebook, f)
    
    log_success(f"Notebook modified successfully")
    return output_path

def run_notebook(notebook_path):
    """
    Run the notebook using nbconvert
    """
    log_info(f"Running notebook: {notebook_path}")
    
    # Create a temporary directory for the output
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.html")
    
    try:
        # Execute the notebook
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "html", 
            "--execute",
            "--ExecutePreprocessor.timeout=600",  # 10-minute timeout
            "--output", output_path,
            notebook_path
        ]
        
        log_info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log_error(f"Notebook execution failed:")
            log_error(result.stderr)
            return False, result.stderr
        
        log_success(f"Notebook executed successfully")
        log_info(f"Output saved to: {output_path}")
        
        # Check for error messages in output
        with open(output_path, 'r') as f:
            content = f.read()
            if "Error" in content or "Exception" in content:
                log_warning("Potential errors found in notebook output")
            else:
                log_success("No obvious errors found in notebook output")
        
        return True, output_path
    
    except Exception as e:
        log_error(f"Error executing notebook: {e}")
        return False, str(e)
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(output_dir)

def main():
    """
    Main function to test the notebook
    """
    log_info("Starting MolGPT notebook test")
    
    # Check if nbconvert is installed
    try:
        subprocess.run(["jupyter", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        log_error("jupyter command not found. Please install jupyter and nbconvert.")
        return False
    
    notebook_path = "MolGPT_Cowboy_Chronicle.ipynb"
    
    # Check if notebook exists
    if not os.path.exists(notebook_path):
        log_error(f"Notebook not found: {notebook_path}")
        return False
    
    # Create a modified version of the notebook
    modified_notebook_path = "MolGPT_Cowboy_Chronicle_Test.ipynb"
    modified_notebook_path = modify_notebook(notebook_path, modified_notebook_path)
    
    # Run the modified notebook
    success, output = run_notebook(modified_notebook_path)
    
    # Clean up
    os.remove(modified_notebook_path)
    
    if success:
        log_success("Notebook test completed successfully")
        return True
    else:
        log_error("Notebook test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 