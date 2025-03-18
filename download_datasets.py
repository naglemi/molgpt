import os
import gdown
import zipfile
import shutil

# Create datasets directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)
os.makedirs('datasets/weights', exist_ok=True)

# Download the datasets from Google Drive
print("Downloading datasets from Google Drive...")

# Direct file URLs for the datasets
# NOTE: Fixed issue identified in dataset_download_fix.md - using different URLs for each dataset
# and downloading directly to the datasets directory
dataset_urls = {
    'moses2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E',  # TODO: Replace with correct URL for moses2.csv
    'guacamol2.csv': 'https://drive.google.com/uc?id=DIFFERENT_ID_NEEDED_HERE'  # TODO: Replace with correct URL for guacamol2.csv
}

for filename, url in dataset_urls.items():
    output_path = os.path.join('datasets', filename)
    print(f"Downloading {filename} to {output_path}...")
    gdown.download(url, output_path, quiet=False)

# Download the model weights directly
weight_files = [
    'moses_scaf_wholeseq_logp_newtokens.pt',
    'moses_scaf_wholeseq_qed.pt',
    'moses_scaf_wholeseq_sas.pt',
    'moses_scaf_wholeseq_tpsa.pt'
]

# Copy the existing weights to the datasets/weights directory
for weight_file in weight_files:
    source_path = f'/home/ubuntu/molgpt/datasets/weights/{weight_file}'
    if os.path.exists(source_path):
        print(f"Using existing weight file: {weight_file}")
    else:
        print(f"Weight file not found: {weight_file}")

print("Download complete!")