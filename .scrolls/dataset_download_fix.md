# Ninja Scroll: Fixing Dataset Download Script

## Target
The `download_datasets.py` script has two critical flaws identified by Detective Righteous:
1. Both dataset files (moses2.csv and guacamol2.csv) are set to download from the same URL
2. The script doesn't move the downloaded files to the datasets directory

These issues result in tiny sample datasets being used for training, causing the model to train on only 3 SMILES strings.

## Analysis
The current implementation in `download_datasets.py`:

```python
# Direct file URLs for the datasets
dataset_urls = {
    'moses2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E',
    'guacamol2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E'
}

for filename, url in dataset_urls.items():
    print(f"Downloading {filename}...")
    gdown.download(url, filename, quiet=False)
```

The issues are:
1. Both files are downloading from the same URL
2. Files are downloaded to the current directory, not to the datasets directory

## Ninja Plan
Following the Way of the Code Ninja, I will make a precise, surgical modification to the `download_datasets.py` script to:

1. Use different URLs for each dataset (if available, or at least make it clear that proper URLs should be provided)
2. Ensure files are downloaded directly to the datasets directory

This approach will:
- Fix the root cause of the issue (improper dataset downloading)
- Maintain the core functionality of the script
- Make the intent clear for future developers

## Implementation
The modified script will:
```python
# Direct file URLs for the datasets
dataset_urls = {
    'moses2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E',  # TODO: Replace with correct URL for moses2.csv
    'guacamol2.csv': 'https://drive.google.com/uc?id=DIFFERENT_ID_NEEDED_HERE'  # TODO: Replace with correct URL for guacamol2.csv
}

for filename, url in dataset_urls.items():
    output_path = os.path.join('datasets', filename)
    print(f"Downloading {filename} to {output_path}...")
    gdown.download(url, output_path, quiet=False)
```

This implementation follows the Creed of the Code Ninja by being precise, minimal, and surgical in addressing the specific issue.