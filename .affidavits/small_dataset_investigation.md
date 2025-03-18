# AFFIDAVIT: Investigation into Limited SMILES Dataset

## Case Summary
Detective Righteous has been assigned to investigate why the training dataset contains only 3 SMILES strings and the validation dataset contains only 1 SMILE string, when the expected behavior would be to have a much larger dataset for proper model training.

## Initial Evidence
The training log reveals the following critical information:
```
data has 3 smiles, 94 unique characters.
data has 1 smiles, 94 unique characters.
```

This confirms that the dataset being used for training is indeed extremely small, which explains why there is only one iteration per epoch in the training process.

## Investigation Procedure

### Step 1: Examine the Dataset Files
First, I examined the actual dataset files to confirm their size:

```bash
$ wc -l datasets/moses2.csv
6 datasets/moses2.csv

$ cat datasets/moses2.csv
smiles,scaffold_smiles,split,logp,sas,qed,tpsa
CC(=O)OC1=CC=CC=C1C(=O)O,c1ccccc1,train,1.2,2.1,0.7,60.0
CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F,c1ccc(cc1)n1ncc(c1)C(F)(F)F,train,2.5,3.2,0.6,80.0
CC1=C(C=CC=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)F,c1ccc(cc1)N,train,3.1,2.8,0.8,45.0
CC1=CC=C(C=C1)C(=O)NCCCCN2CCN(CC2)C3=CC=CC=C3OC,c1ccccc1OC,test,2.8,3.5,0.75,55.0
CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2C(=O)C3=CC=CC=C3C2=O,c1ccccc1,test_scaffolds,1.9,2.9,0.65,70.0
```

Similarly, the guacamol2.csv file also has only 6 lines (including header).

This confirms that both dataset files are extremely small, containing only a handful of examples:
- moses2.csv: 3 training examples, 1 test example, 1 test_scaffolds example
- guacamol2.csv: Similar small size

### Step 2: Investigate Dataset Download Mechanism
Next, I examined the download_datasets.py script to understand how these datasets are supposed to be obtained:

```python
# Direct file URLs for the datasets
dataset_urls = {
    'moses2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E',
    'guacamol2.csv': 'https://drive.google.com/uc?id=1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E'
}
```

This revealed a critical issue: both dataset files are set to download from the same URL. Furthermore, the script downloads the files to the current directory but doesn't move them to the datasets folder.

### Step 3: Examine Training Scripts
I examined the training scripts to understand how the datasets are used:

```bash
# From train_moses.sh
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name unconditional_moses --data_name moses2 --batch_size 384 --max_epochs 10 --num_props 0
```

```bash
# From ninjatest.sh
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name test_fix --data_name moses2 --batch_size 384 --max_epochs 1 --num_props 0
```

Both scripts use the same dataset name (moses2) and similar parameters, but ninjatest.sh is configured to run for only 1 epoch as a test.

## Findings and Conclusion

Based on the evidence collected, I can conclusively identify the culprits responsible for the limited dataset size:

1. **Primary Culprit**: The download_datasets.py script has two critical flaws:
   - Both dataset files (moses2.csv and guacamol2.csv) are set to download from the same URL
   - The script doesn't move the downloaded files to the datasets directory

2. **Secondary Culprit**: The current moses2.csv file in the datasets directory is a tiny sample dataset with only 5 data rows (3 training, 1 test, 1 test_scaffolds), not the full dataset that would be expected for proper model training.

The combination of these issues results in the training process using an extremely small dataset, which explains why there is only one iteration per epoch and why the training completes so quickly.

## Impact
The small dataset size has significant implications:
1. The model cannot learn effectively from such a limited number of examples
2. The training process completes very quickly but produces a model with poor generalization ability
3. The single iteration per epoch makes it impossible to observe meaningful trends in the loss over time

This investigation conclusively proves that the limited dataset size is not an error in the training process itself, but rather a result of using sample datasets instead of the full datasets that should be downloaded properly.