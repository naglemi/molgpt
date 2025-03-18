#!/bin/bash

#unconditional training
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name unconditional_moses --data_name moses2 --batch_size 384 --max_epochs 10 --num_props 0

# property based conditional training
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name logp_moses --data_name moses2 --batch_size 384 --max_epochs 10 --props logp --num_props 1

PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name logp_sas_moses --data_name moses2 --batch_size 384 --max_epochs 10 --props logp sas --num_props 2

# scaffold based conditional training
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name scaffold_moses --data_name moses2 --scaffold --batch_size 384 --max_epochs 10

# scaffold + property based conditional training
PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name logp_scaffold_moses --data_name moses2 --scaffold --batch_size 384 --num_props 1 --max_epochs 10 --props logp

PYTHONPATH=/home/ubuntu/molgpt python train/train.py --run_name logp_sas_scaffold_moses --data_name moses2 --scaffold --batch_size 384 --num_props 2 --max_epochs 10 --props logp sas

