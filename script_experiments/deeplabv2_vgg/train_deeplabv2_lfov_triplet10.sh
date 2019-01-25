#!/bin/bash

source ~/.bash_profile

# download dataset
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/copy_pascal.py

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/train_deeplabv2_lfov_triplet10.py
