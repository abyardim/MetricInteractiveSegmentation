#!/bin/bash

source ~/.bash_profile

# download dataset
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/download_pascal.py

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/repos/pytorch-deeplab-xception/train.py


