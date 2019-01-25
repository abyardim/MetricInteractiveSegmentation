#!/bin/bash

source ~/.bash_profile

# download dataset
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/download_pascal.py

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/repos/pytorch-semseg/train.py --config ~/Python/repos/pytorch-semseg/configs/fcn8s_pascal.yml


