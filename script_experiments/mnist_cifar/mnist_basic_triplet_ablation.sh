#!/bin/bash

source ~/.bash_profile

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/mnist_basic_triplet_ablation.py
