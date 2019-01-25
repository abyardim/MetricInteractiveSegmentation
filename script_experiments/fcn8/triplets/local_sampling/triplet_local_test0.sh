#!/bin/bash
source ~/.bash_profile

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/local_sampling/triplet_local_test0.py

