#!/bin/bash
source ~/.bash_profile

# actually execute script
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_min_test3.py

