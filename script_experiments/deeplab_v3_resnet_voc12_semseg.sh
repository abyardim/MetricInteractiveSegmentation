#!/bin/bash

source ~/.bash_profile


# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/repos/pytorch-deeplab-xception/train.py


