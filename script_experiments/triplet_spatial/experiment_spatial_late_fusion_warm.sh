#!/bin/bash

source ~/.bash_profile

# download dataset
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/copy_pascal.py

# actually execute script
CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=1 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm 




CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=2 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --wd=1e-4

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=3 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --wd=1e-3

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=4 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --wd=3e-3




CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=5 --epochs=50 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=6 --epochs=30 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=7 --epochs=20 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  



CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=8 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --ntripletstrain=250

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=9 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --ntripletstrain=700

CUDA_VISIBLE_DEVICES=$SGE_GPU python ~/Python/script_experiments/triplet_spatial/warm_init.py --runid=10 --epochs=40 --name="lfov-spatial-late" --saveroot=/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet-spatial-late-warm  --ntripletstrain=1000
