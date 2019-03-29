import requests
import os
import tarfile

import shutil


def extract_tar_file(fname, path):
    with tarfile.open(fname) as f:
        f.extractall(path=path)    

if not os.path.isdir("/scratch/yardima/"):
    os.mkdir("/scratch/yardima/")

if not os.path.isdir("/scratch/yardima/datasets"):
    os.mkdir("/scratch/yardima/datasets")

if not os.path.isdir("/scratch/yardima/datasets/voc12/"):
    os.mkdir("/scratch/yardima/datasets/voc12/")


shutil.copyfile("/srv/glusterfs/yardima/datasets/voc2012.tar", "/scratch/yardima/datasets/voc2012.tar")
shutil.copyfile("/srv/glusterfs/yardima/datasets/bsd_bench.tar", "/scratch/yardima/datasets/bsd_bench.tar")

extract_tar_file("/scratch/yardima/datasets/voc2012.tar", "/scratch/yardima/datasets/voc12/")
extract_tar_file("/scratch/yardima/datasets/bsd_bench.tar", "/scratch/yardima/datasets/")

if not os.path.isdir("/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug"):
    os.mkdir("/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug")


src = "/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug"
src_files = os.listdir(src)
dest_folder = "/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationClassAug/"

for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest_folder)




if not os.path.isdir("/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationObjectAug"):
    os.mkdir("/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationObjectAug")


src = "/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationObjectAug"
src_files = os.listdir(src)
dest_folder = "/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012/SegmentationObjectAug/"

for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest_folder)
