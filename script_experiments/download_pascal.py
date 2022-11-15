import requests
from tqdm import tqdm
import os
import tarfile

import ptsemseg
import ptsemseg.loader


if not os.path.isdir("/srv/glusterfs/yardima/datasets/"):
    os.mkdir("/srv/glusterfs/yardima/datasets/")
    
if not os.path.isdir("/srv/glusterfs/yardima/datasets/voc12"):
    os.mkdir("/srv/glusterfs/yardima/datasets/voc12")

def extract_tar_file(fname, path):
    with tarfile.open(fname) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path=path)

def download_file(url, local_filename):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=1024)): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

if not os.path.isdir("/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012"):
    if not os.path.isfile("/srv/glusterfs/yardima/datasets/voc2012.tar"):
        download_file("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                      "/srv/glusterfs/yardima/datasets/voc2012.tar")

    extract_tar_file("/srv/glusterfs/yardima/datasets/voc2012.tar", "/srv/glusterfs/yardima/datasets/voc12/")

if not os.path.isdir("/srv/glusterfs/yardima/datasets/benchmark_RELEASE/"):
    if not os.path.isfile( "/srv/glusterfs/yardima/datasets/bsd_bench.tar"):
        download_file("http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
                      "/srv/glusterfs/yardima/datasets/bsd_bench.tar")

    extract_tar_file("/srv/glusterfs/yardima/datasets/bsd_bench.tar", "/srv/glusterfs/yardima/datasets/")



data_loader = ptsemseg.loader.get_loader("pascal")

data_aug = None

t_loader = data_loader(
        "/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012",
        is_transform=True,
        split="train_aug",
        img_size=("same", "same"),
        augmentations=data_aug)


v_loader = data_loader("/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012",
                        is_transform=True,
                        split="val",
                        img_size=("same", "same"),)

