
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import os

import PIL

class Davis2017(data.Dataset):

    def __init__(
            self,
            *,
            root,
            transforms,
            train
    ):
        self.root = root
        self.train = train


        if train:
            with open(os.path.join(root,"ImageSets/2017/train.txt")) as f:
                names = [line.rstrip('\n') for line in f]
        else:
            with open(os.path.join(root,"ImageSets/2017/val.txt")) as f:
                names = [line.rstrip('\n') for line in f]

        self.names = names


        self.transforms = transforms

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        seq, frame = i
        file_name = str(frame).zfill(5)
        img = PIL.Image.open(os.path.join(self.root, "JPEGImages/480p/", self.names[seq], file_name + ".jpg"))
        lbl = PIL.Image.open(os.path.join(self.root, "Annotations/480p/", self.names[seq], file_name + ".png"))

        out = {"image": img, "label": lbl}
        out = self.transforms(out)

        return out

    def count_frames(self, seq):
        d = os.path.join(self.root, "JPEGImages/480p/", self.names[seq])
        path, dirs, files = next(os.walk(d))
        return len(files)





