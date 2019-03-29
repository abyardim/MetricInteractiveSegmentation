
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import os

import PIL


from random import shuffle

class DavisFrameSampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, davis, n_samples):
        self.davis = davis
#         self.num_seq = num_seq
        self.n_samples = n_samples
        
        self.img_count = 0
        imgs = []
        for i in range(len(davis)):
            self.img_count += dataset.count_frames(i)
            
            for j in range( dataset.count_frames(i)):
                imgs.append([i,j])
        
        self.imgs = imgs
                
    def __iter__(self):
        samples = []
        for i in np.random.permutation(self.img_count):
            frame0 = self.imgs[i]
            frames = [frame0]
            
            candidates = list(range(dataset.count_frames(frame0[0])))
            del candidates[frame0[1]]
            candidates = np.array(candidates, dtype=np.int32)
            for f in np.random.choice(candidates, size=self.n_samples):
                frames.append([frame0[0], f])
                
            samples.append(frames)

        return iter(samples)

    def __len__(self):
        return self.img_count

