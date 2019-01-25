import torch
from torch.utils import data
from torchvision import transforms

class SubDataset(data.Dataset):
    """
    Create a dataset from a subset of entries in a parent dataset.
    """

    def __init__(
            self,
            *,
            parent_dataset,
            indices):
        self.parent_dataset = parent_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.parent_dataset[self.indices[index]]