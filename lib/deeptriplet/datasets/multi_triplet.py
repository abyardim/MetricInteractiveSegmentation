import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps, ImageFilter
import math
import random

from torch.utils import data
from torchvision import transforms

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir + image)
        masks.append(data_dir + mask)

    return images, masks

class PascalMultiTriplet(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.
    """

    def __init__(
            self,
            *,
            pascal_root,
            split_file,
            n_triplets,
            samples_pos,
            samples_neg
    ):
        self.split_file = split_file
        self.pascal_root = pascal_root
        self.n_triplets = n_triplets

        self.n_classes = 21
        
        self.crop_size = 513
        self.base_size = 513

        self.image_list, self.label_list = read_labeled_image_list(self.pascal_root, self.split_file)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        
        self.fill_image = (124, 116, 104)
        self.fill_label = 255
        
        self.samples_pos = samples_pos
        self.samples_neg = samples_neg


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]
        lbl_path = self.label_list[index]

        img = PIL.Image.open(im_path)
        lbl = PIL.Image.open(lbl_path)

        ## augmentation
        img, lbl = self._augment(img, lbl)

        img, lbl = self._random_crop(img, lbl)
        
        img = np.array(img, dtype=np.float32) / 255.0
        lbl = np.array(lbl, dtype=np.long)
        #         lbl[lbl==255] = 0

        img = self.transforms(img)
        
        minrange = [0, 0]
        maxrange = [513, 513]
        
        triplets = self._generate_triplet(lbl)
        

        return (img, *triplets)
    
    
    def _augment(self, img, lbl):
        
        if np.random.rand() > 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            
        if np.random.random() < 0.5:
            img = img.filter(PIL.ImageFilter.GaussianBlur(
                radius=random.random()))

        
        
        return img, lbl

    
    def _random_crop(self, img, mask):
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        padh = self.crop_size - oh if oh < self.crop_size else 0
        padw = self.crop_size - ow if ow < self.crop_size else 0
        if short_size < self.crop_size:
            img = ImageOps.expand(img, border=(padw//2 + 1, padh//2 + 1, padw//2 + 1, padh//2 + 1), fill=self.fill_image)
            mask = ImageOps.expand(mask, border=(padw//2 + 1, padh//2 + 1, padw//2 + 1, padh//2 + 1), fill=self.fill_label)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        ## TODO: unnecessary code
#         minrange = [max(x1 - padw//2 - 1, 0), max(y1 - padh//2 - 1, 0)]
#         maxrange = [min(x1 + 2 * self.crop_size - padw//2 - 1 - w, self.crop_size - 1), 
#                     min(y1 + 2 * self.crop_size - padh//2 - 1 - h, self.crop_size - 1)]
        
        return img, mask # , minrange, maxrange
    

    def _generate_triplet(self, lbl):
        lbl_view = lbl[0:513, 0:513]
        
        options = np.nonzero(lbl_view.reshape(-1) != 255)[0]
        
        if options.shape[0] > 0:
            ai = np.random.randint(low=0, 
                                    high=options.shape[0], 
                                    size=(self.n_triplets,))
            ai = options[ai]
        else:
            ai = np.array([0] * self.n_triplets, dtype=np.int64)
        

        classes, inv_map = np.unique(lbl_view, return_inverse=True)
        n_classes = len(classes)
        inv_map = inv_map.reshape(lbl_view.shape[0], lbl_view.shape[1])
        inv_map_flat = inv_map.reshape(-1)

        class_lookup = (np.arange(n_classes, dtype=np.int32).reshape((1, 1, n_classes)) !=
                        inv_map.reshape(lbl_view.shape[0], lbl_view.shape[1], 1))
        class_lookup = np.transpose(class_lookup, axes=[2, 0, 1])
        
        lbl_view_flat = lbl_view.reshape(-1)
        
        lneg = []
        lpos = []
        for i in range(n_classes):
            lneg.append(np.transpose(np.logical_and(lbl_view != 255, class_lookup[i]).reshape(-1).nonzero()).reshape((-1)))
            lpos.append(np.transpose(
                                np.logical_and(lbl_view != 255, 
                                               np.logical_not(class_lookup[i])).reshape(-1).nonzero()).reshape((-1)))
            
        ni, pi = [], []
        for i in range(self.n_triplets):
            cni = lneg[inv_map_flat[ai[i]]][lneg[inv_map_flat[ai[i]]] != ai[i]] 
            cpi = lpos[inv_map_flat[ai[i]]][lpos[inv_map_flat[ai[i]]] != ai[i]]
            
            for _ in range(self.samples_pos):
                if len(cni) == 0 or len(cpi) == 0:
                    #ni.append(ai[i])
                    pi.append(ai[i])
                else:
                    #ni.append( np.random.choice(cni))
                    pi.append( np.random.choice(cpi))
                    
            for _ in range(self.samples_neg):
                if len(cni) == 0 or len(cpi) == 0:
                    ni.append(ai[i])
                    #pi.append(ai[i])
                else:
                    ni.append( np.random.choice(cni))
                    #pi.append( np.random.choice(cpi))
            
        #aix, aiy = np.unravel_index(ai, dims=(lbl_view.shape[0], lbl_view.shape[1]))
        #aix += minrange[0]
        #aiy += minrange[1]
        #ai = np.stack((aix, aiy))
        
        #pix, piy = np.unravel_index(pi, dims=(lbl_view.shape[0], lbl_view.shape[1]))
        #pix += minrange[0]
        #piy += minrange[1]
        #pi = np.stack((pix, piy))
        
        #nix, niy = np.unravel_index(ni, dims=(lbl_view.shape[0], lbl_view.shape[1]))
        #nix += minrange[0]
        #niy += minrange[1]
        #ni = np.stack((nix, niy))
        
        #triplets = np.stack((ai, pi, ni), axis=0)
        pi =  np.array(pi, dtype=np.int64)
        ni =  np.array(ni, dtype=np.int64)
        
        ai = torch.tensor(ai.reshape(self.n_triplets), dtype=torch.long)
        pi = torch.tensor(pi.reshape(-1, self.samples_pos), dtype=torch.long)
        ni = torch.tensor(ni.reshape(-1, self.samples_neg), dtype=torch.long)
        
        return ai, pi, ni
    


    