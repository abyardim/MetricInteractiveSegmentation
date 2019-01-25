
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL
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


class PascalDataset(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    """

    def __init__(
            self,
            *,
            pascal_root,
            split_file,
            normalize_imagenet=False,
            augment=False,
            pad_zeros=False,
            downsample_label=1,
            scale_low=0.5,
            scale_high=1.5
    ):
        self.split_file = split_file
        self.pascal_root = pascal_root

        self.scale_low = scale_low
        self.scale_high = scale_high

        self.normalize_imagenet = normalize_imagenet

        self.downsample_label = downsample_label

        self.augment = augment
        self.pad_zeros = pad_zeros

        self.n_classes = 21

        self.image_list, self.label_list = read_labeled_image_list(self.pascal_root, self.split_file)

        if self.normalize_imagenet:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]
        lbl_path = self.label_list[index]

        img = PIL.Image.open(im_path)
        lbl = PIL.Image.open(lbl_path)

        ## augmentation
        if self.augment:
            img, lbl = self._augment(img, lbl)
        elif self.downsample_label > 1:
            lbl = lbl.resize((math.ceil(lbl.width / self.downsample_label),
                              math.ceil(lbl.height / self.downsample_label)))

        img = np.array(img, dtype=np.float32) / 255.0
        lbl = np.array(lbl, dtype=np.long)
        #         lbl[lbl==255] = 0

        if self.pad_zeros:
            img, lbl = self._pad_zeros(img, lbl)

        img = self.transforms(img)
        lbl = torch.from_numpy(lbl)

        return img, lbl

    def _augment(self, img, lbl):

        if np.random.rand() > 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        scale = np.random.uniform(low=self.scale_low, high=self.scale_high)
        target_size = (int(round(img.width * scale)),
                       int(round(img.height * scale)))

        img = img.resize(target_size, resample=PIL.Image.BILINEAR)
        lbl = lbl.resize((math.ceil(target_size[0] / self.downsample_label),
                          math.ceil(target_size[1] / self.downsample_label)))

        return img, lbl

    def _pad_zeros(self, img, lbl):

        th, tw = 513, 513
        thl, twl = (math.ceil(513 / self.downsample_label), math.ceil(513 / self.downsample_label))

        h, w = img.shape[0], img.shape[1]

        if w > tw:
            i = random.randint(0, w - tw)
            img = img[:, i:i + tw, :]

            i = round(i / self.downsample_label)
            lbl = lbl[:, i:i + twl]

        if h > th:
            j = random.randint(0, h - th)
            img = img[j:j + th, :, :]

            j = round(j / self.downsample_label)
            lbl = lbl[j:j + thl, :]

        h, w = img.shape[0], img.shape[1]
        hl, wl = lbl.shape[0], lbl.shape[1]

        if self.normalize_imagenet:
            img_padded = np.zeros((th, tw, 3), dtype=np.float32)
            img_padded[:, :, 0] = 0.485
            img_padded[:, :, 1] = 0.456
            img_padded[:, :, 2] = 0.406
        else:
            img_padded = np.zeros((th, tw, 3), dtype=np.float32)

        lbl_padded = np.ones((thl, twl), dtype=np.long) * 255

        start_h = (th - h) // 2
        start_w = (tw - w) // 2

        start_hl = (thl - hl) // 2
        start_wl = (twl - wl) // 2

        img_padded[start_h:start_h+h, start_w:start_w+w, :] = img
        lbl_padded[start_hl:start_hl+hl, start_wl:start_wl+wl] = lbl

        return img_padded, lbl_padded

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

