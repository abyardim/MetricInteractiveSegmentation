
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d

# generate embedding model
class FCN8Embedding(nn.Module):
    def __init__(self, embedding_dim=30, learned_billinear=True):
        super(FCN8Embedding, self).__init__()
        self.learned_billinear = learned_billinear
        self.embedding_dim = embedding_dim
        self.loss = functools.partial(cross_entropy2d,
                                      size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.embedding_layer = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.embedding_dim, 1),
        )

        self.embed_pool4 = nn.Conv2d(512, self.embedding_dim, 1)
        self.embed_pool3 = nn.Conv2d(256, self.embedding_dim, 1)

        if self.learned_billinear:
            self.upembed2 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, 4,
                                               stride=2, bias=False)
            self.upembed4 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, 4,
                                               stride=2, bias=False)
            self.upembed8 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, 16,
                                               stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(m.in_channels, 
                                                          m.out_channels, 
                                                          m.kernel_size[0]))


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        embed = self.embedding_layer(conv5)

        if self.learned_billinear:
            upembed2 = self.upembed2(embed)
            embed_pool4c = self.embed_pool4(conv4)[:, :, 5:5+upembed2.size()[2],
                                                         5:5+upembed2.size()[3]]
            upembed_pool4 = self.upembed4(upembed2 + embed_pool4c)
            
            embed_pool3c = self.embed_pool3(conv3)[:, :, 9:9+upembed_pool4.size()[2],
                                                         9:9+upembed_pool4.size()[3]]

            out = self.upembed8(embed_pool3c + upembed_pool4)[:, :, 31:31+x.size()[2],
                                                                    31:31+x.size()[3]]
            return out.contiguous()                  

        else:
            embed_pool4 = self.embed_pool4(conv4)
            embed_pool3 = self.embed_pool3(conv3)
            embed = F.upsample_bilinear(embed, embed_pool4.size()[2:])
            embed += embed_pool4
            embed = F.upsample_bilinear(embed, embed_pool3.size()[2:])
            embed += embed_pool3
            out = F.upsample_bilinear(embed, x.size()[2:])

        return out
        