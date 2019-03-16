import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DecoderSpatialEarly(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, dynamic=False, size=(129, 129)):
        super(DecoderSpatialEarly, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
            
        self.size = size
        self.dynamic = dynamic

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        # 304 -> 306 for two additional spatial channels
        self.last_conv = nn.Sequential(nn.Conv2d(306, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        x = np.linspace(-1., 1., self.size[1], dtype=np.float32)
        y = np.linspace(-1., 1., self.size[0], dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        spatial_channels = np.append(np.expand_dims(xx,axis=0), np.expand_dims(yy,axis=0), axis=0)
        spatial_channels = np.expand_dims(spatial_channels, axis=0)

        self.spatial_channels = torch.tensor(spatial_channels, device='cuda:0')
    
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, low_level_feat), dim=1)
        
        if self.dynamic:
            batch_size = x.shape[0]
            
            xc = np.linspace(-x.shape[3] / 129., x.shape[3] / 129., x.shape[3], dtype=np.float32)
            yc = np.linspace(-x.shape[2] / 129., x.shape[2] / 129., x.shape[2], dtype=np.float32)
            xx, yy = np.meshgrid(xc, yc)

            spatial_channels = np.append(np.expand_dims(xx,axis=0), np.expand_dims(yy,axis=0), axis=0)
            spatial_channels = np.expand_dims(spatial_channels, axis=0)

            self.spatial_channels = torch.tensor(spatial_channels, device='cuda:0')
            
            x = torch.cat((x, self.spatial_channels.expand(batch_size, -1, -1, -1)), dim=1)
        else:
            batch_size = x.shape[0]
            
            x = torch.cat((x, self.spatial_channels.expand(batch_size, -1, -1, -1)), dim=1)
        
        
        x = self.last_conv(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder_spatial_early(num_classes, backbone, BatchNorm, dynamic=False, size=(129, 129)):
    return DecoderSpatialEarly(num_classes, backbone, BatchNorm, dynamic=dynamic, size=size)