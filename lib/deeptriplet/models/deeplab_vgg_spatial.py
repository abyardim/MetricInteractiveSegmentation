
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import numpy as np

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class DeepLab_VGG_Spatial(nn.Module):

    def __init__(self, version="largefov", n_classes=21, size=(65,65), dynamic=False):
        super(DeepLab_VGG_Spatial, self).__init__()

        self.n_classes = n_classes
        self.size = size
        self.dynamic = dynamic

        if version == "largefov":
            self.use_aspp = False
#         elif version == "aspp-s":
#             self.use_aspp = True
#             dilations = [2, 4, 8, 12]
#         elif version == "aspp-l":
#             self.use_aspp = True
#             dilations = [6, 12, 18, 24]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
        )

        if self.use_aspp:
            pass
#             self.aspp_branches = []

#             for d in dilations:
#                 fc6 = nn.Sequential(
#                     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=d, padding=d),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout2d()
#                 )

#                 fc7 = nn.Sequential(

#                     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout2d(),
#                 )

#                 fc8 = nn.Conv2d(in_channels=1024, out_channels=n_classes, kernel_size=1)

#                 self.aspp_branches.append(nn.Sequential(fc6, fc7, fc8))

#             self.aspp_branches = nn.ModuleList(self.aspp_branches)
        else:
            self.fc6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=12, padding=12),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )

            self.fc7 = nn.Sequential(

                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )
            
            x = np.linspace(-1., 1., self.size[1], dtype=np.float32)
            y = np.linspace(-1., 1., self.size[0], dtype=np.float32)
            xx, yy = np.meshgrid(x, y)

            spatial_channels = np.append(np.expand_dims(xx,axis=0), np.expand_dims(yy,axis=0), axis=0)
            spatial_channels = np.expand_dims(spatial_channels, axis=0)
            
            self.spatial_channels = torch.tensor(spatial_channels, device='cuda:0')
            
            self.fc8 = nn.Conv2d(in_channels=1026, out_channels=n_classes, kernel_size=1)

    def forward(self, input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if not self.use_aspp:
            x = self.fc6(x)
            x = self.fc7(x)
            
            if self.dynamic:
                batch_size = x.shape[0]
                dim1 = x.shape[2]
                dim2 = x.shape[3]
                
                cx = np.linspace(-dim2 / 65., dim2 / 65., dim2, dtype=np.float32)
                cy = np.linspace(-dim1 / 65., dim1 / 65., dim1, dtype=np.float32)
                xx, yy = np.meshgrid(cx, cy)

                spatial_channels = np.append(np.expand_dims(xx,axis=0), np.expand_dims(yy,axis=0), axis=0)
                spatial_channels = np.expand_dims(spatial_channels, axis=0)

                self.spatial_channels = torch.tensor(spatial_channels, device='cuda:0')

                x = torch.cat((x, self.spatial_channels.expand(batch_size, -1, -1, -1)), 1)
                x = self.fc8(x)
            else:
                batch_size = x.shape[0]
                x = torch.cat((x, self.spatial_channels.expand(batch_size, -1, -1, -1)), 1)
                x = self.fc8(x)

            return x
        else:
            out0 = self.aspp_branches[0](x)
            out1 = self.aspp_branches[1](x)
            out2 = self.aspp_branches[2](x)
            out3 = self.aspp_branches[3](x)

            return out0 + out1 + out2 + out3

    def init_parameters(self, pretrain_vgg16_dict):
        """
        Load VGG parameters from model dict

        Can be used for loading VGG pretrained on ImageNet.
        Note: Image are expected to be normalized by
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        and images have to be loaded in to a range of [0, 1]
        as stated in the documentation https://pytorch.org/docs/stable/torchvision/models.html
        """

        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        features = list(pretrain_vgg16_dict.items())
        i = 0

        for idx, conv_block in enumerate(conv_blocks):
            for l1 in conv_block:
                if isinstance(l1, nn.Conv2d):
                    #                     print(i)
                    #                     print(l1.weight.size())
                    #                     print(features[i][1].size())
                    #                     print(l1.bias.size())
                    #                     print(features[i+1][1].size())
                    assert l1.weight.size() == features[i][1].size()
                    assert l1.bias.size() == features[i + 1][1].size()

                    l1.weight.data.copy_(features[i][1].data)
                    l1.bias.data.copy_(features[i + 1][1].data)

                    i += 2

    def init_vgg_imagenet(self):
        self.init_parameters(model_zoo.load_url(model_urls['vgg16']))

    def get_parameter_group(self, bias, final):
        if final:
            if self.use_aspp:
                for b in self.aspp_branches:
                    for m1 in b.modules():
                        if isinstance(m1, nn.Conv2d):
                            if bias:
                                yield m1.bias
                            else:
                                yield m1.weight

            else:
                for b in [self.fc6, self.fc7, self.fc8]:
                    for m in b.modules():
                        if isinstance(m, nn.Conv2d):
                            if bias:
                                yield m.bias
                            else:
                                yield m.weight
        else:
            conv_blocks = [self.conv1,
                           self.conv2,
                           self.conv3,
                           self.conv4,
                           self.conv5]

            for m1 in conv_blocks:
                for m2 in m1.modules():
                    if isinstance(m2, nn.Conv2d):
                        if bias:
                            yield m2.bias
                        else:
                            yield m2.weight

    def get_parameter_group_v2(self, bias, final):
        if final:
            if self.use_aspp:
                for b in self.aspp_branches:
                    for m1 in b.modules():
                        if isinstance(m1, nn.Conv2d) and m1.out_channels == self.n_classes:
                            if bias:
                                yield m1.bias
                            else:
                                yield m1.weight

            else:
                for b in [self.fc6, self.fc7, self.fc8]:
                    for m in b.modules():
                        if isinstance(m, nn.Conv2d) and m.out_channels == self.n_classes:
                            if bias:
                                yield m.bias
                            else:
                                yield m.weight
        else:
            conv_blocks = [self.conv1,
                           self.conv2,
                           self.conv3,
                           self.conv4,
                           self.conv5]

            for m1 in conv_blocks:
                for m2 in m1.modules():
                    if isinstance(m2, nn.Conv2d):
                        if bias:
                            yield m2.bias
                        else:
                            yield m2.weight

            if self.use_aspp:
                for b in self.aspp_branches:
                    for m1 in b.modules():
                        if isinstance(m1, nn.Conv2d) and m1.out_channels != self.n_classes:
                            if bias:
                                yield m1.bias
                            else:
                                yield m1.weight

            else:
                for b in [self.fc6, self.fc7, self.fc8]:
                    for m in b.modules():
                        if isinstance(m, nn.Conv2d) and m.out_channels != self.n_classes:
                            if bias:
                                yield m.bias
                            else:
                                yield m.weight

