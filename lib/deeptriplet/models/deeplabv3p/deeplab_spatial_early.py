import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .backbone import build_backbone
from .decoder_spatial_early import build_decoder_spatial_early

class DeepLabSpatialEarly(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, dynamic_coordinates=False, spatial_size=(129, 129)):
        super(DeepLabSpatialEarly, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder_spatial_early(num_classes, backbone, BatchNorm, dynamic=dynamic_coordinates, size=spatial_size)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                            
    def init_from_semseg_model(self,d):
        del d['state_dict']['decoder.last_conv.8.weight']
        del d['state_dict']['decoder.last_conv.8.bias']
        d['state_dict']['decoder.last_conv.0.weight'] = torch.cat((d['state_dict']['decoder.last_conv.0.weight'], 
                                                                   torch.Tensor(256,2,3,3).normal_(std=2e-4).cuda()), 
                                                                  dim=1)
        self.load_state_dict(d['state_dict'], strict=False)


if __name__ == "__main__":
    model = DeepLabSpatialEarly(num_classes=32, backbone='resnet', output_stride=16, dynamic_coordinates=True)
    model.eval().cuda()
    input = torch.rand(1, 3, 200, 513).cuda()
    output = model(input)
    print(output.size())


