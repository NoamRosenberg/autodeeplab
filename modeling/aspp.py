import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from operations import ABN


def SeparateConv(C_in, C_out, kernel_size, stride, padding, dilation, bias, BatchNorm):
    return nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                                   padding=padding, dilation=dilation, groups=C_in, bias=False),
                         BatchNorm(C_in),
                         nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
                         )


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, separate=False):
        super(_ASPPModule, self).__init__()
        if separate:
            self.atrous_conv = SeparateConv(inplanes, planes, kernel_size, 1, padding, dilation, False, BatchNorm)
        else:
            self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                         stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ASPP_train(nn.Module):
    def __init__(self, backbone, output_stride, filter_multiplier=20, steps=5, BatchNorm=ABN, separate=False):
        super(ASPP_train, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'autodeeplab':
            inplanes = int(filter_multiplier * steps * (output_stride / 4))
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm, separate=separate)
        self.aspp3 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm, separate=separate)
        self.aspp4 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm, separate=separate)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(
                                                 inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        return self.dropout(x)
        # return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def build_aspp(backbone, output_stride, BatchNorm, args, separate):
    return ASPP_train(backbone, output_stride, args.filter_multiplier, 5, BatchNorm, separate)
