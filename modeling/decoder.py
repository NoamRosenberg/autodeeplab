import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def SeparateConv(C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, bias=False):
    return nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                                   padding=padding, dilation=dilation, groups=C_in, bias=False),
                         nn.BatchNorm2d(C_in),
                         nn.ReLU(),
                         nn.Conv2d(C_in, C_out, kernel_size=1,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(C_out),
                         nn.ReLU()
                         )


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, args, separate):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24

        elif backbone == 'autodeeplab':
            low_level_inplanes = args.filter_multiplier * args.steps
        else:
            raise NotImplementedError

        self.conv_feature = nn.Conv2d(
            low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.feature_projection = nn.Sequential(
            self.conv_feature, self.bn1, self.relu)
        concate_channel = 48 + 256
        if separate == True:
            self.conv1 = nn.Sequential(SeparateConv(concate_channel, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.Dropout(0.5))
            self.conv2 = nn.Sequential(SeparateConv(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.Dropout(0.1))

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(concate_channel, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.last_linear = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1)

        self._init_weight()

    def forward(self, x, low_level_feat):

        low_level_feat = self.feature_projection(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[
            2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        # x = self.last_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.last_linear(x)
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


def build_decoder(num_classes, backbone, BatchNorm, args, separate):
    return Decoder(num_classes, backbone, BatchNorm, args, separate)
