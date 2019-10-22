import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from operations import ABN, NaiveBN


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=19,
                 use_ABN=True, freeze_bn=False, args=None, separate=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if use_ABN:
            BatchNorm = ABN
        else:
            BatchNorm = NaiveBN

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, args)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, args, separate)
        self.decoder = build_decoder(
            num_classes, backbone, BatchNorm, args, separate)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input_feature):
        x, low_level_feat = self.backbone(input_feature)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_feature.shape[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, ABN):
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


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
