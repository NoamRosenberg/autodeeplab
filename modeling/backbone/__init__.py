from modeling.backbone import resnet, xception, drn, mobilenet
from retrain_model.new_model import get_default_net

def build_backbone(backbone, output_stride, BatchNorm, args):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'autodeeplab':
        return get_default_net(filter_multiplier = args.filter_multiplier)
    else:
        raise NotImplementedError
