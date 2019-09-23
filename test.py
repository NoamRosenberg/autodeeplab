import torch.nn as nn
# from modeling.backbone.resnet import ResNet101
import argparse
from utils.copy_state_dict import copy_state_dict
from modeling.deeplab import DeepLab
from auto_deeplab import AutoDeeplab
from thop import profile
from torchsummary import summary
import warnings
from new_model import get_default_net
from config_utils.search_args import obtain_search_args
warnings.filterwarnings('ignore')


def total_params(model, log=True):
    params = sum(p.numel() / 1000.0 for p in model.parameters())
    if log:
        print(">>> total params: {:.2f}K".format(params))
    return params


def each_param(model):
    for p in model.parameters():
        print(">>> {:.5f}K".format(p.numel()))


if __name__ == "__main__":
    import torch
    import time

    args = obtain_search_args()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model = AutoDeeplab(num_classes=19, num_layers=12, criterion=criterion, filter_multiplier=args.filter_multiplier,
                        block_multiplier=args.block_multiplier, step=args.step, args=args).cuda()
    input = torch.randn(2, 3, 65, 65).cuda()
    output = model(input)
    print(output)
