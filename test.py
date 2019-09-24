import torch.nn as nn
# from modeling.backbone.resnet import ResNet101
from auto_deeplab import AutoDeeplab
import warnings
from config_utils.search_args import obtain_search_args
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

warnings.filterwarnings('ignore')
from retrain_model.build_autodeeplab import Train_Autodeeplab


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

    args = obtain_retrain_autodeeplab_args()
    model = Train_Autodeeplab(19, args).cuda()
    x = torch.randn(2, 3, 64, 64).cuda()
    print(model(x).shape)
