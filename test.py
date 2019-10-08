import torch.nn as nn
# from modeling.backbone.resnet import ResNet101
from auto_deeplab import AutoDeeplab
import warnings
from config_utils.search_args import obtain_search_args
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

#
# warnings.filterwarnings('ignore')
# from retrain_model.build_autodeeplab import Train_Autodeeplab
#
#
# def total_params(model, log=True):
#     params = sum(p.numel() / 1000.0 for p in model.parameters())
#     if log:
#         print(">>> total params: {:.2f}K".format(params))
#     return params
#
#
# def each_param(model):
#     for p in model.parameters():
#         print(">>> {:.5f}K".format(p.numel()))
#
#
import torch
from retrain_model.new_model import get_default_net
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

args = obtain_retrain_autodeeplab_args()
args.num_classes = 19
model = Retrain_Autodeeplab(args)

x = torch.randn(2, 3, 129, 129)
x = model(x)
print(x.shape)
