import torch.nn as nn
# from modeling.backbone.resnet import ResNet101
from auto_deeplab import AutoDeeplab
import warnings
from config_utils.search_args import obtain_search_args
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

#
#
import torch
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.optimizer_distributed import Optimizer

args = obtain_retrain_autodeeplab_args()

args.num_classes = 19
model = Retrain_Autodeeplab(args)
x = torch.randn(2, 3, 129, 129)
model(x)

