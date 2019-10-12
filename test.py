import torch.nn as nn
# from modeling.backbone.resnet import ResNet101
from auto_deeplab import AutoDeeplab
import warnings
from config_utils.search_args import obtain_search_args
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

#
#
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from config_utils.search_args import obtain_search_args
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.optimizer_distributed import Optimizer
from auto_deeplab import AutoDeeplab

#
# # model(torch.randn(2, 3, 129, 129))

# from decoding_formulas import Decoder
#
# checkpoint = torch.load(r'H:\CVPR2019\result\naive\checkpoint.pth.tar')
# betas = checkpoint['state_dict']['betas']
# alphas = checkpoint['state_dict']['alphas']
# print(betas)
# print(alphas)
# decoder = Decoder(alphas, betas, 5)

#
args = obtain_search_args()

args.num_classes = 19
# model = Retrain_Autodeeplab(args)
model = AutoDeeplab(19, 12, args=args,filter_multiplier=4)

grads = {}
#
#
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook
#
#
# # x = torch.randn(1, requires_grad=True)
# # y = 3 * x
# # z = y * y
#

model = model.cuda()
x = torch.randn(2, 3, 65, 65).cuda()
output = model(x)
output1 = torch.mean(output)
# 为中间变量注册梯度保存接口，存储梯度时名字为 y。
model.betas.register_hook(save_grad('beta'))
model.alphas.register_hook(save_grad('alpha'))
model.beta_0.register_hook(save_grad('beta_0'))

# 反向传播
output1.backward()

# 查看 y 的梯度值
# print(grads['beta'].detach().cpu().numpy())
# print(grads['alpha'])
print(grads['beta'])
torch.save(grads['beta'].detach().cpu().numpy(),'beta.npy')
np.save('beta.npy',grads['beta'].detach().cpu().numpy())



# import numpy as np
# beta = np.load('beta.npy')

