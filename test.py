import warnings
import torch
from auto_deeplab import AutoDeeplab
warnings.filterwarnings('ignore')

model = AutoDeeplab(19, 12).cuda()

criterion = torch.nn.MSELoss().cuda()

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


x = torch.randn((2, 3, 64, 64), requires_grad=True).cuda()

y = model(x)

print(y.shape)


label = torch.randn((2, 19, 64, 64)).cuda()

z = criterion(y, label)
# 为中间变量注册梯度保存接口，存储梯度时名字为 y。
model.betas.register_hook(save_grad('y'))

# 反向传播
z.backward()

# 查看 y 的梯度值
print(grads['y'])
print(grads['y'].shape)
