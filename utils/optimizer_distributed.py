# #!/usr/bin/python
# # -*- encoding: utf-8 -*-
#
#
# import torch
# import logging
# import torch.distributed as dist
#
# logger = logging.getLogger()
#
#
# class Optimizer(object):
#     def __init__(self, model, args, max_iteration):
#         if hasattr(model, 'module'):
#             wd_params, non_wd_params = model.module.get_params()
#         else:
#             wd_params, non_wd_params = model.get_params()
#         params_list = [{'params': wd_params, },
#                        {'params': non_wd_params, 'weight_decay': 0}]
#         self.warmup_steps = args.warmup_iters
#         self.warmup_start_lr = args.warmup_start_lr
#         self.lr0 = args.base_lr
#         self.lr = self.lr0
#         self.max_iter = float(max_iteration)
#         self.power = 0.9
#         self.it = 0
#         self.optim = torch.optim.SGD(
#             params_list,
#             lr=self.lr0,
#             momentum=0.9,
#             weight_decay=5e-4)
#         self.warmup_factor = (self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)
#
#     def get_lr(self):
#         if self.it <= self.warmup_steps:
#             lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
#         else:
#             factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)) ** self.power
#             lr = self.lr0 * factor
#         return lr
#
#     def step(self):
#         self.lr = self.get_lr()
#         for pg in self.optim.param_groups:
#             pg['lr'] = self.lr
#         self.optim.param_groups[0]['lr'] = self.lr
#         self.it += 1
#         self.optim.step()
#         if self.it == self.warmup_steps + 2:
#             print('==> warmup done, start to implement poly lr strategy')
#
#     def zero_grad(self):
#         self.optim.zero_grad()


#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging
import torch.distributed as dist


logger = logging.getLogger()


class Optimizer(object):
    def __init__(self, model, lr0, momentum, wd, warmup_steps, warmup_start_lr, max_iter, power):

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        self.optim = torch.optim.SGD(
            model.parameters(),
            lr=lr0,
            momentum=momentum,
            weight_decay=wd)
        self.warmup_factor = (
            self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)

    def get_lr(self):

        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
        else:
            factor = (1 - (self.it - self.warmup_steps) /
                      (self.max_iter - self.warmup_steps)) ** self.power
            lr = self.lr0 * factor
        return lr

    def step(self):

        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            pg['lr'] = self.lr
        self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2 and dist.get_rank() == 0:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def load_state_dict(self, optimizer, iteration):
        self.it = iteration if iteration is not None else 0
        self.optim.load_state_dict(optimizer)

    def zero_grad(self):
        self.optim.zero_grad()
