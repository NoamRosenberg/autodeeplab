#!/usr/bin/python
# -*- encoding: utf-8 -*-


class Config(object):
    def __init__(self):
        ## model and loss
        self.ignore_label = 255
        self.aspp_global_feature = False
        # dataset
        self.n_classes = 19
        self.datapth = '/dataset/Cityscapes_dataset'
        # self.datapth = r'E:\BaiduNetdiskDownload\cityscapes'
        self.gpus = 8
        self.crop_size = (769, 769)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # optimizer
        self.warmup_steps = 1000
        self.warmup_start_lr = 5e-6
        self.lr_start = 3e-2
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr_power = 0.9
        self.max_iter = 41000
        self.max_epoch = 8000
        # training control
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        self.flip = True
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.ims_per_gpu = 2
        self.n_workers = 2 * self.gpus
        self.msg_iter = 50
        self.ohem_thresh = 0.7
        self.respth = './log'
        self.port = 32168
        # eval control
        self.eval_batchsize = 2
        self.eval_n_workers = 2
        self.eval_scale = (1.0,)
        self.eval_scales = (0.25,0.5,0.75,1,1.25,1.5)
        self.eval_flip = True


config_factory = {'resnet_cityscapes': Config(), }
